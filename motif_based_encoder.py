import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import os
import random

gb = torch.tensor([0.25, 0.25, 0.25, 0.25])
BASE_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}  # bases are of the order ACGT
REVERSE_ORDER = [3, 2, 1, 0]


# Helper functions
def one_hot_encode_seq(seq: str, base_to_index) -> torch.tensor:
    """output is a tensor of shape (4, seq_length) where which base each row corresponds to is determined by
    <base_to_index>."""
    seq_length = len(seq)
    one_hot_seq = torch.zeros(len(base_to_index), seq_length, dtype=torch.float32)
    for index, base in enumerate(seq.upper()):
        one_hot_seq[base_to_index[base], index] = 1
    return one_hot_seq


def pad_one_hot_encoded_seq(one_hot_seq, target_length) -> torch.tensor:
    """<one_hot_seq> is of shape (4, seq_length). Return a right-padded sequence to reach <target_length>."""
    seq_length = one_hot_seq.shape[1]
    if seq_length == target_length:  # Do nothing
        return one_hot_seq
    elif seq_length < target_length:
        pad_length = target_length - seq_length
        padding = torch.zeros(one_hot_seq.shape[0], pad_length)
        padded_one_hot_dna = torch.cat([one_hot_seq, padding], dim=1)
        return padded_one_hot_dna
    else:  # seq_length > target_length
        raise ValueError("This function cannot handle an input sequence whose length is greater than target length.")


def rev_comp_one_hot_encoded_dna(one_hot_dna, reverse_order) -> torch.tensor:
    """<one_hot_dna> is of shape (batch_size, 4, seq_length)"""
    # Reverse the sequence (columns)
    rev = torch.flip(one_hot_dna, dims=[2])

    # Get the complement of the bases (rows)
    complement_indices = torch.tensor(reverse_order)
    rev_comp = rev[:, complement_indices, :]

    return rev_comp


# Creating the PWM constraint for the Conv1d layer
def pwm_constraint_conv1d(conv_weights, gb):
    """conv_weights would be of shape (num_of_PWMs, num_of_bases, PWM_width), example (256, 4, 15)
    gb is of shape (num_bases,), where genomic background probabilities = [0.25, 0.25, 0.25, 0.25]"""

    # log2 background (from shape (num_bases,) to (1, num_bases, 1) for correct broadcasting)
    log_gb = torch.log2(gb).view(1, -1, 1)

    # Compute log2 sum over 2^conv_weights per PWM (over bases) (shape = (num_PWMs, 1, PWM_width))
    log_sum_pow_2_col = torch.log2(torch.pow(2, conv_weights).sum(dim=1, keepdim=True))

    # Apply the PWM constraint (broadcast shape = (num_PWMs, num_bases, PWM_width))
    PWM_constraints = conv_weights - log_gb - log_sum_pow_2_col

    # Convert to probabilities (shape = (num_PWMs, num_bases, PWM_width))
    PWM_constraints_probs = gb.view(1, -1, 1) * torch.pow(2, PWM_constraints)

    return PWM_constraints_probs


# Creating the trainable scaling layer
class TrainableScaling(nn.Module):
    def __init__(self, num_PWMs):
        """Both the scale and bias are of size <num_PWMs> and are randomly initialized to values between 0 and 1."""
        super(TrainableScaling, self).__init__()
        self.num_PWMs = num_PWMs

        # Create the trainable scale parameter
        self.scale = nn.Parameter(torch.rand(1, num_PWMs), requires_grad=True)

        # Create the trainable bias parameter
        self.bias = nn.Parameter(torch.rand(1, num_PWMs), requires_grad=True)

    def forward(self, inputs):
        """inputs are of shape (batch_size, num_PWMs. seq_length_after_conv)"""
        # Let the scale and bias parameters match the shape of the inputs (shape = (1, num_PWMs, seq_length_after_conv))
        scale = self.scale.unsqueeze(2).repeat(1, 1, inputs.shape[2])
        bias = self.bias.unsqueeze(2).repeat(1, 1, inputs.shape[2])

        # Apply the scaling and bias parameters
        scaled_inputs = (scale * inputs) - bias

        return scaled_inputs


# Creating the trainable pooling layer
class TrainablePooling(nn.Module):
    def __init__(self, num_PWMs):
        """The pooling parameter alpha is of size <num_PWMs> and is randomly initialized to values between 0 and 1."""
        super(TrainablePooling, self).__init__()
        self.num_PWMs = num_PWMs

        # Create the trainable pooling parameter
        self.pooling = nn.Parameter(torch.rand(1, num_PWMs), requires_grad=True)

    def forward(self, inputs):
        """inputs is of shape (batch_size, num_PWMs. seq_length_after_conv) output is of shape (batch_size, num_PWMs)"""
        # Let the pooling parameter match the shape of the input
        alpha = torch.diag(self.pooling.view(-1))  # shape (num_PWMs, num_PWMs)
        alpha = alpha.unsqueeze(0)  # shape (1, num_PWMs, num_PWMs)

        # Calculate the pooling weights w
        w = torch.matmul(alpha, inputs)  # shape of w (batch_size, num_PWMs, seq_length_after_conv)
        w = F.softmax(w, dim=2)

        # Use the pooling weights <w> to aggregate <inputs> across <seq_length_after_conv>
        pooled_inputs = torch.mul(w, inputs)
        pooled_inputs = pooled_inputs.sum(dim=2)  # shape (batch_size, num_PWMs)

        return pooled_inputs


# Creating the Motif Interactions (attention) layer
class TrainableMotifInteractions(nn.Module):
    def __init__(self, num_PWMs):
        """The motif interactions matrix is of shape (num_PWMs, num_PWMs) and is randomly initialized to values between
        0 and 1."""
        super(TrainableMotifInteractions, self).__init__()
        self.num_PWMs = num_PWMs

        # Create the trainable motif interactions (attention) matrix
        self.motif_interactions = nn.Parameter(torch.rand(num_PWMs, num_PWMs), requires_grad=True)

    def forward(self, inputs):
        """inputs is of shape (batch_size, num_PWMs). output is of shape (batch_size, num_PWMs)."""
        # Calculate the interaction-dependent weight w
        w = torch.sigmoid(torch.matmul(inputs, self.motif_interactions))  # shape (batch_size, num_PWMs)

        # Determine each motif contribution using <w> (element-wise multiplication)
        output = torch.mul(w, inputs)  # shape (batch_size, num_PWMs)

        return output


# Creating the motif-based encoder
class MotifBasedEncoder(nn.Module):
    def __init__(self, num_PWMs=256, PWM_width=15, window=10, num_bases=4):
        super(MotifBasedEncoder, self).__init__()
        # Define the attributes of the encoder
        self.num_PWMs = num_PWMs
        self.PWM_width = PWM_width
        self.window = window
        self.num_bases = num_bases

        # Define the layers of the encoder
        self.PWMs_conv = nn.Conv1d(in_channels=self.num_bases, out_channels=self.num_PWMs, kernel_size=self.PWM_width,
                                   bias=False)
        self.window_pool = nn.MaxPool1d(kernel_size=self.window, stride=self.window)
        self.scaling_layer = TrainableScaling(self.num_PWMs)
        self.pooling_layer = TrainablePooling(self.num_PWMs)
        self.attention_layer = TrainableMotifInteractions(self.num_PWMs)
        self.batch_norm_layer = nn.BatchNorm1d(self.num_PWMs)  # gamma and beta parameters are trainable

    def forward(self, inputs):
        """inputs is of shape (batch_size, num_bases, seq_length) where the length of the sequences would
        have been appropriately padded before being fed into this encoder to <seq_length>"""
        # Get the reverse compliment of the input sequences
        rev_comp = rev_comp_one_hot_encoded_dna(inputs, REVERSE_ORDER)

        # Run both the input seqs and their reverse complements through the PWM convolutional layers
        inputs_conv = self.PWMs_conv(inputs)  # shape (batch_size, num_PWMs, seq_length-PWM_width+1)
        rev_comp_conv = self.PWMs_conv(rev_comp)  # same shape as <inputs_conv>

        # Reverse the order of scores for <rev_comp_inputs_conv> then take better score between the forward and
        # reverse at each position
        rev_order_rev_comp_conv = torch.flip(rev_comp_conv, dims=[2])
        conv_output = torch.maximum(inputs_conv, rev_order_rev_comp_conv)

        # To avoid counting overlaps, take best match in a <self.window> nt window
        conv_output = self.window_pool(conv_output)  # shape (batch_size, num_PWMs, seq_length_after_conv)

        # Apply the scaling layer
        scaled_output = torch.sigmoid(
            self.scaling_layer(conv_output))  # shape (batch_size, num_PWMs. seq_length_after_conv)

        # Apply the pooling layer
        pooled_output = self.pooling_layer(scaled_output)  # shape (batch_size, num_PWMs)

        # Apply the attention (Motif Interactions) layer followed by batch normalization
        output = self.batch_norm_layer(self.attention_layer(pooled_output))

        return output  # shape (batch_size, num_PWMs)
