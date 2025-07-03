import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import json

gb = torch.tensor([0.25, 0.25, 0.25, 0.25])


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
    def __init__(self, seq_length_after_conv, num_PWMs):
        """Both the scale and bias are of size <num_PWMs> and are randomly initialized to values between 0 and 1."""
        super(TrainableScaling, self).__init__()
        self.seq_length_after_conv = seq_length_after_conv
        self.num_PWMs = num_PWMs

        # Create the trainable scale parameter
        self.scale = nn.Parameter(torch.rand(1, num_PWMs), requires_grad=True)

        # Create the trainable bias parameter
        self.bias = nn.Parameter(torch.rand(1, num_PWMs), requires_grad=True)

    def forward(self, inputs):
        """inputs are of shape (batch_size, num_PWMs. seq_length_after_conv)"""
        # Let the scale and bias parameters match the shape of the inputs (shape = (1, num_PWMs, seq_length_after_conv))
        scale = self.scale.unsqueeze(2).repeat(1, 1, self.seq_length_after_conv)
        bias = self.bias.unsqueeze(2).repeat(1, 1, self.seq_length_after_conv)

        # Apply the scaling and bias parameters
        scaled_inputs = (scale * inputs) - bias

        return scaled_inputs


# Creating the trainable pooling layer
class TrainablePooling(nn.Module):
    def __init__(self, seq_length_after_conv, num_PWMs):
        """The pooling parameter alpha is of size <num_PWMs> and is randomly initialized to values between 0 and 1."""
        super(TrainablePooling, self).__init__()
        self.seq_length_after_conv = seq_length_after_conv
        self.num_PWMs = num_PWMs

        # Create the trainable pooling parameter
        self.pooling = nn.Parameter(torch.rand(1, num_PWMs), requires_grad=True)

    def forward(self, inputs):
        """inputs is of shape (batch_size, num_PWMs. seq_length_after_conv)
        output is of shape (batch_size, num_PWMs)"""
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
        """The motiff interactions matrix is of shape (num_PWMs, num_PWMs) and is randomly
        initialized to values between 0 and 1."""
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

