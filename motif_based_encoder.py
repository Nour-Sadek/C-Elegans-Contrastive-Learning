import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

COMPLEMENT_ORDER = torch.tensor([3, 2, 1, 0])


# Helper functions
def one_hot_encode_seq(seq: str, base_to_index: dict[str, int]) -> torch.tensor:
    """Return a tensor that is a one-hot encoded version of <seq> of shape (4, len(<seq>)) where which base each row
    corresponds to is determined by the <base_to_index> dictionary where the keys are nucleotides (strings) and the
    values are the index of each (int)."""

    seq_length = len(seq)
    one_hot_seq = torch.zeros(len(base_to_index), seq_length, dtype=torch.float32)
    for index, base in enumerate(seq.upper()):
        one_hot_seq[base_to_index[base], index] = 1
    return one_hot_seq


def pad_one_hot_encoded_seq(one_hot_seq: torch.tensor, target_length: int) -> torch.tensor:
    """Return a padded version of the <one_hot_seq> tensor, which would be the output of the <one_hot_encode_seq>
    function and of shape (4, seq_length), where either <one_hot_seq> is returned without any modifications if its
    seq_length is equal to <target_length or a right zero-padded version up to <target_length> is returned if
    seq_length < <target_length>.

    Raise a Value Error if <one_hot_seq>'s seq_length is greater than <target_length>."""

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


class ReverseComplement(nn.Module):
    """Defines a module whose action is to return the reverse complement of the one-hot encoded sequences in the input
    tensor following the complement order defined in <self.complement_order>.

    The input needs to be of shape (n, num_bases, seq_length) where n is the number of one-hot encoded sequences in the
    tensor, num_bases = 4 for nucleotides, and seq_length is the length of the sequences."""

    def __init__(self, complement_order: torch.tensor):
        super(ReverseComplement, self).__init__()
        self.complement_order = complement_order

    def forward(self, inputs):
        """Return the reverse complement of <inputs> which is of shape (batch_size * (family_size + 1), num_bases,
        seq_length), as in it the sequences are one-hot-encoded.

        <self.complement_order> determines the new order of the bases so that the new sequences would be the complement
        of those in <inputs>."""

        # Reverse the sequence (columns)
        inputs = torch.flip(inputs, dims=[2])

        # Get the complement of the bases
        inputs = inputs[:, self.complement_order, :]

        return inputs


class PWMConstraint(nn.Module):
    """Defines a module whose action is to return a PWM constrained version of the inputs; how that constraint is
    applied is done similarly to how it was done in Alan et al. 2025.

    The input needs to be of shape (num_PWMs, num_bases, PWM-width), and so each PWM's weights would be constrained so
    that the values reflect the values of a valid PWM."""

    def __init__(self, gb: torch.tensor):
        super(PWMConstraint, self).__init__()
        self.gb = gb

    def forward(self, inputs):
        """Return a version of the <inputs> tensor which abides by rules so that its values can be interpreted as those
        for a PWM of a transcription factor. The implementation of the transformation to apply that constraint follows
        what the Alan et al. 2025 paper did.

        <inputs> should be the weights of the convolutional layer in the <MotifBasedEncoder> class, and consequently it
        is of the shape (num_PWMs, num_bases, PWM_width)."""

        # log2 background (from shape (num_bases,) to (1, num_bases, 1) for correct broadcasting)
        log_gb = torch.log2(self.gb).view(1, -1, 1)

        # Compute log2 sum over 2^conv_weights per PWM (over bases) (shape = (num_PWMs, 1, PWM_width))
        log_sum_pow_2_col = torch.log2(torch.pow(2, inputs).sum(dim=1, keepdim=True))

        # Apply the PWM constraint (broadcast shape = (num_PWMs, num_bases, PWM_width))
        output = inputs - log_gb - log_sum_pow_2_col

        # Convert to probabilities (shape = (num_PWMs, num_bases, PWM_width))
        output = self.gb.view(1, -1, 1) * torch.pow(2, output)

        return output


# Creating the trainable scaling layer
class TrainableScaling(nn.Module):
    """Define a Trainable Scaling module similarly to how it was defined in Ali et al. 2023 paper where it applies a
    motif-based scale and offset to the convolutional output of each PWM scan.

    The input needs to be of shape (n, num_PWMs, seq_length_after_conv) where n is the number of original one-hot
    encoded tensors and seq_length_after_conv is the length of each sequence after it has been scanned by each PWM."""

    def __init__(self, num_PWMs: int):
        """Both the scale and bias are of size <num_PWMs> and are randomly initialized to values between 0 and 1. That
        initialization is changed in the <MotifBasedEncoder> class so that the values for the <scale> are 1 and those
        for <bias> are 0."""

        super(TrainableScaling, self).__init__()
        self.num_PWMs = num_PWMs

        # Create the trainable scale parameter
        self.scale = nn.Parameter(torch.rand(1, num_PWMs), requires_grad=True)

        # Create the trainable bias parameter
        self.bias = nn.Parameter(torch.rand(1, num_PWMs), requires_grad=True)

    def forward(self, inputs):
        """Return a scaled version of <inputs>, which is determined by the trainable <self.scale> and <self.bias>
        parameters, implemented similarly to how it was outlined in Ali et al. 2023 paper. <inputs> and the return
        value of the function are of shape (batch_size * (family_size + 1), num_PWMs, seq_length_after_conv)."""

        # Let the scale and bias parameters match the shape of the inputs (shape = (1, num_PWMs, seq_length_after_conv))
        scale = self.scale.unsqueeze(2).repeat(1, 1, inputs.shape[2])
        bias = self.bias.unsqueeze(2).repeat(1, 1, inputs.shape[2])

        # Apply the scaling and bias parameters
        scaled_inputs = (scale * inputs) - bias

        return scaled_inputs


# Creating the trainable pooling layer
class TrainablePooling(nn.Module):
    """Define a Trainable Pooling module similarly to how it was defined in Ali et al. 2023 paper where it applies a
    motif-based pooling weight to the convolutional layer output after scaling, allowing the sampling of all possible
    motif aggregation strategies from max to average pooling.

    The input needs to be of shape (n, num_PWMs, seq_length_after_conv), which again should be the outputs of the
    TrainableScaling layer.
    """

    def __init__(self, num_PWMs: int):
        """The pooling parameter alpha is of size <num_PWMs> and is randomly initialized to values between 0 and 1.
        That initialization is changed in the <MotifBasedEncoder> class so that the values are 1."""

        super(TrainablePooling, self).__init__()
        self.num_PWMs = num_PWMs

        # Create the trainable pooling parameter
        self.pooling = nn.Parameter(torch.rand(1, num_PWMs), requires_grad=True)

    def forward(self, inputs):
        """Return a pooled version of <inputs>, which is determined by the trainable <self.pooling> parameter,
        implemented similarly to how it was outlined in Ali et al. 2023 paper.

        <inputs> is of shape (batch_size * (family_size + 1), num_PWMs, seq_length_after_conv>) and the return value of
        the function is of the shape (batch_size * (family_size + 1), num_PWMs)."""

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
    """Define a Trainable Motif Interaction module similarly to how it was defined in Ali et al. 2023 paper where it
    applies motifs interaction-dependent weights to the output of the TrainablePooling layer, which allows to take into
    account the contribution of other TFs (motifs), consequently the input needs to be of shape (n, num_PWMs)."""

    def __init__(self, num_PWMs: int):
        """The motif interactions matrix is of shape (num_PWMs, num_PWMs) and is randomly initialized to values between
        0 and 1. That initialization is changed in the <MotifBasedEncoder> class so that the diagonal values are 1
        (it is the identity matrix)."""

        super(TrainableMotifInteractions, self).__init__()
        self.num_PWMs = num_PWMs

        # Create the trainable motif interactions (attention) matrix
        self.motif_interactions = nn.Parameter(torch.rand(num_PWMs, num_PWMs), requires_grad=True)

    def forward(self, inputs):
        """Return an interaction-dependent weight scaled version of <inputs>, which is determined by the trainable
        <self.motif_interactions> parameter, implemented similarly to how it was outlined in Ali et al. 2023 paper.
        <inputs> and the return value of the function are of shape (batch_size * (family_size + 1), num_PWMs)."""

        # Calculate the interaction-dependent weight w
        w = torch.sigmoid(torch.matmul(inputs, self.motif_interactions))  # shape (batch_size, num_PWMs)

        # Determine each motif contribution using <w> (element-wise multiplication)
        output = torch.mul(w, inputs)  # shape (batch_size, num_PWMs)

        return output


# Creating the motif-based encoder
class MotifBasedEncoder(nn.Module):
    """Define a Motif-Based Encoder module similarly to how it was defined in Alan et al. 2025 paper where it applies
    a series of ReverseComplement, 1D Convolutions, Scaling, Pooling, and Attention layers into the input sequences so
    that the model can learn PWM weights that are interpretable and biologically significant.

    The input needs to be of shape (n, num_bases, seq_length) which represents n one-hot encoded sequences to be
    encoded by the model."""

    def __init__(self, num_PWMs: int = 256, PWM_width: int = 15, window: int = 10, num_bases: int = 4,
                 gb: torch.tensor = torch.tensor([0.25, 0.25, 0.25, 0.25]),
                 complement_order: torch.tensor = COMPLEMENT_ORDER, set_initial_values: bool = True):
        """This MotifBasedEncoder follows the same model architecture as the one outlined in Alan et al. 2025 paper."""
        super(MotifBasedEncoder, self).__init__()
        # Define the attributes of the encoder
        self.num_PWMs = num_PWMs
        self.PWM_width = PWM_width
        self.window = window
        self.num_bases = num_bases
        self.gb = gb
        self.complement_order = complement_order

        # Define the layers of the encoder
        self.reverse_complement = ReverseComplement(self.complement_order)
        self.PWM_constraint = PWMConstraint(self.gb)
        self.PWMs_conv = nn.Conv1d(in_channels=self.num_bases, out_channels=self.num_PWMs, kernel_size=self.PWM_width,
                                   bias=False)
        self.window_pool = nn.MaxPool1d(kernel_size=self.window, stride=self.window, ceil_mode=True)
        self.scaling_layer = TrainableScaling(self.num_PWMs)
        self.pooling_layer = TrainablePooling(self.num_PWMs)
        self.attention_layer = TrainableMotifInteractions(self.num_PWMs)
        self.batch_norm_layer = nn.BatchNorm1d(self.num_PWMs)  # gamma and beta parameters are trainable

        # Define custom initial values
        if set_initial_values:
            # For the scaling layer
            init.constant_(self.scaling_layer.scale, 1.0)
            init.constant_(self.scaling_layer.bias, 0.0)
            # For the pooling layer
            init.constant_(self.pooling_layer.pooling, 1.0)
            # For the attention layer
            init.eye_(self.attention_layer.motif_interactions)

    def forward(self, inputs):
        """Return the representation vectors for each one-hot encoded sequence in <inputs>. <inputs> is of shape
        (batch_size * (family_size + 1), num_bases, seq_length), and the length of the sequences would have been
        appropriately padded and/or truncated before being fed into this encoder to <seq_length>.

        First the sequences would be reverse complemented using the ReverseComplement module and then both would be fed
        to a conv1d module, after which the output values from the reverse-complemented would be reversed and the
        maximum value between the forward and reverse scans would be kept, after which a max pool of window
        <self.window> is applied. After that the outputs are fed into the TrainableScaling, TrainablePooling, then
        TrainableMotifInteractions modules, followed by a batch normalization layer.

        The output of the encoder is of shape (batch_size * (family_size + 1), num_PWMs)."""
        
        # Get the reverse compliment of the input sequences
        rev_comp = self.reverse_complement(inputs)

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


class ReverseHomologyModel(nn.Module):
    """Define a Reverse Homology module that applies a non-linear projection head to the encodings of a
    MotifBasedEncoder module. This module might lead to better contrastive learning, as was observed in the simCLR
    paper.

    The input needs to be of shape (n, num_bases, seq_length) which represents n one-hot encoded sequences to be
    encoded by the model."""

    def __init__(self, motif_based_encoder: MotifBasedEncoder, encoder_num_PWMs: int, l2: int, l3: int):
        """A projection head is added after the <motif_based_encoder> to see if that leads to better results in terms
        of contrastive learning, as using a 2 layered projection head in the simCLR paper was shown to substantially
        improve the learning process. For extracting representations after learning, only the representations gotten
        after the <motif_based_encoder> would be used."""

        super(ReverseHomologyModel, self).__init__()
        # Define the motif-based encoder
        self.motif_based_encoder = motif_based_encoder
        # Define the projection head
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_num_PWMs, l2),
            nn.BatchNorm1d(l2),
            nn.ReLU(),
            nn.Linear(l2, l3)
        )

    def forward(self, inputs):
        """Return representation vectors of <inputs> that will be later used for contrastive loss learning that are the
        result of encoding them through a MotifBasedEncoder first, represented by <self.motif_based_encoder> then
        through a Multi-Layer Perceptron (MLP) projection head, represented by <self.projection_head>.

        <inputs> is of shape (batch_size * (family_size + 1), num_bases, seq_length), and the output of the function is
        of shape (batch_size, l3)."""

        motif_based_encoder_representations = self.motif_based_encoder(inputs)
        metric_embeddings = self.projection_head(motif_based_encoder_representations)
        return metric_embeddings  # shape: (batch_size * (family_size + 1), l3)
