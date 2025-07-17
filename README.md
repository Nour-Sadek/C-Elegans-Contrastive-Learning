# Inferring Intrerpretable, TF motif-based representations of Caenorhabditis Elegans promoter regions using contrastive learning with orthology as the learning signal

This project aims to replicate what was previously done for Saccharomyces Cerevisiae's non-coding regulatory regions 
upstream of coding genes, promoters (Alan et al. 2025), but for the corresponding promoter regions for Caenorhabditis 
Elegans genes.

Briefly, the authors aimed to learn interpretable Transcription Factor (TF) motif PWMs as well as TF motif-based 
representations of promoter regions using gene orthology as the learning signal by leveraging contrastive learning. 
Contrastive learning aims to learn appropriate representations that minimize the distance between similar (positive) 
samples and maximize the distance between dissimilar (negative) samples. There are multiple ways to generate similar 
data points, for example creating augmented versions of images for classification tasks. For promoter sequences, the 
authors aimed to use reverse homology, aka gene orthology, to generate augmented versions of promoter sequences where 
for each S. Cerevisiae gene, the upstream regions of orthologous genes from clades of fungi were used as positive 
samples and upstream regions of other genes were used as negative samples. To learn PWMs, the authors used a trainable 
1d convolutional layer where, applying a PWM constraint to make the weights be equivalent to valid PWM probabilities, 
where after training the weights of each kernel would represent a learned PWM of a TF. To make the weights of that 
convolutional layer interpretable, they added downstream layers following what Ali et al. 2023 did where trainable 
scaling, pooling, and attention (motif interaction) layers were added to account for different motif aggregation 
strategies (max versus average pooling of motif scores) and synergistic or saturation interactions across different TFs. 
The overall architecture and contrastive learning strategy used were inspired by the Alex et al. 2022 and Amy et al. 
2020 papers. They then analyzed the learned motifs by comparing them to curated TF motif databases like JASPAR and 
cis-bp using TomTom and clustering the motif-based representations and performing GO enrichment analyses, among others.

I aim to perform similar analyses and implement the motif-based encoder using PyTorch using orthologous genes from 
clad V nematoda as the learning signal.

## Table of Contents

1. Homologous species selection and determination of orthologous genes using OrthoFinder
2. Data collection of upstream non-coding promoter regions of Caenorhabditis Elegans genes and its orthologs
3. Architecture of the motif-based encoder
4. Training of the encoder to infer PWMs of the motifs through reverse homology (contrastive learning) that uses evolution 
(orthologous sequences) as the training signal
5. Encoding of the promoter sequences after training by averaging the representation over all available orthologs
6. Comparison of the learned PWMs to available consensus motif databases for transcription factors using TomTom
7. Go Enrichment Analysis
8. References

## 1. Homologous species selection and determination of orthologous genes using OrthoFinder

### 1.1 Downloading the annotation (gff3) and the genomic and protein sequences (fasta) files for each of the clad Vnematoda species

There are genome lists for 65 clade V nematoda species in the WormBase Parasite database, found [here](https://parasite.wormbase.org/species.html), and for 
species with multiple bioprojects, the bioproject with the most protein-coding genes was chosen. Using the available FTP 
server using the FTP host `ftp.ebi.ac.uk` was used in order to download the gff3 annotation file and the fasta genomic 
and protein sequences files for each of these 64 species. The script `download_files.py` was used to extract the required 
files.

### 1.2 Preparing the protein sequences fasta files for finding orthologs for nematoda promoters using OrthoFinder

After the files are done downloading, the same `downlaod_files.py` then filters the protein sequences fasta files for 
each species so that only the longest protein isoform for each protein-coding gene is kept.

After that, the Orthofinder software is ran using the same diamond_ultra_sens --fewer-files -p options as Alan et al. 
2025 paper. To choose which of the 65 species to provide orthologs for C. elegans, only the species that had at least 
5000 one-to-one ortholog genes with C. Elegans were kept, which amounted to 44 species.

## 2. Data collection of upstream non-coding promoter regions of Caenorhabditis Elegans genes and its orthologs

The files needed for later to be used as inputs to the encoder should be formatted such that each C. elegans gene has a 
json file assoicated with it where each file is a dictionary where species name is the key and the promoter sequence is 
the value. A promoter sequence is considered the upstream region for each orthologous gene up to 800bp or the next 
annotated gene.

The script that will generate these files, alongside secondary files that are used to generate those final files, is the 
`get_promoter_sequences.py` file. This script uses the three files downloaded in step 1 to get the final files.

The step-by-step process of how the final files were generated is as follows:
1. First, go over every gff3 annotation file for every species and extract the following information about every transcript: 
sequence region that contains the transcript (e.g. chromosome name), strand (+ or -), start and end coordinates, and parent 
gene name; for every species, save the information as a json file which represents a dictionary where the key is the 
transcript id and the value is a dictionary that stores the above-mentioned information. In addition, save another json 
file which represents a dictionary where the key is the sequence region name and the value is a dictionary where the key 
is a gene id and the value is a tuple of the start and end coordinated for said gene; these gene ids are sorted in ascending 
order of the start coordinate for each sequence region.
2. Using both of the json files generated in step 1 for each species, determine the start and end coordinates for the 
promoter region associated with each transcript, following the two criteria mentioned previously, which are that the 
promoter region is considered the upstream region for each gene (upstream of the transcript start site) up to 800bp or 
the next annotated gene. For each species, a json file is saved which represents a dictionary which is similar to the 
first json file created in step 1 but information being saved is for the upstream promoter region for each transcript 
and not for the transcript itself.
3. Using the json files generated in step 2 as well as the genomic sequences fasta file for each species, determine the 
nucleotide sequence for each upstream promoter region for each transcript id. For promoter regions on the - strand, the 
reverse complement of the sequence is obtained. For each species, a json file is saved which represents a dictionary 
where the key is the transcript id and the value is a dictionary saving both the parent gene id of the transcript and 
the promoter sequence.
4. Using the json files in step 3 and one of the Excel file generated from running OrthoFinder that specifies the 
orthologous transcripts for each of the C. Elegans transcripts, the final files are generated where every file stores 
the orthologous promoter sequences for a gene.

## 3. Architecture of the motif-based encoder

The same architecture as the one outlines in Alan et al. 2025 was replicated in PyTorch. The `motif_based_encoder.py` 
script contains the module classes that implement the trainable scaling (TrainableScaling), pooling (TrainablePooling), 
and attention (TrainableMotifInteractions) layers, as were described in the Ali et al. 2023 paper, which were used to 
build the MotifBasedEncoder module, as well as the PWM constraint (PWMConstraint) module, as was described in Alan et al. 
2025 paper.

An additional module class called ReverseHomologyModel was created where a non-linear projection head is added after the 
MotifBasedEncoder to see if that addition would lead to better contrastive learning, as was observed in the simCLR paper.

This is the MotifBasedEncoder class:

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

This is how the motif-based encoder runs through the batch of input sequences which are one-hot encoded promoter sequences:
- The reverse complements of the input sequences are computed and then both the forward and reverse sequences are scanned 
over by the convolutional layer whose weights represent the PWMs to be learned. After that, the scanned values for the 
reverse complements are flipped and the max score between the forward and reverse scans at each position are chosen. 
After that, another max pool of a certain window size is done to avoid counting overlapping matches and then the output 
of that pooling is fed to scaling, then pooling, then attention layers after which the representations are batch normalized, 
and that final output of batch of representations are the motif-based representations for each promoter sequence fed into 
the encoder.

## 4. Training of the encoder to infer PWMs of the motifs through reverse homology (contrastive learning) that uses evolution (orthologous sequences) as the training signal

### 4.1 How the training was done with the infoNCE contrastive loss

The training of the encoder is done in the `using_the_model.py` script where after the input sequences have make a pass 
through the encoder, their representations would be evaluated by the contrastive InfoNCE loss function. The training 
was done slightly differently from how it was done in Alan et al. 2025 and more aligned with how it was done in Alex et 
al. 2022 where for each batch of genes, a family of random promoter sequences from each gene's family of homologous 
sequences are chosen to represent the family embedding (the average of embeddings of individual sequences in the family) 
and 1 different sequence is chosen to be part of the target set. There is additionally the hyperparameter target set size 
and if that one is greater than the batch size, then remaining sequences from genes outside the batch would be randomly 
chosen and added to the current batch's target set. And so, for each batch, a family of sequences from each gene's family + extra 
sequences to be added to the target set are run through the encoder to get their representations and then the infoNCE 
loss is calculated, which as explained previously, is aimed to change the encoder parameters so that positive representations
aer closer in distance (representations of the family embedding and another member of the same gene family) and negative 
representations are farther apart (representations of a family embedding and another sequences that isn't part of the 
gene family that that family embedding represents). 

### 4.2 How the genes and orthologous sequences for each gene are chosen to be part of training

Considering that a gene should have at least family_size + 1 promoter sequences in its family, only genes that fit 
that criteria are considered during training, as well as other constraints such as the length of the promoter sequences 
involved should be at least 10% of the size limit which is 80bp (10% of 800bp), after which the remaining sequences that 
will be used for training are padded so that all of them are 800bp and are subsequently one-hot encoded. With a family 
size of 8 and a minimum size limit of 80bp, a total of 12403 genes with a total number of 335816 orthologous promoter 
sequences were considered as valid for the encoder, with these genes being split 90% for training and 10% for validation.

### 4.3 Chosen hyperparameters for training

These are few of the hyperparameters chosen for the encoder:
- family size of 8
- target set size of 400
- batch size of 256
- promoter sequence length of 800bp for training
- promoter sequence length of 500bp for computing motif-based representations
- Adam optimizer with a learning rate of 0.01
- 256 PWMs with a width of 15bp
- temperature of 0.1 for the infoNCE loss
- 100 epochs of training

However, it has been tricky to train the encoder and so hyperparameter tuning of these parameters would be done. Also, so 
far, the weights of the PWM were being constrained after every parameter update, but it seems interesting to try only 
applying the constraint after training is done rather than multiple times during training.

## 8. References

Alan MM, et al. Inferring fungal cis-regulatory networks from genome sequences. biorxiv. 2025. 
doi: https://doi.org/10.1101/2025.02.27.640643

Alex XL, et al. Discovering molecular features of intrinsically disordered regions by using evolution for contrastive learning.
PLOS Comput Biol. 2022;18: e1010238. doi:10.1371/journal.pcbi.1010238

Ali TB, et al. An intrinsically interpretable neural network architecture for sequence to function learning. 
bioRxiv; 2023. p. 2023.01.25.525572. doi:10.1101/2023.01.25.525572

Amy XL, et al. Evolution Is All You Need: Phylogenetic Augmentation for Contrastive Learning. arXiv; 2020. 
doi:10.48550/arXiv.2012.13475

Chen T, et al. A Simple Framework for Contrastive Learning of Visual Representations. arXiv. 2020. 
doi: https://doi.org/10.48550/arXiv.2002.05709

Emms DM, Kelly S. OrthoFinder: phylogenetic orthology inference for comparative genomics. Genome Biol. 2019;20: 238. 
doi:10.1186/s13059-019-1832-y

Kevin LH, et al. WormBase Parasite - a comprehensive resource for helminth genomics. Molecular and Biochemical Parasitology, 
Volume 215, July 2017, Pages 2-10, https://doi.org/10.1016/j.molbiopara.2016.11.005

Shobhit G, John AS, Timothy LB, William SN. Quantifying similarity between motifs. Genome Biology, 8(2):R24, 2007. 
doi: https://doi.org/10.1186/gb-2007-8-2-r24
