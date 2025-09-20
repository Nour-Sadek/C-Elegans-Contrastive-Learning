# Inferring Interpretable, TF and RBP motif-based representations of Caenorhabditis Elegans promoter and 3'UTR regions using contrastive learning with orthology as the learning signal

This project aims to replicate what was previously done for Saccharomyces Cerevisiae's non-coding regulatory regions 
upstream of coding genes, promoters (Alan et al. 2025), but for the corresponding promoter regions for Caenorhabditis 
Elegans genes, as well as for the non-coding regulatory regions downstream of coding genes, 3'UTRs.

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
cis-bp using TomTom and clustering the motif-based representations and performing GO enrichment analysis, among others.

I aim to perform similar analyses and implement the motif-based encoder using PyTorch using orthologous genes from 
clade V nematoda as the learning signal to learn learned motifs-based promoter and 3'UTR representations.

## Table of Contents

1. Homologous species selection and determination of orthologous genes using OrthoFinder
2. Data collection of upstream non-coding promoter and downstream non-coding 3'UTR regions of Caenorhabditis Elegans 
genes and its orthologs
3. Architecture of the motif-based encoder
4. Training of the encoder to infer PWMs of the motifs through reverse homology (contrastive learning) that uses evolution 
(orthologous sequences) as the training signal
5. Encoding of the promoter and 3'UTR sequences after training by averaging the representation over all available orthologs
6. Comparison of the learned PWMs to available consensus motif databases for transcription factors and RNA binding proteins 
using TomTom
7. Extra: Problems faced when building and testing the model
8. References

## 1. Homologous species selection and determination of orthologous genes using OrthoFinder

### 1.1 Downloading the annotation (gff3) and the genomic and protein sequences (fasta) files for each of the clade V nematoda species

There are genome lists for 65 clade V nematoda species in the WormBase Parasite database, found [here](https://parasite.wormbase.org/species.html), and for 
species with multiple bioprojects, the bioproject with the most protein-coding genes was chosen. The available FTP 
server with the FTP host `ftp.ebi.ac.uk` was used in order to download the gff3 annotation file and the fasta genomic 
and protein sequences files for each of these 64 species. The script `download_files.py` was used to extract the required 
files.

### 1.2 Preparing the protein sequences fasta files for finding orthologs for nematoda promoters using OrthoFinder

After the files are done downloading, the same `downlaod_files.py` then filters the protein sequences fasta files for 
each species so that only the longest protein isoform for each protein-coding gene is kept.

After that, the Orthofinder software is run using the same diamond_ultra_sens --fewer-files -p options as Alan et al. 
2025 paper. To choose which of the 65 species to provide orthologs for C. elegans, only the species that had at least 
5000 one-to-one ortholog genes with C. Elegans were kept, which amounted to 44 species.

## 2. Data collection of upstream non-coding promoter and downstream non-coding 3'UTR regions of Caenorhabditis Elegans genes and their orthologs

The files needed for later to be used as inputs to the encoder should be formatted such that each C. elegans gene has a 
json file associated with it where each file is a dictionary where species name is the key and the orthologous sequence is 
the value; one set of files for the promoter sequences and another set of files for the 3'UTR sequences. A promoter sequence 
is considered the upstream region for each orthologous gene up to 800bp or the next annotated gene, and the 3'UTR sequence is 
considered the downstream region for each orthologous gene up to 200bp or the next annotated gene.

The script that will generate these files, alongside secondary files that are used to generate those final files, is the 
`get_orthologous_sequences.py` file. This script uses the three files downloaded in step 1 to get the final files.

The step-by-step process of how the final files were generated is as follows:
1. First, go over every gff3 annotation file for every species and extract the following information about every transcript: 
sequence region that contains the transcript (e.g. chromosome name), strand (+ or -), start and end coordinates, and parent 
gene name; for every species, save the information as a json file which represents a dictionary where the key is the 
transcript id and the value is a dictionary that stores the above-mentioned information. In addition, save another json 
file which represents a dictionary where the key is the sequence region name and the value is a dictionary where the key 
is a gene id and the value is a tuple of the start and end coordinated for said gene; these gene ids are sorted in ascending 
order of the start coordinate for each sequence region.
2. Using both of the json files generated in step 1 for each species, determine the start and end coordinates for the 
promoter and 3'UTR regions associated with each transcript, following the criteria mentioned previously, which are that the 
promoter region is considered the upstream region for each gene (upstream of the transcript start site) up to 800bp or 
the next annotated gene and that the 3'UTR region is considered the downstream region for each gene (downstream of the 
gene annotation) up to 200bp or the next annotated gene. For each species, two json files are saved which represents a dictionary 
which is similar to the first json file created in step 1 but information being saved is for the upstream promoter region in one 
json file and for the downstream 3'UTR region for each transcript and not for the transcript itself.
3. Using the json files generated in step 2 as well as the genomic sequences fasta file for each species, determine the 
nucleotide sequence for each upstream promoter and 3'UTR downstream region for each transcript id. For promoter and 3'UTR regions 
on the - strand, the reverse complement of the sequence is obtained. For each species, two json files are saved which represent 
a dictionary where the key is the transcript id and the value is a dictionary saving both the parent gene id of the transcript and 
the promoter sequence in one json file and the 3'UTR sequence in another json file.
4. Using the json files in step 3 and one of the Excel files generated from running OrthoFinder that specifies the 
orthologous transcripts for each of the C. Elegans transcripts, the final files are generated where a set of files store 
the orthologous promoter sequences for a gene and another set store the orthologous 3'UTR sequences for a gene.

## 3. Architecture of the motif-based encoder

The same architecture as the one outlined in Alan et al. 2025 was replicated in PyTorch. The `motif_based_encoder.py` 
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

        The input needs to be of shape (num_seqs, num_bases, seq_length) which represents num_seqs one-hot encoded sequences
        to be encoded by the model."""

        def __init__(self, num_PWMs: int = 256, PWM_width: int = 15, window: int = 10, num_bases: int = 4,
                     set_initial_values: bool = True, consider_reverse_complement=True):
            """This MotifBasedEncoder follows the same model architecture as the one outlined in Alan et al. 2025 paper."""

            super(MotifBasedEncoder, self).__init__()
            # Define the attributes of the encoder
            self.num_PWMs = num_PWMs
            self.PWM_width = PWM_width
            self.window = window
            self.num_bases = num_bases
            self.gb = GB
            self.complement_order = COMPLEMENT_ORDER
            self.consider_reverse_complement = consider_reverse_complement

            # Define the layers of the encoder
            self.reverse_complement = ReverseComplement()
            self.PWM_constraint = PWMConstraint()
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
                init.constant_(self.pooling_layer.pooling, 2.0)
                # For the attention layer
                init.eye_(self.attention_layer.motif_interactions)

        def forward(self, inputs):
            """Return the representation vectors for each one-hot encoded sequence in <inputs>. <inputs> is of shape
            (num_seqs, num_bases, seq_length), and the length of the sequences would have been appropriately padded and/or
            truncated before being fed into this encoder to <seq_length>.

            First the sequences would be PWM scaled through the PWM_constraint module, then reverse complemented using the
            ReverseComplement module and then both would be fed to a conv1d module, after which the output values from the
            reverse-complemented would be reversed and the maximum value between the forward and reverse scans would be
            kept. If <consider_reverse_complement> was set to False, then the original input only would be fed into the
            conv1d module. After that, a max pool of window <self.window> is applied. After that the outputs are fed into
            the TrainableScaling, TrainablePooling, then TrainableMotifInteractions modules, followed by a batch
            normalization layer.

            The output of the encoder is of shape (num_seqs, num_PWMs)."""

            scaled_PWM_weights = self.PWM_constraint(self.PWMs_conv.weight)

            if self.consider_reverse_complement:  # for promoter sequences

                # Get the reverse compliment of the input sequences
                rev_comp = self.reverse_complement(inputs)

                # Run both the input seqs and their reverse complements through the PWM convolutional layers
                inputs_conv = F.conv1d(inputs, scaled_PWM_weights, bias=None, stride=self.PWMs_conv.stride,
                                       padding=self.PWMs_conv.padding, dilation=self.PWMs_conv.dilation,
                                       groups=self.PWMs_conv.groups)
                rev_comp_conv = F.conv1d(rev_comp, scaled_PWM_weights, bias=None, stride=self.PWMs_conv.stride,
                                         padding=self.PWMs_conv.padding, dilation=self.PWMs_conv.dilation,
                                         groups=self.PWMs_conv.groups)

                # Reverse the order of scores for <rev_comp_inputs_conv> then take better score between the forward and
                # reverse at each position
                rev_order_rev_comp_conv = torch.flip(rev_comp_conv, dims=[2])
                conv_output = torch.maximum(inputs_conv, rev_order_rev_comp_conv)

            else:  # for 3'UTR sequences

                # Run only the forward input seqs through the PWM convolutional layer
                conv_output = F.conv1d(inputs, scaled_PWM_weights, bias=None, stride=self.PWMs_conv.stride,
                                       padding=self.PWMs_conv.padding, dilation=self.PWMs_conv.dilation,
                                       groups=self.PWMs_conv.groups)

            # To avoid counting overlaps, take best match in a <self.window> nt window
            conv_output = self.window_pool(conv_output)  # shape (num_seqs, num_PWMs, seq_length_after_conv)

            # Apply the scaling layer
            scaled_output = torch.sigmoid(
                self.scaling_layer(conv_output))  # shape (num_seqs, num_PWMs. seq_length_after_conv)

            # Apply the pooling layer
            pooled_output = self.pooling_layer(scaled_output)  # shape (num_seqs, num_PWMs)

            # Apply the attention (Motif Interactions) layer followed by batch normalization
            output = self.batch_norm_layer(self.attention_layer(pooled_output))

            return output  # shape (num_seqs, num_PWMs)

This is how the motif-based encoder runs through the batch of input sequences which are one-hot encoded promoter sequences:

- The weights of the PWM convolutional layer are scaled by the `PWM_Constraint` module where the values are similar to 
that of PWMs. After that, the reverse complements of the input sequences are computed and then both the forward and reverse sequences are scanned 
over by the PWM-constrained convolutional layer whose weights represent the PWMs to be learned. After that, the scanned values for the 
reverse complements are flipped and the max score between the forward and reverse scans at each position are chosen. That is 
done for promoter sequences, where `consider_reverse_complement=True`; for 3'UTR sequences, where `consider_reverse_complement=False`, 
only the forward sequences are considered. After that, another max pool of a certain window size is done to avoid counting 
overlapping matches and then the output of that pooling is fed to scaling, then pooling, then attention layers after which the 
representations are batch normalized, and that final output of batch of representations are the motif-based representations for 
each promoter sequence fed into the encoder.

## 4. Training of the encoder to infer PWMs of the motifs through reverse homology (contrastive learning) that uses evolution (orthologous sequences) as the training signal

### 4.1 How the training was done with the infoNCE contrastive loss

The training of the encoder is done in the `using_the_model.py` script where after the input sequences have made a pass 
through the encoder, their representations would be evaluated by the contrastive InfoNCE loss function. 
Training was done similarly to how it was done in Alan et al. 2025, with a slight modification. For each batch of genes, 
all the orthologous sequences per gene family are run through the encoder, alongside additional sequences depending on the 
value of the target set size hyperparameter where if it is greater than the current batch size, target set size - batch size
sequences from genes outside the batch would be randomly chosen and added to the current batch's target set. And so, for each 
batch, after the sequences are run through the encoder, the infoNCE loss is calculated, which as explained previously, 
is aimed to change the encoder parameters so that positive representations are closer in distance (representations of the 
family embedding and another member of the same gene family) and negative representations are farther apart (representations 
of a family embedding and another sequences that isn't part of the gene family that that family embedding represents). In 
calculating the infoNCE loss, every sequence of every gene in the batch is considered as a positive sequence, the family of 
sequences to represent the family embedding would be randomly chosen per positive sequence based on the family size hyperparameter, 
and the negative sequences are chosen at random from the other genes, one sequence per gene, as well as the extra sequences 
that were added as inputs to the encoder solely to act as negative sequences in the loss function. The training is done through 
the `train_motif_based_encoder` function, and the calculation of the infoNCE loss is done through the `infoNCE_loss` and 
`calculate_logits` functions.

### 4.2 How the genes and orthologous sequences for each gene are chosen to be part of training

Considering that a gene should have at least family_size + 1 promoter sequences in its family, only genes that fit 
that criteria are considered during training, as well as other constraints such as the length of the promoter sequences 
involved should be at least 10% of the size limit which is 80bp (10% of 800bp) and the length of the 3'UTR sequences 
should be a minimum of 20bp (10% of 200bp), after which the remaining sequences that will be used for training are padded so 
that all of them are 800bp for promoters and 200bp for 3'UTRs and are subsequently one-hot encoded. With a family 
size of 8 and a minimum size limit of 80bp, a total of 12403 genes with a total number of 335816 orthologous promoter 
sequences were considered as valid for the encoder, with these genes being split 90% for training and 10% for validation. 
Determining which genes are chosen is done through the `read_files` function, and the 90/10 split is done using the 
`split_data` function.

### 4.3 Choice of hyperparameters: Hyperparameter tuning using Pytorch-lightning and Optuna

These are few of the hyperparameters chosen for the encoder:
- family size of 8
- target set size of 400
- batch size of 32
- promoter sequence length of 800bp for training
- promoter sequence length of 500bp for computing motif-based representations
- 3'UTR sequence length of 200bp for training
- 3'UTR sequence length of 200bp for computing motif-based representation
- Adam optimizer with a learning rate of 0.01
- 256 PWMs with a width of 15bp for promoters
- 128 PWMs with a width of 10bp for 3'UTRs
- temperature of 0.15 for the infoNCE loss
- 100 epochs of training 

The following hyperparameters were tuned, both when the family size was 8 and when it was 4:

    # Hyperparameter search space
    target_set_size = trial.suggest_categorical("target_set_size", [400, 600, 800])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2 ** i for i in range(5, 8)])
    temperature = trial.suggest_float("temperature", 0.05, 0.9)

Hyperparameter tuning was done using Pytorch-lightning and Optuna, which can be found in the `hyperparameter_tune.py` script. 
It wasn't as straightforward as it was implementing the hyperparameter search on the CIFAR-10 dataset in my previous projects, 
so it took some further understanding of these two packages, but I was ultimately able to make it work on my model.

After running 20 trials where each trial runs for a maximum of 10 epochs, the trial that reached the minimum validation loss 
of 5.2 was the one with the following hyperparameters:
    
    batch_size = 32 | learning_rate = 0.0119 | temperature = 0.14949 | target_set_size = 400

### 4.4 The loss and accuracy curves during training for both the training and validation datasets

Out of the clad V species, I focused on the caenorhabditis only species out of them (22) due to high divergence among the 
clad V species as a whole.

After training for 100 epochs, using the resources of the Compute Canada cluster, these are the loss and accuracy curves 
after training on the orthologous promoter sequences, with the addition of the non-linear projection head:

<img width="2967" height="2968" alt="Image" src="https://github.com/user-attachments/assets/66fd722e-e502-4e75-94a6-59aa0678c5e1" />

These are the loss and accuracy curves after training on the orthologous 3'UTR sequences:

<img width="2967" height="2968" alt="Image" src="https://github.com/user-attachments/assets/09dd0ebd-e658-40f6-8c83-9aade09f6d3c" />

## 5. Encoding of the promoter and 3'UTR sequences after training by averaging the representation over all available orthologs

### 5.1 How the genes to be included in the representation were chosen
After the model finished training, the valid genes were chosen again, but now each gene only required to have a minimum 
of two orthologous sequences, and each sequence's length required to be a minimum of the length of the PWM to be included. 
For the promoter sequences, a representation target length of 500bp was used and so sequences that were longer than that 
were truncated down to 500bp from their upstream end. For the 3'UTR sequences, a representation target length of 200bp, same 
as the length used for training, was used.

### 5.2 How the representation heatmaps where generated
After the valid genes and their orthologous sequences were chosen, the representation of each gene was determined, where 
the final representation was the average among the PWM scores of all orthologous sequences and the final representation 
would be saved as a json file, where each key is a gene id and each gene id's value is a dictionary where there is a Phylogenetically 
Averaged Motif (PAM) score for each PWM. This representation is generated using the `determine_rhiepa_representation` function 
in `creating_files_for_analysis.py` script. Then, this json file would be fed into the `create_java_treeview_files` function 
in the same script to cluster the representations and generate the required files to visualize the heatmap in Java TreeView.

### 5.3 The heatmaps for the learned representations of the promoter and 3'UTR sequences

## 6. Comparison of the learned PWMs to available consensus motif databases for transcription factors and RNA binding proteins using TomTom

After the model has finished training, the model's learned PWMs, which are the convolutional layer's weights, would be 
PWM-constrained then saved in MEME format using the `pwm_to_meme` function in the `create_files_for_analysis.py` script. 
Then, these motifs would be uploaded in the TomTom submission form, found at this link: https://meme-suite.org/meme/tools/tomtom, 
which is a Motif Comparison Tool part of the MEME suite where it compares user-given motifs to a database of known motifs. 
Each given motif can match to multiple motifs in the database; TomTom will rank the matches and produce an alignment of the 
PWMs between the matched motifs in the database and the user-given motifs.

For the weights of a model trained on promoter sequences, they were compared to two motif databases:
- the JASPAR (NON-REDUNDANT) DNA, JASPAR CORE (2024) nematodes database which is made up of 103 motifs, between 5 and 15 in width (average width 8.7).
- the CIS-BP 2.00 Single Species DNA, Caenorhabditis_elegans database which is made up of 287 motifs, between 6 and 21 in width (average width 9.8).

For the weights of a model trained on 3'UTR sequences, they were compared to the CISBP-RNA Single Species RNA, Caenorhabditis_elegans 
database which is made up of 20 motifs, between 6 and 8 in width (average width 7.1).

## 7. Extra: Problems faced when building and testing the model

The contrastive learning algorithm went through multiple iterations, and I learned a lot by going through setbacks when 
building and testing the model. Here are some of them, and how I handled each one:

### 7.1 How and when to apply the PWM constraint on the weights of the convolutional layer

At first, I attempted to apply the PWM constraint after the model updated the parameters in every epoch, however the way 
I implemented this led to breaking of the computational graph and the PWM weights would stop updating after a couple of 
epoch runs, and get stuck at 0.25 (akin to equal probability of each of the four bases, as in random initialization). 

To fix this problem, I opted for not attempting to update the PWM weights at all during training, but after training, to 
apply the PWM constraint on the weights then visualize them as PWM motifs; however that didn't work either and that is 
because when the constraint was not being applied during training, the model was not scanning the DNA sequences with PWM motifs 
and so it wouldn't learn them.

I looked further into how I can implement the constraint during training, and came up with applying the constraint to the 
weights in the forward call of the module, then scan the sequences with these PWM-constrained weights. This wouldn't break 
the computational graph because the weights of the convolutional layer themselves are not being changed to fit the constraint 
but rather a constrained copy of them is used during scanning, and so the model will be learning to fit the PWM-constrained 
version of the convolutional layers, and therefore the parameter updates should reflect that. However, after training, the 
constraint needs to be applied again so that the weights can be properly visualized as PWMs. This change lead to proper 
training and learning of biologically-relevant PWMs.

### 7.2 How many positive sequences to consider in every batch

At first, to reduce the length of training time, I opted to pursue a similar training strategy that Alex et al. 2022 did, which 
was to rather than consider every sequence as a positive anchor in a batch, I would only consider one sequence per gene as 
a positive sequence, and so the number of positive sequences would be equal to the batch size. Additionally, the family 
and negative sequences would be fixed per batch rather than randomly choosing them for every positive sequence. While the training 
was much faster using this method (about 5 times faster), the current training strategy led to much better learning, i.e. better 
motifs being learned, so I opted to stick to choosing every sequence in a batch as the positive anchor per epoch.

## 8. References

Alan MM, et al. Inferring fungal cis-regulatory networks from genome sequences via unsupervised and interpretable representation 
learning. biorxiv. 2025. doi: https://doi.org/10.1101/2025.02.27.640643

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
