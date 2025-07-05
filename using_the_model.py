import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import os
import random

from motif_based_encoder import one_hot_encode_seq, pad_one_hot_encoded_seq, MotifBasedEncoder


def read_files(files_dir, target_length, min_num_orthologs, min_sequence_length, base_to_index) -> dict[str, list[torch.tensor]]:
    """files_dir should be the path to the directory that contains the json files that contain the orthologous promoters
    per gene.
    target_length is 800 for training but 500 for encoding the promoter sequences after training.
    min_num_orthologs is 5 (family_size of 4 + 1) for training and 2 after training.
    min_sequence_length is 10% of target_length for training and PWM_width after training."""
    valid_genes = {}
    for file_name in os.listdir(files_dir):
        gene_name = file_name.split("_")[0]
        path = os.path.join(files_dir, file_name)

        with open(path, "r") as file:
            gene_promoters = json.load(file)
        valid_sequences = []
        for sequence in gene_promoters.values():
            # Check if invalid bases are present
            if not set(sequence.upper()).issubset(base_to_index.keys()):
                continue
            # Check if the length of the sequence is greater than min_sequence_length
            if min_sequence_length is not None and len(sequence) < min_sequence_length:
                continue
            valid_sequences.append(sequence.upper())
        # check if enough promoter sequences are still valid
        if len(valid_sequences) < min_num_orthologs:
            continue
        # One-hot encode the sequences and ensure they are as long as <target_length>
        fixed_valid_sequences = []
        for sequence in valid_sequences:
            one_hot_seq = one_hot_encode_seq(sequence, base_to_index)
            if len(sequence) < target_length:  # add 0 padding to the right
                one_hot_seq = pad_one_hot_encoded_seq(one_hot_seq, target_length)
            elif len(sequence) > target_length:  # truncate the sequence from the start
                one_hot_seq = one_hot_seq[:, (target_length - len(sequence)):]
            fixed_valid_sequences.append(one_hot_seq)
        # Save the promoters of this gene as valid to be passed through the encoder
        valid_genes[gene_name] = fixed_valid_sequences

    return valid_genes


def split_data(valid_genes, train_ratio=0.9, seed=2025) -> tuple[dict, dict]:
    """Returns two dictionaries where they contain train_ratio and 1 - train_ration of the data in <valid_genes>
    respectively and that split is randomly assigned based on the <seed> for reproducibility.

    This function will be used to create train and validation data sets.
    """
    keys = list(valid_genes.keys())

    # Randomly shuffle the keys then get the split index based on <train_ratio>
    random.Random(seed).shuffle(keys)
    split_idx = int(len(keys) * train_ratio)

    # Split keys into train and val data sets
    train_keys = keys[:split_idx]
    val_keys = keys[split_idx:]
    train_set_genes = {gene: valid_genes[gene] for gene in train_keys}
    val_set_genes = {gene: valid_genes[gene] for gene in val_keys}

    return train_set_genes, val_set_genes


def infoNCE_loss(seqs_embeddings, family_size, temperature):
    """<seqs_embeddings> is of shape (num_seqs, num_PWMs). Excluding the last batch, num_seqs = batch_size * (
    family_size + 1)."""
    num_seqs = seqs_embeddings.shape[0]
    batch_size = num_seqs // (family_size + 1)

    # Create the indicator variable
    indicator_variable = torch.zeros(batch_size, dtype=torch.long)  # positive class is at zero index for each query

    # Create the (p) variable which represents dot products between the family embeddings and single sequences, scaled
    # by temperature
    p = torch.zeros(batch_size, num_seqs - (family_size + 1) + 1)

    # Determine the values of the <p> variable for each family
    curr_sample = 0
    for start_of_family in range(0, num_seqs, family_size + 1):
        # Get the family embedding representation
        family_tensors = seqs_embeddings[start_of_family:(start_of_family + family_size)]
        family_embedding = torch.mean(family_tensors, dim=0)

        # Get the positive anchor embedding
        positive_anchor = seqs_embeddings[start_of_family + family_size]
        positive_anchor = positive_anchor.unsqueeze(0)
        # Extract the negative single sequence embeddings
        negative_sequences_embeddings = torch.cat((seqs_embeddings[:start_of_family],
                                                   seqs_embeddings[start_of_family + family_size + 1:]), dim=0)
        # Combine the embeddings of the single sequences where the positive anchor embedding is on top and all other
        # negative single sequences are under it
        single_sequences_embeddings = torch.cat((positive_anchor, negative_sequences_embeddings), dim=0)

        # Calculate the dot product between the family and single sequences embeddings, scaled by temperature
        family_scores = torch.matmul(single_sequences_embeddings, family_embedding) / temperature

        # Update the <p> variable
        p[curr_sample] = family_scores
        curr_sample = curr_sample + 1

    # Calculate the Categorical Cross Entropy loss between the indicator variable and <p> (does softmax internally)
    loss = F.cross_entropy(p, indicator_variable)

    return loss
