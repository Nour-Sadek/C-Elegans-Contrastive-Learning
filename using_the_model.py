import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import json
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

from motif_based_encoder import one_hot_encode_seq, pad_one_hot_encoded_seq, pwm_constraint_conv1d, MotifBasedEncoder

# Constants
gb = torch.tensor([0.25, 0.25, 0.25, 0.25])
BASE_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}  # bases are of the order ACGT
SEED = 2025
FILES_DIR = "./ortholog_promoters_per_gene"
TRAIN_LENGTH = 800
TRAIN_MIN_NUM_ORTHOLOGS = 5
TRAIN_MIN_SEQ_LENGTH = 80
MODEL_OUTPUTS_DIR = "model_outputs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def split_into_batches(genes_list, batch_size, seed=None):
    # randomly shuffle the genes
    if seed is not None:
        random.Random(seed).shuffle(genes_list)
    else:
        random.shuffle(genes_list)

    # Split into inner lists of genes of size <batch_size>
    genes_in_batches = [genes_list[i:i + batch_size] for i in range(0, len(genes_list), batch_size)]

    return genes_in_batches


def get_encoder_inputs_from_batch(genes_dict, batch_of_genes, family_size, seed=None):
    if seed is not None:
        random.seed(seed)

    # Create a nested list of sequences per gene
    batch_of_seqs = [random.sample(genes_dict[gene], family_size + 1) for gene in batch_of_genes]

    # Flatten the list so that all sequences are in one list (first <family_size> + 1 seqs refer to first gene,
    # followed by seqs for the second gene and so on)
    batch_of_seqs_flat = [seq for list_of_seqs in batch_of_seqs for seq in list_of_seqs]

    # Stack the seqs into a single tensor of shape (num_seqs, num_bases, seq_length) where the shape of the seqs was
    # (num_bases, seq_length)
    inputs = torch.stack(batch_of_seqs_flat)

    random.seed()

    return inputs  # shape (num_seqs, num_bases, seq_length)


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


def train_motif_based_encoder(valid_genes, model, family_size, batch_size, learning_rate, temperature, num_epochs):
    # Set up the model
    initial_weights = pwm_constraint_conv1d(model.PWMs_conv.weight.data,
                                            gb)  # Save the initial weights of the convolutional layer (PWMs)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Split the data into training and validation data sets
    train_set, val_set = split_data(valid_genes)

    # Set up the training loop
    train_loss_epochs = []
    val_loss_epochs = []
    for epoch in range(num_epochs):
        model.train()
        # Randomly split the training set into groups of size <batch_size>
        all_train_genes = list(train_set.keys())
        genes_in_batches = split_into_batches(all_train_genes, batch_size, seed=epoch)
        train_loss_step = []
        # Loop over every batch of genes
        for batch_of_genes in genes_in_batches:
            optimizer.zero_grad()
            # Only keep <family_size> + 1 orthologous sequences for each gene and consider the first <family_size>
            # elements to be the query set (represent the family embeddings) and the last element to be the target set
            # for that family.
            inputs = get_encoder_inputs_from_batch(train_set, batch_of_genes, family_size,
                                                   seed=epoch)  # shape (num_seqs, num_bases, seq_length)
            inputs = inputs.to(device)
            seqs_embeddings = model(inputs)  # shape (num_seqs, num_PWMs)
            loss = infoNCE_loss(seqs_embeddings, family_size, temperature)
            loss.backward()
            optimizer.step()
            # Implement the constraint on the PWM motifs layer
            with torch.no_grad():
                constrained_weights = pwm_constraint_conv1d(model.PWMs_conv.weight.data, gb)
                model.PWMs_conv.weight.data.copy_(constrained_weights)
            train_loss_step.append(loss.item())
        train_loss_epochs.append(np.array(train_loss_step).mean())

        # Set up the validation loop to calculate the validation loss after one epoch of training
        model.eval()
        # Split the validation set into groups of size <batch_size>
        all_val_genes = list(val_set.keys())
        genes_in_batches = split_into_batches(all_val_genes, batch_size, seed=SEED)
        val_loss_step = []
        with torch.no_grad():
            for batch_of_genes in genes_in_batches:
                inputs = get_encoder_inputs_from_batch(val_set, batch_of_genes, family_size, seed=SEED)
                inputs = inputs.to(device)
                seqs_embeddings = model(inputs)
                loss = infoNCE_loss(seqs_embeddings, family_size, temperature)
                val_loss_step.append(loss.item())
            val_loss_epochs.append(np.array(val_loss_step).mean())

        # Print the loss of the training and validation data sets after one epoch
        print(
            f"After epoch {epoch + 1}: training loss = {train_loss_epochs[epoch]}, validation loss = {val_loss_epochs[epoch]}")

    weights_after_training = model.PWMs_conv.weight.data.clone()
    print("Finished Training!")

    return initial_weights, weights_after_training, train_loss_epochs, val_loss_epochs, model


if __name__ == "__main__":
    # Get the valid genes
    # valid_genes = read_files(FILES_DIR, target_length=TRAIN_LENGTH, min_num_orthologs=TRAIN_MIN_NUM_ORTHOLOGS,
    #                          min_sequence_length=TRAIN_MIN_SEQ_LENGTH, base_to_index=BASE_TO_INDEX)

    # Save <valid_genes> to load later (it is taking a while...)
    # file_name = f"./valid_genes_for_encoder.pkl"
    # with open(file_name, "wb") as f:
    #     pickle.dump(valid_genes, f)

    # Determine the number of valid genes and orthologous sequences
    # seqs_num = 0
    # for gene in valid_genes:
    #     seqs_num = seqs_num + len(valid_genes[gene])

    # print(f"{len(valid_genes)} will be processed through the encoder with total {seqs_num} orthologous sequences.")
    file_name = f"./valid_genes_for_encoder.pkl"
    with open(file_name, "rb") as f:
        valid_genes = pickle.load(f)
    print("The training of the model will start.")

    # Train the model on the valid genes
    encoder = MotifBasedEncoder(num_PWMs=256, PWM_width=15, window=10, num_bases=4)
    encoder.to(device)
    initial_weights, weights_after_training, train_loss_epochs, val_loss_epochs, model = train_motif_based_encoder(
        valid_genes, encoder, family_size=4, batch_size=256, learning_rate=0.001, temperature=0.1, num_epochs=100)

    # Save the model and the weights values
    os.makedirs(MODEL_OUTPUTS_DIR, exist_ok=True)
    torch.save(initial_weights, f"./{MODEL_OUTPUTS_DIR}/initial_conv_weights.pt")
    torch.save(weights_after_training, f"./{MODEL_OUTPUTS_DIR}/weights_of_conv_after_training.pt")
    torch.save(model, f"./{MODEL_OUTPUTS_DIR}/model_after_training.pt")

    print("The loss values for the training set are:")
    print(train_loss_epochs)
    print()
    print("The loss values for the validation set are:")
    print(val_loss_epochs)
    print()
    print("The program finished running!")

    # Plot loss of training and validation over epochs
    epochs = range(1, len(train_loss_epochs) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_loss_epochs, label='Train Loss')
    plt.plot(epochs, val_loss_epochs, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("loss_curve_during_training.png", dpi=300, bbox_inches="tight")

