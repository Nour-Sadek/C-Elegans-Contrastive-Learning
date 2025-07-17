import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import json
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

from motif_based_encoder import one_hot_encode_seq, pad_one_hot_encoded_seq, MotifBasedEncoder

# Constants
gb = torch.tensor([0.25, 0.25, 0.25, 0.25])
BASE_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}  # bases are of the order ACGT
FILES_DIR = "./ortholog_promoters_per_gene"
TRAIN_LENGTH = 800
TRAIN_MIN_NUM_ORTHOLOGS = 9
TRAIN_MIN_SEQ_LENGTH = 80
MODEL_OUTPUTS_DIR = "model_outputs_idr"

REPRESENTATION_LENGTH = 500
REPRESENTATION_MIN_NUM_ORTHOLOGS = 2
REPRESENTATION_MIN_SEQ_LENGTH = 15
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.devide("cpu")

os.makedirs(MODEL_OUTPUTS_DIR, exist_ok=True)


"""TRAIN_MIN_NUM_ORTHOLOGS = 5 => 13354 genes; TRAIN_MIN_NUM_ORTHOLOGS = 9 => 12403 genes (335816 sequences)."""


def read_files(files_dir: str, target_length: int, min_num_orthologs: int, min_sequence_length: int,
               base_to_index: dict[str, int]) -> dict[str, list[torch.tensor]]:
    """Return a dictionary of the form:
    key: gene id (string)
    value: list of tensors where each is a one-hot encoded sequence of an orthologous sequence

    <files_dir> is the path to the directory that has the json files where each file's name is a gene id and its
    contents is a dictionary of the form:
    key: name of the species
    value: string of the promoter sequence corresponding to the ortholog of the gene which is the name of the file

    The conditions for the inclusion of the gene and the valid orthologous promoters are as follows, for each gene:
    - The length of the promoter sequence is at least <min_sequence_length>
    - After the first condition is satisfied, there needs to be at least <min_num_orthologs> sequences remaining for the
    gene to be a valid gene for further processing

    For the sequences of the genes that satisfied the above two conditions, they are further processed by one-hot
    encoding to a tensor using the nucleotides order as specified by <base_to_index>, then their lengths fixed to be
    <target_length> where if the sequence is shorter than <target_length>, it is zero-padded on the right and if the
    sequence is longer  than <target_length>, it is truncated from the start till it reached <target_length>.

    For the current run, a <target_length> of 800bp for training and 500bp for encoding the promoter sequences after
    training is used. <min_num_orthologs> is 9 (family_size of 8 + 1 target) for training and 2 for encoding after
    training. <min_sequence_length> is 10% of <target_length> (80bp) for training and <PWM_width> (15) for encoding
    after training."""

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
                one_hot_seq = one_hot_seq[:, (len(sequence) - target_length):]
            fixed_valid_sequences.append(one_hot_seq)
        # Save the promoters of this gene as valid to be passed through the encoder
        valid_genes[gene_name] = fixed_valid_sequences

    return valid_genes


def split_data(valid_genes: dict[str, list[torch.tensor]], train_ratio: float = 0.9,
               seed: int = 2025) -> tuple[dict[str, list[torch.tensor]], dict[str, list[torch.tensor]]]:
    """Return two dictionaries where they contain <train_ratio> and 1 - <train_ratio> of the data in <valid_genes>
    respectively and that split is pseudo-randomly determined based on the <seed> for reproducibility.

    This function will be used to create train and validation data sets."""

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


def split_into_batches(genes_list: list[str], batch_size: int) -> list[list[str]]:
    """Return a list of inner lists where each inner list is of size <batch_size> (excluding the last inner list that
    might contain fewer items if len(<genes_list>) is not divisible by <batch_size>) and contains gene names. These
    gene names are taken from <genes_list> which is a list of gene names which should be the names of the genes that
    have been deemed as valid for further processing.

    This split of gene names into inner lists is done randomly, and each inner list is going to be used as a batch for
    training/validation."""

    # randomly shuffle the genes
    random.shuffle(genes_list)

    # Split into inner lists of genes of size <batch_size>
    genes_in_batches = [genes_list[i:i + batch_size] for i in range(0, len(genes_list), batch_size)]

    return genes_in_batches


def get_encoder_inputs_from_batch(genes_dict: dict[str, list[torch.tensor]], batch_of_genes: list[str],
                                  family_size: int) -> torch.tensor:
    """Return the inputs of the encoder for each batch, which is a tensor of shape (num_seqs, num_bases, seq_length).

    <genes_dict> is a dictionary of the form: key (gene id) and value (list of one-hot encoded tensors for the
    orthologous promoter sequences). <geneS_dict> could either be the train set or validation set which are the result
    of calling the function <split_data>. <batch_of_genes> is a list of gene names that represent the genes whose
    sequences will be processed for the current batch; it is one of the inner lists of the list which is the result of
    calling the function <split_into_batches>.

    <num_seqs> is determined as follows:
    - For each gene in <batch_of_genes>, extract randomly <family_size> + 1 one-hot encoded orthologous sequences from
    <genes_dict>. Each of these orthologous sequences are a tensor of shape (num_bases, seq_length).
    - These sequences for each gene are stacked on top of each other and so <num_seqs> = (<family_size> + 1) *
    len(<batch_of_genes>)
    <num_bases> = 4 (4 nucleotides)
    <seq_length> = target_length which was used during the call of the function <read_files>, which is 800 for sequences
    used for training and 500 for sequences used for encoding."""

    # Create a nested list of sequences per gene
    batch_of_seqs = [random.sample(genes_dict[gene], family_size + 1) for gene in batch_of_genes]

    # Flatten the list so that all sequences are in one list (first <family_size> + 1 seqs refer to first gene,
    # followed by seqs for the second gene and so on)
    batch_of_seqs_flat = [seq for list_of_seqs in batch_of_seqs for seq in list_of_seqs]

    # Stack the seqs into a single tensor of shape (num_seqs, num_bases, seq_length) where the shape of the seqs was
    # (num_bases, seq_length)
    inputs = torch.stack(batch_of_seqs_flat)

    return inputs  # shape (num_seqs, num_bases, seq_length)


def get_seqs_outside_batch(genes_dict: dict[str, list[torch.tensor]],
                           batch_of_genes: list[str], n: int) -> list[torch.tensor]:
    """Return a list of <n> tensors which are one-hot encoded sequences of shape (num_bases, seq_length). Each sequence
    is randomly chosen from the list of orhologous sequences, stored in <genes_dict> for genes not part of genes in
    <batch_of_genes>. These sequences will act as the negative sequences in the target sets for the current batch of
    genes <batch_of_genes>.

    <n> is determined as follows:
    - there is a <target_set_size> chosen during training that determines the number of sequences in the target set,
    which is made up of 1 positive target sequence and <target_set_size> - 1 negative target sequences. One target
    sequence per gene in the batch are chosen first as target sequences, and if <target_set_size> is greater than
    len(<batch_of_genes>), the remaining genes n = <target_set_size> - len(<batch_of_genes>) to act as negative
    sequences in the target set are chosen randomly from groups of orthologous sequences from genes not part of the
    current batch where one sequence per gene is chosen."""

    # Get the genes in <genes_dict> that are not included in <batch_of_genes>
    non_batch_genes = [gene for gene in genes_dict.keys() if gene not in batch_of_genes]

    # Randomly choose <n> genes from <non_batch_genes> and randomly choose one sequence from each randomly chosen gene
    if len(non_batch_genes) >= n:
        chosen_genes = random.sample(non_batch_genes, n)
        negative_sequences_outside_batch = [random.sample(genes_dict[gene], 1)[0] for gene in chosen_genes]
    else:
        raise ValueError(f"Number of genes outside of the current batch {len(non_batch_genes)} is less than the number "
                         f"of genes required to reach target set size.")

    return negative_sequences_outside_batch


def calculate_logits(seqs_embeddings: torch.tensor, family_size: int, num_genes_in_batch: int, target_set_size: int,
                     temperature: float) -> tuple[torch.tensor, torch.tensor]:
    """Return a tuple of labels and logits tensors where the labels tensor, for each gene family in the batch, specifies
    the index of the class that would be considered as the correct classification (the index of the dot product between
    the family embeddings and the positive target) and the logits tensor specifies, also for each gene family in the
    batch, the dot product between the family embedding and the target set, where the target set is made up of 1
    positive target and <target_set_size> - 1 negative targets. These dot products are scaled by <temperature>.

    The size of the labels return value tensor is <num_genes_in_batches> and the shape of the logits return value
    tensor is (num_genes_in_batches, target_set_size>).

    The input to the encoder whose output is <seqs_embeddings> tensor, which is of shape (num_seqs, num_PWMs) where
    num_seqs = (<num_genes_in_batch> * (<family_size> + 1) + (target_set_size - <num_genes_in_batches>)), follows this
    order:
    - There are <num_genes_in_batch> gene families where each family has <family_size> + 1 orthologous sequences,
    followed by <target_set_size> - <num_genes_in_batches> out-of-batch sequences where each sequence belongs to a
    different out-of-batch gene family.
    - Every (family_size) + 1 representations are a group that represent the representations of orthologous sequences
    from a single gene
    - The first <family_size> representations would be used to represent the family embeddings of that gene and the last
    sequence in that group would be used as the positive target for the current gene family and a negative target for
    the other gene families.
    - The last (target_set_size - <num_genes_in_batches>) representations represent individual sequences from
    out-of-batch genes that would in addition to the other <num_genes_in_batches> target sequences, be used as negative
    sequences in the target set. If <num_genes_in_batches> is equal to <target_set_size>, these sequences would not be
    needed.

    The temperature-scaled dot product between the family embeddings and the positive target is placed at index 0 and
    that between the family embeddings and the negative targets are placed at the rest of the indices for each gene
    family, and so the labels return output tensor would be a tensor of 0s as the correct classifications would be at
    index 0 for each gene family."""

    seqs_embeddings = F.normalize(seqs_embeddings, dim=1)
    num_batch_seqs = num_genes_in_batch * (family_size + 1)
    # Create the labels (indicator variable)
    labels = torch.zeros(num_genes_in_batch, dtype=torch.long)  # positive class is at zero index for each query

    # Create the logits variable which represents dot products between the family embeddings and single sequences,
    # scaled by temperature
    logits = torch.zeros(num_genes_in_batch, target_set_size)

    # Get the indices of all negative samples (anchors + negative samples from outside of the current batch)
    all_batch_negative_samples_indices = [i for i in range(family_size, num_batch_seqs, family_size + 1)]
    all_non_batch_negative_indices = [i for i in range(all_batch_negative_samples_indices[-1] + 1, len(seqs_embeddings))]
    all_negative_samples_indices = all_batch_negative_samples_indices + all_non_batch_negative_indices

    curr_sample = 0
    for start_of_family in range(0, num_batch_seqs, family_size + 1):
        # Get the family embedding representation
        family_tensors = seqs_embeddings[start_of_family:(start_of_family + family_size)]
        family_embedding = torch.mean(family_tensors, dim=0)  # shape (num_PWMs,)
        family_embedding = F.normalize(family_embedding, dim=0)

        # Get the positive anchor embedding
        positive_anchor = seqs_embeddings[start_of_family + family_size]  # shape (num_PWMs,)
        positive_anchor = positive_anchor.unsqueeze(0)  # shape (1, num_PWMs)
        # Extract the negative single sequence embeddings
        curr_negative_samples = [index for index in all_negative_samples_indices if index != start_of_family + family_size]
        negative_sequences_embeddings = seqs_embeddings[curr_negative_samples]  # shape (<target_set_size> - 1, num_PWMs)

        # Combine the embeddings of the single sequences where the positive anchor embedding is on top and all other
        # negative single sequences are under it
        single_sequences_embeddings = torch.cat((positive_anchor, negative_sequences_embeddings), dim=0)  # shape (<target_set_size>, num_PWMs)

        # Calculate the dot product between the family and single sequences embeddings, scaled by temperature
        family_scores = torch.matmul(single_sequences_embeddings, family_embedding) / temperature

        # Update the logits variable
        logits[curr_sample] = family_scores
        curr_sample = curr_sample + 1

    return labels, logits


def infoNCE_loss(seqs_embeddings: torch.tensor, family_size: int, num_genes_in_batch: int, target_set_size: int,
                 temperature: float):
    """Return the infoNCE loss for the samples in <seqs_embeddings>, which is of shape (num_seqs, num_PWMs). Excluding
    the last batch, num_seqs = (<num_genes_in_batch> * (<family_size> + 1) + (target_set_size - <num_genes_in_batches>)).

    The docstring of the <calculate_logits> function explains in detail how the infoNCE loss is calculated."""

    # Determine the labels and logits
    labels, logits = calculate_logits(seqs_embeddings, family_size, num_genes_in_batch, target_set_size, temperature)

    # Calculate the Categorical Cross Entropy loss between the indicator variable <labels> and <logits>
    # (does softmax internally)
    loss = F.cross_entropy(logits, labels)

    return loss


def evaluate_representations(model: MotifBasedEncoder, data_set: dict[str, list[torch.tensor]], family_size: int,
                             batch_size: int, target_set_size: int, temperature: float) -> float:
    """Return the accuracy of the model <model> of identifying the positive target as the correct target (classification
    accuracy) for samples in <data_set>. <data_set> here can be either the <val_set> or <train_set>. The expectation for
    a random classifier is 1/<target_set_size>, and since the accuracy is dependent on the logits produced by the
    contrastive infoNCE loss function, parameters such as the <family_size> and <temperature> are needed."""

    correct = 0
    total = 0

    # Set up the validation loop to calculate the validation loss after one epoch of training
    model.eval()
    # Split the validation set into groups of size <batch_size>
    all_genes = list(data_set.keys())
    genes_in_batches = split_into_batches(all_genes, batch_size)

    with torch.no_grad():
        for batch_of_genes in genes_in_batches:
            num_genes_in_batch = len(batch_of_genes)
            inputs = get_encoder_inputs_from_batch(data_set, batch_of_genes, family_size)
            # Fetch the extra negative sequences from genes outside the current batch
            num_non_batch_negative_seqs_to_add = target_set_size - num_genes_in_batch
            negative_seqs_outside_batch = get_seqs_outside_batch(data_set, batch_of_genes,
                                                                 num_non_batch_negative_seqs_to_add)
            # Add those sequences to the end of the <inputs>
            inputs = torch.cat([inputs, torch.stack(negative_seqs_outside_batch)], dim=0)
            inputs = inputs.to(device)
            seqs_embeddings = model(inputs)

            labels, logits = calculate_logits(seqs_embeddings, family_size, num_genes_in_batch, target_set_size, temperature)
            predicted = torch.argmax(logits, dim=1)  # shape (len(batch_of_genes),)

            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()

    accuracy = round((correct / total) * 100, 2)
    return accuracy


def train_motif_based_encoder(train_set: dict[str, list[torch.tensor]], val_set: dict[str, list[torch.tensor]],
                              model: MotifBasedEncoder, family_size: int, batch_size: int, learning_rate: float,
                              temperature: float, num_epochs: int,
                              target_set_size: int) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return a tuple of 4 lists of floats, each of size <num_epochs>, that specify the training loss, validation loss,
    training accuracy, and validation accuracy after every training epoch respectively. The model <model> is used and
    trained on <train_set>, and validated at every epoch on <val_set>.

    The adam optimizer with learning rate <learning_rate> is used and a learning rate scheduler where the learning rate
    decreases if the validation accuracy plateaus with patience of 10. The samples are batched with batch size
    <batch_size> where the <split_into_batches> function is used to split the <train_set> and <val_set> into batches and
    then for every batch of genes, the <get_encoder_inputs_from_batch> function is used to get the sequences in the
    right form for being input into the encoder, which requires the <family_size> parameter. <model> uses the infoNCE
    loss for parameter updates, which requires the <temperature> parameter, and the <target_set_size> parameter to
    determine the number of target set sequences to be used for every family embedding.

    After every parameter update, different constraints to the <model> weights are enforced:
    - the weights for the PWM_convs layer in <model> are constrained by the PWM_constraint module.
    - the bias parameter of the TrainableScaling layer and the pooling parameter of the TrainablePooling layer are
    clamped to a minimum value of 0
    - the scale parameter of the TrainableScaling layer is clamped to a minimum value of 0.1."""

    # Set up the model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    # Set up the training loop
    train_loss_epochs = []
    val_loss_epochs = []
    train_accuracy_epochs = []
    val_accuracy_epochs = []
    for epoch in range(num_epochs):
        model.train()
        # Randomly split the training set into groups of size <batch_size>
        all_train_genes = list(train_set.keys())
        genes_in_batches = split_into_batches(all_train_genes, batch_size)
        train_loss_step = []
        # Loop over every batch of genes
        for batch_of_genes in genes_in_batches:
            optimizer.zero_grad()
            num_genes_in_batch = len(batch_of_genes)

            # Only keep <family_size> + 1 orthologous sequences for each gene and consider the first <family_size>
            # elements to be the query set (represent the family embeddings) and the last element to be the target set
            # for that family.
            inputs = get_encoder_inputs_from_batch(train_set, batch_of_genes, family_size)  # shape (num_seqs, num_bases, seq_length)
            # Fetch the extra negative sequences from genes outside the current batch
            num_non_batch_negative_seqs_to_add = target_set_size - num_genes_in_batch
            negative_seqs_outside_batch = get_seqs_outside_batch(train_set, batch_of_genes,
                                                                 num_non_batch_negative_seqs_to_add)
            # Add those sequences to the end of the <inputs>
            inputs = torch.cat([inputs, torch.stack(negative_seqs_outside_batch)], dim=0)
            inputs = inputs.to(device)
            seqs_embeddings = model(inputs)  # shape (num_seqs, num_PWMs) (num_seqs, l3)
            loss = infoNCE_loss(seqs_embeddings, family_size, num_genes_in_batch, target_set_size, temperature)
            loss.backward()
            optimizer.step()
            # Implement constraints on multiple layers
            with torch.no_grad():
                # Implement the constraint on the PWM motifs
                model.PWMs_conv.weight.data.copy_(model.PWM_constraint(model.PWMs_conv.weight.data))
                # Implement the constraint on the Scaling layer
                model.scaling_layer.scale.clamp_(min=0.1)
                model.scaling_layer.bias.clamp_(min=0)
                # Implement the constraint on the pooling layer
                model.pooling_layer.pooling.clamp_(min=0)
            train_loss_step.append(loss.item())
        train_loss_epochs.append(np.array(train_loss_step).mean())

        # Set up the validation loop to calculate the validation loss after one epoch of training
        model.eval()
        # Split the validation set into groups of size <batch_size>
        all_val_genes = list(val_set.keys())
        genes_in_batches = split_into_batches(all_val_genes, batch_size)
        val_loss_step = []
        with torch.no_grad():
            for batch_of_genes in genes_in_batches:
                num_genes_in_batch = len(batch_of_genes)
                inputs = get_encoder_inputs_from_batch(val_set, batch_of_genes, family_size)
                # Fetch the extra negative sequences from genes outside the current batch
                num_non_batch_negative_seqs_to_add = target_set_size - num_genes_in_batch
                negative_seqs_outside_batch = get_seqs_outside_batch(val_set, batch_of_genes,
                                                                     num_non_batch_negative_seqs_to_add)
                # Add those sequences to the end of the <inputs>
                inputs = torch.cat([inputs, torch.stack(negative_seqs_outside_batch)], dim=0)
                inputs = inputs.to(device)
                seqs_embeddings = model(inputs)
                loss = infoNCE_loss(seqs_embeddings, family_size, num_genes_in_batch, target_set_size, temperature)
                val_loss_step.append(loss.item())
            val_loss_epochs.append(np.array(val_loss_step).mean())

        scheduler.step(val_loss_epochs[epoch])

        # Determine the current accuracy of the model
        training_accuracy = evaluate_representations(model, train_set, family_size, batch_size, target_set_size, temperature)
        validation_accuracy = evaluate_representations(model, val_set, family_size, batch_size, target_set_size, temperature)
        train_accuracy_epochs.append(training_accuracy)
        val_accuracy_epochs.append(validation_accuracy)

        # Print the loss of the training and validation data sets after one epoch
        print(
            f"After epoch {epoch + 1}: training loss = {train_loss_epochs[epoch]}, validation loss = {val_loss_epochs[epoch]}, \n"
            f"               training accuracy = {training_accuracy}, validation accuracy = {validation_accuracy}")

    print("Finished Training!")

    return train_loss_epochs, val_loss_epochs, train_accuracy_epochs, val_accuracy_epochs


def get_rhiepa_representations(model: MotifBasedEncoder,
                               valid_genes: dict[str, list[torch.tensor]]) -> dict[str, list[float]]:
    """Return the representation for every gene's promoter in <valid_genes> as determined by the trained model <model>.
    The return value is a dictionary of the form:
    key: gene id (string)
    value: list of pam scores of size num_PWMs where there is a PAM score for every PWM which the model learned during
    previous training. The PAM scores are determined by taking the average of the representation of all the orthologous
    promoter sequences for each gene."""

    representation = {}
    for gene in valid_genes:
        inputs = torch.stack(valid_genes[gene])  # shape (num_orthologs, num_bases, seq_length)
        inputs = inputs.to(device)
        seqs_embeddings = model(inputs)  # shape (num_orthologs, num_PWMs)
        pam_scores = torch.mean(seqs_embeddings, dim=0)
        representation[gene] = pam_scores.tolist()
    return representation


if __name__ == "__main__":

    # Get the valid genes
    valid_genes = read_files(FILES_DIR, target_length=TRAIN_LENGTH, min_num_orthologs=TRAIN_MIN_NUM_ORTHOLOGS,
                             min_sequence_length=TRAIN_MIN_SEQ_LENGTH, base_to_index=BASE_TO_INDEX)

    # Save <valid_genes> to load later (it is taking a while...)
    file_path = f"./{MODEL_OUTPUTS_DIR}/valid_genes_for_encoder.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(valid_genes, f)

    # Determine the number of valid genes and orthologous sequences
    seqs_num = 0
    for gene in valid_genes:
        seqs_num = seqs_num + len(valid_genes[gene])
    print(f"{len(valid_genes)} will be processed through the encoder with total {seqs_num} orthologous sequences.")

    # load back the valid genes (if needed)
    # file_path = f"./valid_genes_for_encoder_fam_size_8.pkl"
    # with open(file_path, "rb") as f:
    #     valid_genes = pickle.load(f)
    print(f"The training of the model will start with {len(valid_genes)} genes.")

    # Split the data into training and validation data sets
    train_set, val_set = split_data(valid_genes)

    # Train the model on the valid genes
    encoder = MotifBasedEncoder(num_PWMs=256, PWM_width=15, window=10, num_bases=4, set_initial_values=True)
    encoder.to(device)

    # Save the initial weights of the model
    torch.save(encoder.state_dict(), f"./{MODEL_OUTPUTS_DIR}/model_before_training.pt")

    # Train the encoder
    train_loss_epochs, val_loss_epochs, train_accuracy_epochs, val_accuracy_epochs = train_motif_based_encoder(
        train_set, val_set, encoder, family_size=8, batch_size=256, learning_rate=0.1, temperature=0.1, num_epochs=100,
        target_set_size=400)

    # Save the model after training
    torch.save(encoder.state_dict(), f"./{MODEL_OUTPUTS_DIR}/model_after_training.pt")
    
    # How to load the model back
    # model = MotifBasedEncoder(num_PWMs=256, PWM_width=15, window=10, num_bases=4)
    # model.load_state_dict(torch.load(f"./{MODEL_OUTPUTS_DIR}/model_after_training.pt"))

    print("The loss values for the training set are:")
    print(train_loss_epochs)
    print()
    print("The loss values for the validation set are:")
    print(val_loss_epochs)
    print()
    print("The accuracy values for the training set are:")
    print(train_accuracy_epochs)
    print()
    print("The accuracy values for the validation set are:")
    print(val_accuracy_epochs)
    print()
    print("The program finished running!")

    # Plot loss and accuracy of training and validation sets over epochs during training
    epochs = range(1, len(train_loss_epochs) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_epochs, label='Train Loss')
    plt.plot(epochs, val_loss_epochs, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy_epochs, label='Train Accuracy')
    plt.plot(epochs, val_accuracy_epochs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"./{MODEL_OUTPUTS_DIR}/plots.png", dpi=300, bbox_inches="tight")
