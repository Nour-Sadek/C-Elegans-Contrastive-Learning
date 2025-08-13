from Bio import SeqIO
from Bio import Cluster
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import json
import os
import requests
import pandas as pd

import torch
from motif_based_encoder import MotifBasedEncoder, ReverseHomologyModel

ENSEMBL_REST = "https://rest.ensembl.org"
SOURCE_SPECIES = "caenorhabditis_elegans"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def determine_rhiepa_representation(model: MotifBasedEncoder, valid_genes: dict[str, list[torch.tensor]],
                                    file_name: str = "rhiepa_representation") -> None:
    """Save the representation for every gene's promoter in <valid_genes> as determined by the trained model <model> as
    a json file of the form dict[str, dict[str, float]]

    The json file is a dictionary of the form:
    key: gene id (string)
    value: dictionary where the key is the matrix name (generated in the function) and the value is the corresponding
    PAM score for that PWM for that gene. There is a PAM score for every PWM which the model learned during previous
    training. The PAM scores are determined by taking the average of the representation of all the orthologous
    sequences for each gene."""

    representation = {}
    motifs_names = [f"matrix_{i}" for i in range(len(model.PWMs_conv.weight))]
    i = 1
    for gene in valid_genes:
        inputs = torch.stack(valid_genes[gene])  # shape (num_orthologs, num_bases, seq_length)
        inputs = inputs.to(device)
        seqs_embeddings = model(inputs)  # shape (num_orthologs, num_PWMs)
        pam_scores = torch.mean(seqs_embeddings, dim=0)
        representation[gene] = dict(zip(motifs_names, pam_scores.tolist()))
        print(f"Representation for gene {i}: {gene} has been determined.")
        i = i + 1

    # Save the representation as a json file
    with open(f"{file_name}.json", "w") as file:
        json.dump(representation, file, indent=4)


def create_java_treeview_files(file_path: str, gene_ids_names_descriptions_file_path: str, cluster_method: str = "a",
                               distance_function: str = "u") -> None:
    """<file_path> is a string that represents the path to the json file that would be created after calling the
    <determine_rhiepa_representation> function and saving its output as a json file. It is a dictionary of the format:
    key is the gene id and the value is another dictionary where the key is the motif_name and the value is the PAM
    score.

    This function takes this representation file, performs hierarchical clustering with the <cluster_method> clustering
    method and <distance_function> distance function using Cluster 3.0, then saves the resulting tree structure as Java
    TreeView compatible files.

    "a" denotes the pairwise average-linkage clustering method, and "u" denotes the un-centered correlation distance
    function.

    <gene_ids_descriptions_file_path> is a json file that maps each gene id with its corresponding gene name and
    description, if available. If this parameter is given, then instead of the row names being only the gene ids, they
    would also include the gene name and description in this format: gene_id | gene_name | gene_description. This json
    file can be created by running the <get_gene_descriptions_names> function and saving the output dictionary as a json
    file."""

    # Get the file name
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load in the json dictionary
    with open(file_path, "r") as file:
        representation = json.load(file)

    # Convert dictionary to dataframe
    df = pd.DataFrame.from_dict(representation, orient="index")

    # median center the columns (motifs)
    df = df - df.median()

    # Change the row index names from just including gene ids to also including gene names and descriptions if available
    if gene_ids_names_descriptions_file_path is not None:
        # load the gene ids, names, and descriptions dictionary
        with open(gene_ids_names_descriptions_file_path, "r") as file:
            gene_info = json.load(file)
        gene_info = pd.DataFrame(gene_info)

        # Add the gene name and description to each gene id
        # Merge <df> with <gene_info>'s gene_id
        df_reset = df.reset_index().rename(columns={"index": "gene_id"})
        merged_df = pd.merge(df_reset, gene_info, on="gene_id", how="left")

        # Combine the <gene_id>, <gene_name>, and <description> columns into one where the values are separated by |
        merged_df["combined_id"] = merged_df["gene_id"] + " | " + merged_df["gene_name"] + " | " + merged_df[
            "description"]

        # Drop the <gene_id>, <gene_name>, and <description> columns and assign <combined_id> as the new index
        df = merged_df.drop(columns=["gene_id", "gene_name", "description"])
        df = df.set_index("combined_id")

    # Save the <df> as a csv file to be a suitable input to Cluster 3.0
    df.index.name = "FILE"
    df.to_csv(f"{file_name}.txt", sep="\t")
    print("The representation csv file has been created and will be clustered using Cluster 3.0")

    # Load the csv file in Cluster 3.0 version in Python
    handle = open(f"{file_name}.txt")
    record = Cluster.read(handle)

    # Cluster both the motifs and genes, followed by scaling
    gene_tree = record.treecluster(transpose=False, method=cluster_method, dist=distance_function)  # for genes
    gene_tree.scale()
    motif_tree = record.treecluster(transpose=True, method=cluster_method, dist=distance_function)  # for motifs
    motif_tree.scale()

    # Save the tree diagram as files for correct input into Java TreeView
    record.save(file_name, gene_tree, motif_tree)
    print("The Java TreeView files have been saved and the representation heat map is ready to be visualized!")


def pwm_to_meme(PWMs: torch.tensor, alphabet: str = "ACGT") -> str:
    """Return a string that represents the motifs in <PWMs> stored in MEME format with the alphabet <alphabet>."""

    num_cols = len(PWMs[0][0])
    meme = []
    meme.append("MEME version 4\n\n")
    meme.append(f"ALPHABET= {alphabet}\n\n")
    meme.append("strands: + -\n\n")
    meme.append("Background letter frequencies\n")
    meme.append("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

    k = 0

    for pwm in PWMs:

      meme.append(f"MOTIF matrix{k}_length=15\n")
      meme.append(f"letter-probability matrix: alength=4 w={num_cols}\n")

      for i in range(num_cols):
          column = [pwm[j][i] for j in range(4)]  # <alphabet> order
          row = " ".join(f"{val}" for val in column)
          meme.append(" " + row + "\n")
      meme.append("\n")

      k = k + 1

    return "".join(meme)


def convert_fasta_to_json(fasta_folder_path: str, output_folder_path: str) -> None:
    """Given a folder <fasta_folder_path> that contains fasta files, where each file corresponds to the orthologous
    sequences of a gene where each record id is a species name and the corresponding record seqs is the orthologous
    sequence of the gene in that species, convert these files to json files and store them in the <output_folder_path>
    directory where for each gene file, the key would be the species name (record id) and the value would be the
    orthologous sequence (record seq)."""

    # Make the <output_folder_path> directory if it doesn't already exist
    os.makedirs(output_folder_path, exist_ok=True)

    for filename in os.listdir(fasta_folder_path):
        if filename.endswith(".fa"):
            # Convert the fasta file into a dictionary where each record's id is a key and its seq is the value
            fasta_path = os.path.join(fasta_folder_path, filename)
            fasta_dict = {record.id: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")}

            # Save the dictionary as a json file
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(output_folder_path, json_filename)
            with open(json_path, "w") as json_file:
                json.dump(fasta_dict, json_file, indent=4)

    print(f"Successfully converted FASTA files to JSON files and saved to: {output_folder_path}.")


def convert_json_to_fasta(json_folder_path: str, output_folder_path: str) -> None:
    """Given a folder <json_folder_path> that contains json files, where each file corresponds to the orthologous
    sequences of a gene where each key would be the species name and the corresponding value would be the orthologous
    sequence of the gene in that species, convert these files to fasta files and store them in the <output_folder_path>
    directory where for each gene file, the record id would be the species name and the corresponding record seq would
    be the orthologous sequence."""

    # Make the <output_folder_path> directory if it doesn't already exist
    os.makedirs(output_folder_path, exist_ok=True)

    for filename in os.listdir(json_folder_path):
        if filename.endswith(".json"):
            # Convert the json file into a fasta file where each key is the record's description and
            # the key's value is the record's seq
            json_path = os.path.join(json_folder_path, filename)
            with open(json_path) as json_file:
                json_dict = json.load(json_file)
            records = [SeqRecord(Seq(seq), id=key, description="") for key, seq in json_dict.items() if seq != ""]

            # Save the records as a fasta file
            fasta_filename = os.path.splitext(filename)[0] + ".fa"
            fasta_path = os.path.join(output_folder_path, fasta_filename)
            SeqIO.write(records, fasta_path, "fasta")

    print(f"Successfully converted JSON files to FASTA files and saved to: {output_folder_path}.")


def get_gene_descriptions_names(gene_ids: list[str]) -> dict[str: list[str]]:
    """Return a dictionary where there are three different key-value pairs which are parallel lists where the values at
    each position relate to each other; it is in this format to be able to easily be loaded as a pandas dataframe.

    first key is "gene_id" where its value is a list of WormBase gene ids, where those are taken from the <gene_ids>
    list argument
    second key is "gene_name" and third key is "description" where their values are lists whose contents are filled by
    fetch requests using Ensembl's API through the URL: <ENSEMBL_REST>/lookup/id/<gene_id>?expand=1;species=<SOURCE_SPECIES>.
    If that url doesn't contain corresponding values for a gene_id of either, they are replaced with an empty string."""

    gene_info = {"gene_id": [], "gene_name": [], "description": []}
    i = 1
    for gene_id in gene_ids:
        # Get gene info for source species
        url = f"{ENSEMBL_REST}/lookup/id/{gene_id}?expand=1;species={SOURCE_SPECIES}"
        r = requests.get(url, headers={"content-Type": "application/json"})
        if r.ok:
            if i % 1000 == 0:
                print(f"{i} gene id information have been fetched.")
            i = i + 1
            # Get the common gene name, if it is available, else the canonical transcript name
            request = r.json()
            if "display_name" in request:
                gene_name = request["display_name"]
            else:
                if "canonical_transcript" in request:
                    gene_name = request["canonical_transcript"]
                else:
                    gene_name = ""
            # Get the gene description, if it is available
            if "description" in request:
                description = request["description"]
            else:
                description = ""
            # Add the information to the dictionary
            gene_info["gene_id"].append(gene_id)
            gene_info["gene_name"].append(gene_name)
            gene_info["description"].append(description.split("[")[0].strip())
    return gene_info


if __name__ == "__main__":

    model_type = None  # "MotifBasedEncoder" or "ReverseHomologyModel
    model_path = f"../model_outputs_promoter_clad_V_proj_head_PWMs_constrained_fam_size_8/model_after_training.pt"
    meme_file_name = "my_motifs"
    num_PWMs = 256
    PWM_width = 15
    window = 10
    l2 = 64
    l3 = 64

    if model_type in ["MotifBasedEncoder", "ReverseHomologyModel"]:

        if model_type == "MotifBasedEncoder":
            model = MotifBasedEncoder(num_PWMs=num_PWMs, PWM_width=PWM_width, window=window, num_bases=4)
            model.load_state_dict(torch.load(model_path))
            PWMs = model.PWM_constraint(model.PWMs_conv.weight)

        else:  # model_type == "ReverseHomologyModel"
            encoder = MotifBasedEncoder(num_PWMs=num_PWMs, PWM_width=PWM_width, window=window, num_bases=4)
            model = ReverseHomologyModel(encoder, num_PWMs, l2, l3)
            model.load_state_dict(torch.load(model_path))
            PWMs = model.motif_based_encoder.PWM_constraint(model.motif_based_encoder.PWMs_conv.weight)

        # Convert the PWMs into MEME format
        meme_content = pwm_to_meme(PWMs)

        # Save the PWMs in MEME format into a file
        with open(f"{meme_file_name}.txt", "w") as f:
            f.write(meme_content)

        print(f"{num_PWMs} motifs have been saved in MEME format in {meme_file_name}.txt file.")
