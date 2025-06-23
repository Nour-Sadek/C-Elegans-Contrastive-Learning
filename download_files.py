import pandas as pd
import ftplib
import gzip
import os
from Bio import SeqIO


# Define FTP server, urls, and WBPS release number
FTP_HOST = "ftp.ebi.ac.uk"
WBPS_RELEASE_NUMBER = "WBPS19"
BASE_PATH = f"/pub/databases/wormbase/parasite/releases/{WBPS_RELEASE_NUMBER}/species/"
FILE_TYPES = ["genomic.fa.gz", "annotations.gff3.gz", "protein.fa.gz"]

nematoda_species = pd.read_excel(
    "species_Nematoda--___.xlsx")  # 65 species belong to Clade V (78 total bio projects exist for these 65 species)

# Save the species name and corresponding BioProjectID for Clade V species
clade_V_genomes_info = {}

clade_V = nematoda_species[nematoda_species["Clade"] == "Clade V"]
clade_V_species_unique = list(clade_V["Species Name"].unique())

# For the species that has multiple BioProjects, choose the bio-project that has the greater number of coding genes
for species in clade_V_species_unique:
    entries = clade_V[clade_V["Species Name"] == species]
    if len(entries) == 1:
        bioproject_id = entries["BioProject ID"].item()
    else:
        row_of_choice = entries.loc[entries['Number of Coding Genes'].idxmax()]
        bioproject_id = row_of_choice["BioProject ID"]
    clade_V_genomes_info[species] = bioproject_id

# Save the database that has information only for the species that will be considered
# filtered = clade_V[clade_V.apply(lambda row: clade_V_genomes_info.get(row['Species Name']) == row['BioProject ID'], axis=1)]
# filtered.to_excel("clade_V_info.xlsx", index=False)

# Connect to FTP
ftp = ftplib.FTP(FTP_HOST)
ftp.login()

# Download the annotations file (gff3) and the genomic and protein sequences for each clade V species
for species, bioproject_id in clade_V_genomes_info.items():
    # Change the name of the species to a FTP server compatible one
    species_name = species.lower().replace("_", "").split(" ")
    if "sp." in species_name:
        species_name.remove("sp.")
    species_name = "_".join(species_name)
    # Prepare path for download
    dir_path = f"{BASE_PATH}{species_name}/{bioproject_id}/"
    try:
        ftp.cwd(dir_path)
        filenames = ftp.nlst()  # list of file names available in that directory
        for file_type in FILE_TYPES:
            matching_files = [f for f in filenames if f.endswith(file_type)]
            for filename in matching_files:
                print(f"Downloading {filename} from {dir_path}")
                with open(filename, "wb") as f:
                    ftp.retrbinary(f"RETR {filename}", f.write)
    except ftplib.error_perm:
        print(f"Directory not found: {dir_path}")


# Filter the protein files so that only the longest protein isoform is kept

PROTEINS_DIR = "./protein_sequences"
FILTERED_PROTEINS_DIR = "./filtered_protein_sequences"


def get_gene_id(description):
    description_list = description.split()  # split by any kind of whitespace
    # first check for the presence of the "gene" descriptor
    for item in description_list:
        if item.startswith("gene="):
            return item[5:]
    # if the "gene" descriptor does not exist, then search for the "transcript" descriptor
    for item in description_list:
        if item.startswith("transcript="):
            return item[11:]
    # if neither exist, return None
    return None


def keep_longest_isoform(file):
    basename = file.replace(".protein.fa.gz", "")
    input_path = os.path.join(PROTEINS_DIR, file)
    output_path = f"{FILTERED_PROTEINS_DIR}/{basename}_filtered.protein.fa"

    longest_isoforms = {}

    with gzip.open(input_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            gene_id = get_gene_id(record.description)

            current_longest = longest_isoforms.get(gene_id)

            if not current_longest or len(record.seq) > len(current_longest.seq):
                longest_isoforms[gene_id] = record

    with open(output_path, "w") as out_handle:
        SeqIO.write(longest_isoforms.values(), out_handle, "fasta")


i = 1
for file in os.listdir(PROTEINS_DIR):
    if file.endswith(".protein.fa.gz"):
        keep_longest_isoform(file)
        print(f"Saved filtered file {i} for {file}")
        i = i + 1
