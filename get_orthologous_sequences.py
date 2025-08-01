import pandas as pd
import json
import os
from Bio import SeqIO

COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
WBPS_RELEASE_NUMBER = "WBPS19"
SOURCE_SPECIES = "caenorhabditis_elegans"
PROMOTER_MAX_LEN = 800
UTR_MAX_LEN = 200

# Extract species name and bio project ids per species genome annotations
clade_V_info = pd.read_excel("clade_V_info.xlsx")
species = list(clade_V_info["Species Name"])
bio_project_ids = list(clade_V_info["BioProject ID"])


def reverse_compliment(seq):
    """Return the reverse complement of a string of bases"""
    return "".join(COMPLEMENT[base] for base in reversed(seq))


def format_species_name(species_name):
    """Return a formatted form of <species_name> that corresponds to the format used in the file names.
    Examples:
        - Caenorhabditis Elegans would be formatted to caenorhabditis_elegans.
        - Theristus sp. LFF4_11 would be formatted to theristus_lff411
    """
    formatted_name = species_name.lower().replace("_", "").split(" ")
    if "sp." in formatted_name:
        formatted_name.remove("sp.")
    formatted_name = "_".join(formatted_name)
    return formatted_name


# Go over every gff3 file for every species and extract info about every transcript
# {chromosome, strand, start, end, parent_gene} as well as the sorted genes per chromosome in ascending order

GENERAL_INFO_DIR = "general_info"
GENES_SORTED_DIR = "genes_sorted"

os.makedirs(GENERAL_INFO_DIR, exist_ok=True)
os.makedirs(GENES_SORTED_DIR, exist_ok=True)

i = 1
GFF_DIR = "./annotation_sequences"
for file_name in os.listdir(GFF_DIR):
    species_name = file_name.split(".")[0]
    path = os.path.join(GFF_DIR, file_name)

    mRNA_info = {}
    genes_sorted = {}
    with open(path, "r") as gff:
        for line in gff:
            if line.startswith("#"):
                continue
            features = line.strip().split()
            if features[2] not in ["mRNA", "gene"]:
                continue
            chrom, feature_type, start, end, strand, description = features[0], features[2], int(features[3]), int(
                features[4]), features[6], features[8]
            # Go through the attributes and extract the transcript_id and parent gene_id for this mRNA
            if feature_type == "mRNA":
                transcript_id = None
                gene_id = None
                description_list = description.split(";")
                for attribute in description_list:
                    # Get the transcript_id
                    if attribute.startswith("ID="):
                        if attribute.startswith("ID=transcript:") or attribute.startswith("ID=Transcript:"):
                            transcript_id = attribute[14:]
                        else:
                            transcript_id = attribute[3:]
                    # Get the gene_id
                    if attribute.startswith("Parent="):
                        if attribute.startswith("Parent=gene:") or attribute.startswith("Parent=Gene:"):
                            gene_id = attribute[12:]
                        else:
                            gene_id = attribute[7:]
                # Save the info for the current mRNA transcript
                mRNA_info[transcript_id] = {"chromosome": chrom,
                                            "strand": strand,
                                            "start": start,
                                            "end": end,
                                            "parent_gene_id": gene_id}

            # Extract this gene's id and start coordinate in the current chromosome
            if feature_type == "gene":
                # Get the gene_id
                gene_id = None
                description_list = description.split(";")
                for attribute in description_list:
                    if attribute.startswith("ID="):
                        if attribute.startswith("ID=gene:") or attribute.startswith("ID=Gene:"):
                            gene_id = attribute[8:]
                        else:
                            gene_id = attribute[3:]
                    # Save the info of the current gene's chromosome and start location
                    if chrom not in genes_sorted:
                        genes_sorted[chrom] = {gene_id: (start, end)}
                    else:
                        genes_sorted[chrom][gene_id] = (start, end)

    # Sort the genes in each chromosome based on their start coordinates
    genes_sorted = {
        chrom: dict(sorted(genes.items(), key=lambda item: item[1][0]))
        for chrom, genes in genes_sorted.items()
    }

    # Save the mRNA_info and genes_sorted as separate json files
    file_name = f"./{GENERAL_INFO_DIR}/{species_name}_transcripts_general_info.json"
    with open(file_name, "w") as file:
        json.dump(mRNA_info, file, indent=4)
    file_name = f"./{GENES_SORTED_DIR}/{species_name}_genes_sorted.json"
    with open(file_name, "w") as file:
        json.dump(genes_sorted, file, indent=4)

    print(f"Finished processing {i} {species_name} gff3 annotation file.")
    i = i + 1

# Now that the info for every transcript has been saved, go through every transcript and determine the start and end
# coordinates of the promoter and 3'UTR regions associated with each transcript for each species

GENERAL_PROMOTERS_INFO_DIR = "general_info_promoters"
GENERAL_UTR_INFO_DIR = "general_info_UTRs"
os.makedirs(GENERAL_PROMOTERS_INFO_DIR, exist_ok=True)
os.makedirs(GENERAL_UTR_INFO_DIR, exist_ok=True)

i = 1
for species_name in species:

    # Get the correct format for species name (e.g.: caenorhabditis_elegans instead of Caenorhabditis Elegans)
    species_name = format_species_name(species_name)

    # Get the transcripts general info file for current species
    path = f"./{GENERAL_INFO_DIR}/{species_name}_transcripts_general_info.json"
    with open(path, "r") as file:
        transcripts_info = json.load(file)
    # Get the order of genes per chromosome for current species
    path = f"./{GENES_SORTED_DIR}/{species_name}_genes_sorted.json"
    with open(path, "r") as file:
        genes_order = json.load(file)

    promoters_info = {}
    UTRs_info = {}

    # Go through every transcript and determine the start and end coordinates for the associated promoter and 3'UTR
    # making sure that it doesn't extend to the territory of any other gene
    for transcript_id, info in transcripts_info.items():
        # find the position of this transcript's parent gene
        gene_id = info["parent_gene_id"]
        genes_of_chrom = genes_order[info["chromosome"]]
        genes = list(genes_of_chrom.keys())
        idx_curr_gene = genes.index(gene_id)

        # promoter or 3'UTR might extend to nearby genes' territory so have to check the coordinates of near-by genes
        # Due to artifact in some gff files, have to check if the nearby genes have the same coordinates as the
        # current gene, if yes move on to the next nearby gene until we reach a gene that has different coordinates

        # Find value of previous gene end
        while True:
            prev_gene = genes[idx_curr_gene - 1] if idx_curr_gene > 0 else None
            if prev_gene is None or (
                    genes_of_chrom[prev_gene][0] != genes_of_chrom[gene_id][0] and genes_of_chrom[prev_gene][1] <
                    genes_of_chrom[gene_id][0]):
                break
            idx_curr_gene = idx_curr_gene - 1
        if prev_gene is not None:
            prev_gene_end = genes_of_chrom[prev_gene][1]
        else:
            prev_gene_end = 0

        # Find value of next gene start
        while True:
            next_gene = genes[idx_curr_gene + 1] if idx_curr_gene < len(genes) - 1 else None
            if next_gene is None or (
                    genes_of_chrom[next_gene][0] != genes_of_chrom[gene_id][0] and genes_of_chrom[next_gene][0] >
                    genes_of_chrom[gene_id][1]):
                break
            idx_curr_gene = idx_curr_gene + 1
        if next_gene is not None:
            next_gene_start = genes_of_chrom[next_gene][0]
        else:
            # region might extend beyond the chromosome range, will be handled when sequences are extracted
            next_gene_start = float('inf')

        # Determine the coordinates for the promoter and 3'UTR regions
        if info["strand"] == "+":
            # Find the tentative promoter positions
            promoter_start = info["start"] - PROMOTER_MAX_LEN
            promoter_end = info["start"] - 1
            # Find the tentative 3'UTR positions
            utr_start = info["end"] + 1
            utr_end = info["end"] + UTR_MAX_LEN
            # Update position if it extends to nearby genes
            promoter_start = max(promoter_start, prev_gene_end + 1)
            utr_end = min(utr_end, next_gene_start - 1)

        elif info["strand"] == "-":
            # Find the tentative promoter positions
            promoter_start = info["end"] + 1
            promoter_end = info["end"] + PROMOTER_MAX_LEN
            # Find the tentative 3'UTR positions
            utr_start = info["start"] - UTR_MAX_LEN
            utr_end = info["start"] - 1
            # Update position if it extends to nearby genes
            promoter_end = min(promoter_end, next_gene_start - 1)
            utr_start = max(utr_start, prev_gene_end + 1)

        else:
            print(
                f"Wrong strand information saved for transcript {transcript_id} for species {species_name}. "
                f"Value of {info['strand']} was given when only + or - are allowed.")
            continue

        # Update the promoters and 3'UTRs dictionaries to add current <transcript_id> info
        promoters_info[transcript_id] = {"chromosome": info["chromosome"],
                                         "strand": info["strand"],
                                         "start": promoter_start,
                                         "end": promoter_end,
                                         "parent_gene_id": gene_id}

        UTRs_info[transcript_id] = {"chromosome": info["chromosome"],
                                    "strand": info["strand"],
                                    "start": utr_start,
                                    "end": utr_end,
                                    "parent_gene_id": gene_id}

    # Save the promoters_info for current species as a json file
    file_name = f"./{GENERAL_PROMOTERS_INFO_DIR}/{species_name}_promoters_general_info.json"
    with open(file_name, "w") as file:
        json.dump(promoters_info, file, indent=4)

    # Save the UTRs_info for current species as a json file
    file_name = f"./{GENERAL_UTR_INFO_DIR}/{species_name}_3UTRs_general_info.json"
    with open(file_name, "w") as file:
        json.dump(UTRs_info, file, indent=4)

    print(f"Finished determining the promoter and 3'UTR coordinates for {i} {species_name}.")
    i = i + 1

# Now that the info for every transcript's promoter and 3'UTR has been saved, go through every transcript's promoter
# and 3'UTR info and determine the nucleotide sequence of each for every species
PROMOTER_SEQUENCES_DIR = "promoter_sequences_per_species"
UTR_SEQUENCES_DIR = "3UTR_sequences_per_species"
os.makedirs(PROMOTER_SEQUENCES_DIR, exist_ok=True)
os.makedirs(UTR_SEQUENCES_DIR, exist_ok=True)

GENOME_DIR = "genomic_sequences"

for i in range(len(species)):

    species_name = species[i]
    project_id = bio_project_ids[i]

    # Get the correct format for species name (e.g.: caenorhabditis_elegans instead of Caenorhabditis Elegans)
    species_name = format_species_name(species_name)

    # Get the promoters general info file for current species
    path = f"./{GENERAL_PROMOTERS_INFO_DIR}/{species_name}_promoters_general_info.json"
    with open(path, "r") as file:
        promoters_info = json.load(file)
    # Get the 3'UTRs general info file for current species
    path = f"./{GENERAL_PROMOTERS_INFO_DIR}/{species_name}_3UTRs_general_info.json"
    with open(path, "r") as file:
        UTRs_info = json.load(file)
    # Load the genomic sequences per chromosome for current species
    path = f"./{GENOME_DIR}/{species_name}.{project_id}.{WBPS_RELEASE_NUMBER}.genomic.fa"
    genome = SeqIO.to_dict(SeqIO.parse(path, "fasta"))

    promoter_sequences = {}
    UTR_sequences = {}

    # Go through every transcript and extract the promoter and 3'UTR sequences
    for transcript_id, info in promoters_info.items():
        chrom = info["chromosome"]
        strand = info["strand"]
        promoter_start = info["start"]
        promoter_end = info["end"]
        UTR_start = UTRs_info[transcript_id]["start"]
        UTR_end = UTRs_info[transcript_id]["end"]

        # Access chromosome sequence
        chrom_seq = genome[chrom].seq
        max_chrom_len = len(chrom_seq)
        # Update end in case it is greater than length of chromosome
        promoter_end = min(promoter_end, max_chrom_len)
        UTR_end = min(UTR_end, max_chrom_len)

        # Access sequence
        promoter_seq = str(chrom_seq[promoter_start - 1:promoter_end]).upper()
        UTR_seq = str(chrom_seq[UTR_start - 1:UTR_end]).upper()

        if strand == "-":
            promoter_seq = reverse_compliment(promoter_seq)
            UTR_seq = reverse_compliment(UTR_seq)

        # Save the info of the current transcript
        promoter_sequences[transcript_id] = {"parent_gene_id": info["parent_gene_id"],
                                             "promoter_sequence": promoter_seq}
        UTR_sequences[transcript_id] = {"parent_gene_id": info["parent_gene_id"],
                                        "UTR_sequence": UTR_seq}

    # Save the promoters and 3'UTR sequences for current species as a json file
    file_name = f"./{PROMOTER_SEQUENCES_DIR}/{species_name}_promoter_sequences.json"
    with open(file_name, "w") as file:
        json.dump(promoter_sequences, file, indent=4)
    file_name = f"./{UTR_SEQUENCES_DIR}/{species_name}_UTR_sequences.json"
    with open(file_name, "w") as file:
        json.dump(UTR_sequences, file, indent=4)

    print(f"Finished determining the promoter and 3'UTR sequences for {i + 1} {species_name}.")

# Final step is to generate the files that will be fed into the convolutional neural network, where each file
# corresponds to a <SOURCE_SPECIES> gene's promoter and 3'UTR sequences of its orthologs

# Determining the ortholog genes to keep (keep only the ones with one-to-one relationship and for those species that
# have at least 5000 one-to-one ortholog genes with <SOURCE_SPECIES>

stats = pd.read_excel("./Orthologues/OrthologuesStats_one-to-one.xlsx")  # Starts with 65 species
stats = stats.set_index('Species')
c_elegans_stats = stats.loc['caenorhabditis_elegans.PRJNA13758.WBPS19_filtered.protein', :]
c_elegans_stats = c_elegans_stats[c_elegans_stats >= 5000]  # 44 species are left

species_to_keep = c_elegans_stats.index.tolist()

ortho_genes = pd.read_csv("./Orthologues/caenorhabditis_elegans.PRJNA13758.WBPS19_filtered.protein.tsv", sep="\t")
ortho_genes = ortho_genes[ortho_genes['Species'].isin(species_to_keep)]  # Only keep the species of interest
# Keep the genes that have a one-to-one ortholog only
ortho_genes = ortho_genes[~ortho_genes['caenorhabditis_elegans.PRJNA13758.WBPS19_filtered.protein'].str.contains(',') &
                          ~ortho_genes['Orthologs'].str.contains(',')]

ORTHOLOG_PROMOTERS_DIR = "ortholog_promoters_per_gene"
ORTHOLOG_UTR_DIR = "ortholog_3UTRs_per_gene"

os.makedirs(ORTHOLOG_PROMOTERS_DIR, exist_ok=True)
os.makedirs(ORTHOLOG_UTR_DIR, exist_ok=True)

i = 1
for source_transcript, source_transcript_orthologs in ortho_genes.groupby(f"{SOURCE_SPECIES}.PRJNA13758.WBPS19_filtered.protein"):
    source_transcript_ortholog_promoters = {}
    source_transcript_ortholog_UTRs = {}
    for _, row in source_transcript_orthologs.iterrows():
        ortholog_species = row["Species"].split(".")[0]
        ortholog_species_transcript_id = row["Orthologs"]
        if ortholog_species == "parelaphostrongylus_tenuis":
            # Example: It is actually rna-gnl|WGS:JAHQIW|KIN20_014332-T1_mrna but in OrthoFinder Excel file, it is
            # rna-gnl|WGS_JAHQIW|KIN20_014332-T1_mrna
            splits = ortholog_species_transcript_id.split("|")
            second_split = splits[1][:3] + ":" + splits[1][4:]
            ortholog_species_transcript_id = "|".join([splits[0], second_split, splits[2]])

        # Open the promoter sequences file for <ortholog_species>
        path = f"./{PROMOTER_SEQUENCES_DIR}/{ortholog_species}_promoter_sequences.json"
        with open(path, "r") as file:
            ortholog_species_promoter_sequences = json.load(file)
        # Open the 3'UTR sequences file for <ortholog_species>
        path = f"./{UTR_SEQUENCES_DIR}/{ortholog_species}_UTR_sequences.json"
        with open(path, "r") as file:
            ortholog_species_UTR_sequences = json.load(file)

        # Get the sequence of the promoter and 3'UTR for that <ortholog_species_transcript_id>
        if ortholog_species_transcript_id in ortholog_species_promoter_sequences:
            promoter_seq = ortholog_species_promoter_sequences[ortholog_species_transcript_id]["promoter_sequence"]
            UTR_seq = ortholog_species_UTR_sequences[ortholog_species_transcript_id]["UTR_sequence"]
        # Specifically for the transcripts of the 5 species: caenorhabditis_brenneri, caenorhabditis_briggsae,
        # caenorhabditis_japonica, caenorhabditis_remanei, and pristionchus_pacificus
        elif f"{ortholog_species_transcript_id}.1" in ortholog_species_promoter_sequences:
            promoter_seq = ortholog_species_promoter_sequences[f"{ortholog_species_transcript_id}.1"][
                "promoter_sequence"]
            UTR_seq = ortholog_species_UTR_sequences[f"{ortholog_species_transcript_id}.1"][
                "UTR_sequence"]
        # Specifically for some transcripts of species ancylostoma_ceylanicum
        elif f"transcript:{ortholog_species_transcript_id[11:]}" in ortholog_species_promoter_sequences:
            promoter_seq = ortholog_species_promoter_sequences[f"transcript:{ortholog_species_transcript_id[11:]}"][
                "promoter_sequence"]
            UTR_seq = ortholog_species_UTR_sequences[f"transcript:{ortholog_species_transcript_id[11:]}"][
                "UTR_sequence"]
        else:
            print(
                f"Failed to fetch promoter and 3'UTR sequences for ortholog species {ortholog_species} "
                f"and transcript_id {ortholog_species_transcript_id}.")
            continue
        source_transcript_ortholog_promoters[ortholog_species] = promoter_seq
        source_transcript_ortholog_UTRs[ortholog_species] = UTR_seq

    # Save the ortholog promoter and 3'UTR sequences for current <SOURCE_SPECIES> gene as a json file
    file_name = f"./{ORTHOLOG_PROMOTERS_DIR}/{source_transcript}_orthologs_promoters.json"
    with open(file_name, "w") as file:
        json.dump(source_transcript_ortholog_promoters, file, indent=4)
    file_name = f"./{ORTHOLOG_UTR_DIR}/{source_transcript}_orthologs_UTRs.json"
    with open(file_name, "w") as file:
        json.dump(source_transcript_ortholog_UTRs, file, indent=4)
    if i % 100 == 0:
        print(f"Ortholog promoters and 3'UTRs for {i} genes have bene processed.")
    i = i + 1

# Adding the promoter and 3'UTR sequences for {SOURCE_SPECIES}
path = f"./{PROMOTER_SEQUENCES_DIR}/{SOURCE_SPECIES}_promoter_sequences.json"
with open(path, "r") as file:
    c_elegans_promoter_sequences = json.load(file)
path = f"./{UTR_SEQUENCES_DIR}/{SOURCE_SPECIES}_UTR_sequences.json"
with open(path, "r") as file:
    c_elegans_UTR_sequences = json.load(file)

i = 1
for source_transcript, _ in ortho_genes.groupby(f"{SOURCE_SPECIES}.PRJNA13758.WBPS19_filtered.protein"):
    # Load the file that already has promoter and 3'UTR sequences for the orthologs
    promoter_file_name = f"{ORTHOLOG_PROMOTERS_DIR}/{source_transcript}_orthologs_promoters.json"
    with open(promoter_file_name, "r") as file:
        transcript_promoters = json.load(file)
    UTR_file_name = f"{ORTHOLOG_UTR_DIR}/{source_transcript}_orthologs_UTRs.json"
    with open(UTR_file_name, "r") as file:
        transcript_UTRs = json.load(file)

    # Add the promoter and 3'UTR sequences for <SOURCE_SPECIES>
    if source_transcript in c_elegans_promoter_sequences:
        transcript_promoters[SOURCE_SPECIES] = c_elegans_promoter_sequences[source_transcript]["promoter_sequence"]
        transcript_UTRs[SOURCE_SPECIES] = c_elegans_UTR_sequences[source_transcript]["UTR_sequence"]
    elif f"{source_transcript}.1" in c_elegans_promoter_sequences:
        transcript_promoters[SOURCE_SPECIES] = c_elegans_promoter_sequences[f"{source_transcript}.1"]["promoter_sequence"]
        transcript_UTRs[SOURCE_SPECIES] = c_elegans_UTR_sequences[f"{source_transcript}.1"][
            "UTR_sequence"]
    else:
        print(f"Wasn't able to fetch transcript_id {source_transcript} for {SOURCE_SPECIES}.")

    # Save back the file with the promoter and 3'UTR sequences of <SOURCE_SPECIES> added
    with open(promoter_file_name, "w") as file:
        json.dump(transcript_promoters, file, indent=4)
    with open(UTR_file_name, "w") as file:
        json.dump(transcript_UTRs, file, indent=4)
    if i % 100 == 0:
        print(f"Promoter and 3'UTR sequences for {SOURCE_SPECIES} has been added for {i} genes.")
    i = i + 1

# Done creating the files to load into the motif-based encoder!
