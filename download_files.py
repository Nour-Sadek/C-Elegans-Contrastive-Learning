import pandas as pd
import ftplib

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
i = 1
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
                print(f"Downloading {i}: {filename} from {dir_path}")
                with open(filename, "wb") as f:
                    ftp.retrbinary(f"RETR {filename}", f.write)
    except ftplib.error_perm:
        print(f"Directory not found: {dir_path}")
    i = i + 1
