# Read the barcode column from our dataset
import pandas as pd
from helper_functions import encoding, k_mer_sequences, create_kmer_count, run_pca, visualize_pca, generate_cell_type_mapping, create_cell_type_age_feature


# Read in our data
df = pd.read_csv('data/alzheimers_RNA_data.csv', header=None)
df = df.drop(index=0).reset_index(drop=True)

# before preprocessing
original_df = df.copy()

# df = df.sample(n=100, random_state=42)

# Rename our columns
df.columns = ['Barcode', 'SampleID', 'Diagnosis', 'Batch', 'Cell.Type', 'Cluster', 'Age', 'Sex', 'PMI', 'Tangle.Stage', 'Plaque.Stage', 'RIN']

# Retrieve diagnosis
diagnosis_raw = df['Diagnosis']

diagnosis = []
for d in diagnosis_raw:
    diagnosis.append(d)

# Filter our dataset
AD = 0
control = 0
for d in diagnosis:
  if d == 'AD':
    AD += 1
  else:
    control += 1

# Filtering Dataset
AD_filtered_sample = df[df['Diagnosis'] == 'AD'].sample(20000)
control_filtered_sample = df[df['Diagnosis'] == 'Control'].sample(20000)
samples = pd.concat([AD_filtered_sample, control_filtered_sample], ignore_index=True)

# Remove "-1" in barcodes to preserve our features
barcode_column = samples['Barcode']
barcodes = []
for barcode in barcode_column:
    parts = barcode.split("-")
    barcodes.append(parts[0])
print("Barcodes are: ", barcodes)

# Process Diagnosis Data
# binary_diagnosis = []

# for diag in samples["Diagnosis"]:
#   if diag == "AD":
#     binary_diagnosis.append(1)
#   else:
#     binary_diagnosis.append(0)

# Process every single DNA encoding
all_one_hot_encodings = []
for barcode in barcodes:
  one_hot_encoding = encoding(barcode)
  all_one_hot_encodings.append(one_hot_encoding)

# all_kmer_sequences stores the kmer_sequence of each barcode
# all_kmer_counts stores the count of each kmer_sequence
all_kmer_sequences = []
all_kmer_counts = []
for one_hot_encoding in all_one_hot_encodings:
  # the kmer_sequence for each barcode
  kmer_sequence = k_mer_sequences(one_hot_encoding, 3)
  all_kmer_sequences.append(kmer_sequence)
  all_kmer_counts.append(create_kmer_count({}, kmer_sequence))

# run PCA on one hot encodings
pca_df = run_pca(all_one_hot_encodings, 5)  # adjust number of components

# run PCA on kmer sequences
pca_kmer_df = run_pca(all_one_hot_encodings, 5)  # adjust number of components

# visualize PCA
visualize_pca(pca_kmer_df)


# add feature of cell type + age interaction
all_cell_types = df['Cell.Type']
cell_type_mapping = generate_cell_type_mapping(all_cell_types)

print("Cell Type Mapping:", cell_type_mapping)

interaction_columns = create_cell_type_age_feature(df)

df = pd.concat([df, interaction_columns], axis=1)

print("DF", df)
