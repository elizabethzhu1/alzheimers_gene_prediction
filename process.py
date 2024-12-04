# Read the barcode column from our dataset
import pandas as pd
from helper_functions import encoding, k_mer_sequences, create_kmer_count, run_pca, visualize_pca, generate_cell_type_mapping, create_cell_type_age_feature
from sklearn.preprocessing import StandardScaler

# Read in our data
df = pd.read_csv('data/alzheimers_RNA_data.csv', header=None)
df = df.drop(index=0).reset_index(drop=True)

# Shuffle the DataFrame rows
df = df.sample(frac=1).reset_index(drop=True)

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
df = pd.concat([AD_filtered_sample, control_filtered_sample], ignore_index=True)

# Remove "-1" in barcodes to preserve our features
barcode_column = df['Barcode']
barcodes = []
for barcode in barcode_column:
    parts = barcode.split("-")
    barcodes.append(parts[0])
# print("Barcodes are: ", barcodes)

# # convert AD/Control diagnosis into binary values
# df['Diagnosis'] = df['Diagnosis'].map({'AD': 1, 'Control': 0})  # Binary encoding

# Process Diagnosis Data
y_labels = []

for diag in df["Diagnosis"]:
  if diag == "AD":
    y_labels.append(1)
  else:
    y_labels.append(0)
  
y_labels = pd.DataFrame(y_labels)

# remove labels from df
df = df.drop(columns='Diagnosis')

# Process every single DNA encoding
# all_one_hot_encodings = []
# for barcode in barcodes:
#   one_hot_encoding = encoding(barcode)
#   all_one_hot_encodings.append(one_hot_encoding)

# all_kmer_sequences stores the kmer_sequence of each barcode
# all_kmer_counts stores the count of each kmer_sequence
all_kmer_sequences = []
all_kmer_counts = []
for barcode in barcodes:
  # the kmer_sequence for each barcode
  kmer_sequences = k_mer_sequences(barcode, 3)
  all_kmer_sequences.extend(kmer_sequences)
  all_kmer_counts.append(create_kmer_count({}, kmer_sequences))

unique_kmers = list(set(all_kmer_sequences))
print('unique_kmers length', len(unique_kmers))

df_kmers = pd.DataFrame(columns=unique_kmers)

df = pd.concat([df, df_kmers], axis=1)

for i, row in df.iterrows():
  for kmer in unique_kmers:
    if kmer in barcode:
      df.at[i, kmer] = 1
    else:
      df.at[i, kmer] = 0

# print(df)
# print(df.shape)

# add feature of cell type + age interaction
all_cell_types = df['Cell.Type']
cell_type_mapping = generate_cell_type_mapping(all_cell_types)

print("Cell Type Mapping:", cell_type_mapping)

interaction_columns = create_cell_type_age_feature(df)

df = pd.concat([df, interaction_columns], axis=1)

print(df)

# numerical_features = ['Age', 'PMI', 'RIN', 'Cell.Type_EX_Age', 'Cell.Type_INH_Age', 'Cell.Type_MG_Age', 'Cell.Type_ODC_Age', 'Cell.Type_OPC_Age', 'Cell.Type_PER.END_Age']

# Remove non-numerical columns
# df = df.select_dtypes(include=['int64', 'float64'])

# print("df num", df)

# Normalize numerical features
# scaler = StandardScaler()
# df = scaler.fit_transform(df)

# run PCA on one hot encodings
# pca_df = run_pca(all_one_hot_encodings, 5)  # adjust number of components

# run PCA on kmer sequences
# pca_kmer_df = run_pca(all_one_hot_encodings, 5)  # adjust number of components

# visualize PCA
# visualize_pca(pca_kmer_df)

# print("DF", df)

# Print the result
# print(numerical_df)

numerical_features = [
    'Age', 'RIN', 'Cell.Type_EX_Age', 'Cell.Type_INH_Age',
    'Cell.Type_MG_Age', 'Cell.Type_ODC_Age', 'Cell.Type_OPC_Age',
    'Cell.Type_PER.END_Age'
]

numerical_features.extend(unique_kmers)

print(numerical_features)

df = df[numerical_features]

df.to_csv('data/processed_X.csv', index=False)
y_labels.to_csv('data/processed_y.csv', index=False)


# add main() function and pass in input data as argument
