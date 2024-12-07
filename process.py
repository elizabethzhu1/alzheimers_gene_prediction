import pandas as pd
from helper_functions import encoding, k_mer_sequences, create_kmer_count, run_pca, visualize_pca, generate_cell_type_mapping, create_cell_type_age_feature, generate_unique_kmers, run_pca, generate_gc_ratios
from sklearn.preprocessing import StandardScaler

# Read in our data
df = pd.read_csv('data/alzheimers_RNA_data.csv', header=None)
df = df.drop(index=0).reset_index(drop=True)

# Rename columns
df.columns = ['Barcode', 'SampleID', 'Diagnosis', 'Batch', 'Cell.Type', 'Cluster', 'Age', 'Sex', 'PMI', 'Tangle.Stage', 'Plaque.Stage', 'RIN']

unique_samples = []

df['SampleID'] = df['SampleID'].str.removeprefix('Sample-')

# Count the frequency of each value in column 'A'
sample_counts = df['SampleID'].value_counts()

# Map the frequencies back to the DataFrame
df['frequency'] = df['SampleID'].map(sample_counts)

# Sort the DataFrame by frequency (descending by default)
df_sorted = df.sort_values(by='frequency', ascending=False).reset_index(drop=True)

df_sorted.to_csv('data/alzheimers_RNA_data_sorted.csv', index=False)

# Retrieve diagnosis column
diagnosis = df['Diagnosis'].to_list()

# Encode diagnosis data into y_labels column
y_labels = []
for diag in df["Diagnosis"]:
  if diag == "AD":
    y_labels.append(1)
  else:
    y_labels.append(0)

y_labels = pd.DataFrame(y_labels)

# Remove y_labels from df
df = df.drop(columns='Diagnosis')

# Feature Engineering: add cell type + age interaction feature
all_cell_types = df['Cell.Type']
cell_type_mapping = generate_cell_type_mapping(all_cell_types)

interaction_columns = create_cell_type_age_feature(df)

df = pd.concat([df, interaction_columns], axis=1)

# Feature Engineering: add GC content as a feature
gc_ratios = generate_gc_ratios(df)

df['GC_Content'] = pd.Series(gc_ratios, index=df.index)

# Generate kmers
two_mers = generate_unique_kmers(df, 2)
three_mers = generate_unique_kmers(df, 3)

print('two_mers length', len(two_mers))
print('three_mers length', len(three_mers))

unique_kmers = two_mers + three_mers
# unique_kmers = two_mers
 
df_kmers = pd.DataFrame(columns=unique_kmers)

# Populate kmer columns (if a given kmer appears in the sequence or not)
for kmer in unique_kmers:    
    # Set column to count of kmers in sequence
    df_kmers[kmer] = df.index.map(lambda x: df['Barcode'][x].count(kmer) if kmer in df['Barcode'][x] else 0)

# Only include numerical_features
numerical_features = [
    'Age', 'RIN', 'Cell.Type_EX_Age', 'Cell.Type_INH_Age',
    'Cell.Type_MG_Age', 'Cell.Type_ODC_Age', 'Cell.Type_OPC_Age',
    'Cell.Type_PER.END_Age'
]

print("features", numerical_features)
print("num features", len(numerical_features))

# Add unique kmer sequences as columns to df
pca_kmers_df = run_pca(df_kmers)

# visualize the variance of PCA
# visualize_pca(pca_kmers_df)

# Add pca-reduced kmer sequences as columns to df
df_pca = pd.concat([df[numerical_features], pca_kmers_df], axis=1)
df = pd.concat([df[numerical_features], df_kmers], axis=1)

print("df.columns", df.columns)

df.to_csv('data/processed_X.csv', index=False)
df_pca.to_csv('data/processed_X_pca.csv', index=False)
y_labels.to_csv('data/processed_y.csv', index=False)

# add main() function and pass in input data as argument

print("df.shape", df.shape)
