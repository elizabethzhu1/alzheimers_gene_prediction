from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


def one_hot_encode(nucleotide):
    if nucleotide == 'A':
       return [1, 0, 0, 0]
    elif nucleotide == 'T':
       return [0, 1, 0, 0]
    elif nucleotide == 'C':
       return [0, 0, 1, 0]
    elif nucleotide == 'G':
       return [0, 0, 0, 1]


def encoding(sequence):
    encoded_seq = []
    for nucleotide in sequence:
        # so that every four numbers represent one nucleotide
        encoded_seq.append(tuple(one_hot_encode(nucleotide)))
    return encoded_seq


# k-mer sequences
def k_mer_sequences(encoded_seq, k):
    kmer_sequences = []
    for i in range(len(encoded_seq)):
      if i+k < len(encoded_seq):
        kmer_sequences.append(encoded_seq[i:i+k])
      else:
        # print(kmer_sequences)
        return kmer_sequences
      

# given a sequence, maps all found k-mer sequences to frequency count
def create_kmer_count(kmer_counts, kmer_sequence):
  for sequence in kmer_sequence:
    if tuple(sequence) not in kmer_counts:
      kmer_counts[tuple(sequence)] = 0
    kmer_counts[tuple(sequence)] += 1

  return kmer_counts


# generates all unique kmer sequences given a dataset
def generate_unique_kmers(df, k):
    all_kmer_sequences = []
    for i, row in df.iterrows():
      barcode = row['Barcode']

      # remove the suffix
      parts = barcode.split("-")
      barcode = parts[0]

      all_kmer_sequences.extend(k_mer_sequences(barcode, k))

    unique_kmers = list(set(all_kmer_sequences))

    return unique_kmers


# pass in as input: all_cell_types = df['Cell.Type']
def generate_cell_type_mapping(all_cell_types):
   # Examine the different cell types for our data
    cell_type_mapping = {}

    for cell in all_cell_types:
        if cell not in cell_type_mapping:
            cell_type_mapping[cell] = 0
        cell_type_mapping[cell] += 1

    return cell_type_mapping


def create_cell_type_age_feature(df):
    # Convert the categorical variables into indicator variables with new columns
    cell_df = pd.get_dummies(df, columns=['Cell.Type'], drop_first=True)
    scaler = StandardScaler()
    cell_df['Age'] = scaler.fit_transform(cell_df[['Age']])

    # Identify the "Age" column and "Cell" column and create interaction features between them
    for column in cell_df.columns:
        # Identify whether current column is a cell type
        if 'Cell' in column:
            cell_df[column] = cell_df[column].astype(int)
            cell_df[f'{column}_Age'] = cell_df[column] * cell_df['Age']

    interaction_columns = [col for col in cell_df.columns if 'Age' in col and 'Cell.Type_' in col]
    interaction_df = cell_df[interaction_columns]

    return interaction_df

def run_pca(df):
    # Make sure to scale data 
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Initialize PCA 
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)

    # Create PCA Data frame 
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=df.index)
    return pca_df
