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
    cell_df = pd.get_dummies(df, columns=['Cell.Type'], drop_first=True)
    scaler = StandardScaler()
    cell_df['Age'] = scaler.fit_transform(cell_df[['Age']])

    for column in cell_df.columns:
        print(column)
        if 'Cell' in column:  # Identify one-hot encoded cell type columns
            cell_df[column] = cell_df[column].astype(int)
            cell_df[f'{column}_Age'] = cell_df[column] * cell_df['Age']

    interaction_columns = [col for col in cell_df.columns if 'Age' in col and 'Cell.Type_' in col]
    interaction_df = cell_df[interaction_columns]

    return interaction_df


def run_pca(all_one_hot_encodings, n_components):
   # PCA to transform the one-hot encodings of barcodes
    scaler = StandardScaler()

    # First flatten all_one_hot_encodings and then turn it into a dataframe
    flattened_one_hot_encodings = [ [value for tupl in sequence for value in tupl] for sequence in all_one_hot_encodings ]

    df = pd.DataFrame(flattened_one_hot_encodings)
    X_normalized = scaler.fit_transform(df)

    # Define number of principal components to retain
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_normalized)
    pca_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

    print(pca_df.head())
    print("Explained variance by each component:", pca.explained_variance_ratio_)

    return pca_df


def run_pca_kmer(all_kmer_sequences, n_components):
    scaler = StandardScaler()
    # First flatten all_kmer_sequences and then turn it into a dataframe
    flattened_kmer_encodings = [element for sequence in all_kmer_sequences for tup in sequence for element in tup]
    df = pd.DataFrame(flattened_kmer_encodings)
    X_normalized = scaler.fit_transform(df)

    # Define number of principal components to retain
    n_components = 3
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_normalized)
    pca_kmer_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2', 'PC3'])
    print(pca_kmer_df.head())
    print("Explained variance by each component:", pca.explained_variance_ratio_)


def visualize_pca(pca_df):
   # Visualization for PCA on the first two principal components of the kmer sequences
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Kmer sequences for DNA')
    plt.grid(True)
    plt.show()
