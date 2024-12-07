from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


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


# generate gc percentages for each barcode
def generate_gc_ratios(df):
  gc_ratios = []
  for barcode in df['Barcode']:
    gc_count = 0
    for nucleotide in barcode:
      if nucleotide == 'G' or nucleotide == 'C':
        gc_count += 1
    
    gc_ratio = gc_count / len(barcode)
    gc_ratios.append(gc_ratio)
  
  return gc_ratios


# We want to acknowledge that although we came up with the idea of using interaction features,
# we used ChatGPT in figuring out syntaxes for DataFrame processing to achieve the effects
# of interaction features
def create_cell_type_age_feature(df):
    cell_df = pd.get_dummies(df, columns=['Cell.Type'], drop_first=True)
    scaler = StandardScaler()
    cell_df['Age'] = scaler.fit_transform(cell_df[['Age']])

    for column in cell_df.columns:
        print(column)
        if 'Cell' in column:
            cell_df[column] = cell_df[column].astype(int)
            cell_df[f'{column}_Age'] = cell_df[column] * cell_df['Age']

    interaction_columns = [col for col in cell_df.columns if 'Age' in col and 'Cell.Type_' in col]
    interaction_df = cell_df[interaction_columns]

    return interaction_df


# We want to acknowledge that while we came up with the idea of running PCA, we used ChatGPT 
# in conceptualizing how to write the PCA code and specific syntaxes for achieving the PCA effects.
def run_pca(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=5)
    principal_components = pca.fit_transform(df_scaled)

    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=df.index)

    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_

    print("VARIANCE OF PCA", explained_variance)
    print("VARIANCE RATIO OF PCA", explained_variance_ratio)

    return pca_df
