import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('data/kmer_sequences.csv', index_col=0)  # Adjust index_col if necessary

# Optional: Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Initialize PCA, you can modify the number of components
pca = PCA(n_components=2)  # Adjust components based on your needs
principal_components = pca.fit_transform(df_scaled)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components,
                      columns=['Principal Component 1', 'Principal Component 2'],
                      index=df.index)

pca_df.to_csv('pca_output.csv')