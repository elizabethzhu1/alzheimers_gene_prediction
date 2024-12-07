# Predicting Alzheimer's From snRNA Sequences: A Convolutional Neural Network Approach

We train a Convolutional Neural Network to predict Alzheimer's Disease given snRNA Sequences, sampled from the nuclei of human brain cells, and additional biomarkers such as age and cell type. We first perform k-mer sequencing to extract local gene patterns, extending our feature space to include 2-sequence and 3-sequence nucleotide patterns. We then use Principal Component Analysis (PCA) to reduce the dimensionality of our feature space. After training our neural network on both Alzheimer's diagnoses and control data, we obtain a test accuracy of 75.98 percent without PCA, and 75.23 percent having run PCA on the k-mer sequences. 

NCBI Gene Expression Omnibus Dataset (GSE174367): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174367
