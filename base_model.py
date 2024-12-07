from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from helper_functions import encoding
import numpy as np
import pandas as pd

df = pd.read_csv('data/alzheimers_RNA_data.csv', header=None)
df = df.drop(index=0).reset_index(drop=True)

# Rename columns
df.columns = ['Barcode', 'SampleID', 'Diagnosis', 'Batch', 'Cell.Type', 'Cluster', 'Age', 'Sex', 'PMI', 'Tangle.Stage', 'Plaque.Stage', 'RIN']

# Retrieve diagnosis column
diagnosis = df['Diagnosis'].to_list()

# Process Diagnosis Data
binary_diagnosis = []

for diag in diagnosis:
  if diag == "AD":
    binary_diagnosis.append(1)
  else:
    binary_diagnosis.append(0)

# Retrieve barcode column
barcode_column = df['Barcode'].to_list()

barcodes = []
for barcode in barcode_column:
    parts = barcode.split("-")
    barcodes.append(parts[0])

all_one_hot_encodings = []
for barcode in barcodes:
  one_hot_encoding = encoding(barcode)
  all_one_hot_encodings.append(one_hot_encoding)

flattened_encodings = [np.array(seq).flatten() for seq in all_one_hot_encodings]

X = np.array(flattened_encodings)

 # Prepare the data
y_labels = pd.read_csv('data/processed_y.csv')

# Split dataset into train, test, validation
x_train = X[:48173]
x_valid = X[48173:56316]
x_test = X[56316:]

print(x_train.shape)
print(x_valid.shape)

y_train = y_labels.iloc[:48173]
y_valid = y_labels.iloc[48173:56316]
y_test = y_labels.iloc[56316:]

# Two baseline models
model = LogisticRegression()
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)

accuracy_train = accuracy_score(y_train, y_pred_train)
precision_train = precision_score(y_train, y_pred_train)

accuracy_test = accuracy_score(y_test, y_pred)
precision_test = precision_score(y_test, y_pred)

print(f"Test Accuracy: ", accuracy_test)
print(f"Test Precision: ", precision_test)

print(f"Train Accuracy: ", accuracy_train)
print(f"Train Precision: ", precision_train)