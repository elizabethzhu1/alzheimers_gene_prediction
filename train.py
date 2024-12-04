import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Prepare the data
X = pd.read_csv('data/processed_X.csv')
y_labels = pd.read_csv('data/processed_y.csv')

x_train, x_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)

# encode categorical features that are relevant 

# X_num = (X_num - X_num.mean(axis=0)) / X_num.std(axis=0)  # Standardize features

print(X.shape[0])
print(y_labels.shape[0])

# reshape for CNN input (PyTorch expects tensors)
X_tensor = torch.tensor(x_train.values, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32).squeeze(1)

X_test_tensor = torch.tensor(x_test.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).squeeze(1)

print("Number of NaNs in X_tensor:", torch.isnan(X_tensor).sum().item())


dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# define the CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: Try Kernel Size being 2 as well 
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1)
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # downsamples by 2x

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(32 * 11, 64)
        
        self.fc2 = nn.Linear(64, 1)  # 2 classes

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = F.relu(self.conv1(x))  # conv1 -> ReLU -> Pool
        # x = self.pool(x)
        x = F.relu(self.conv2(x))
        # x = self.pool(x)

        # print("x (before flatten)", x)
        # print("x.shape[0]", x.shape[0])

        x = torch.flatten(x, 1) # flatten all dimensions except batch size

        # print("x", x)
        # print("x.shape[0]", x.shape[0])

        x = F.relu(self.fc1(x))

        # print("x", x)
        # print("x.shape[0]", x.shape[0])


        x = self.fc2(x).squeeze(1)
        # x = self.sigmoid(self.fc2(x))

        # print("x", x)
        # print("x.shape[0]", x.shape[0])

        return x
    
        # apply dropout?

model = CNN()

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

y_test = y_test.to_numpy()
y_train = y_train.to_numpy()

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluate the model on train dataset
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).float()

    correct = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == y_train[i]:
            correct += 1

    print("Train Accuracy is :", correct / len(y_train), " after epoch ", epoch)

# Evaluate the model on test dataset
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze()
    predictions = (predictions > 0.5).float()  # Convert to binary predictions
    print("Predictions:", predictions.numpy())

correct = 0
for i in range(len(predictions)):
    if int(predictions[i]) == y_test[i]:
        correct += 1

print("Test Accuracy is :", correct / len(y_test), " after epoch ", epoch)