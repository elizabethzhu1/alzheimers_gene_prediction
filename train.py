import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import argparse
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def main(dataset):
    # Prepare the data
    X = pd.read_csv(dataset)
    y_labels = pd.read_csv('data/processed_y.csv')

    # Split dataset into train, test, validation
    x_train = X.iloc[:48173]
    x_valid = X.iloc[48173:56316]
    x_test = X.iloc[56316:]

    print(x_train.shape)
    print(x_valid.shape)

    y_train = y_labels.iloc[:48173]
    y_valid = y_labels.iloc[48173:56316]
    y_test = y_labels.iloc[56316:]

    print(y_train.shape)
    print(y_valid.shape)

    num_features = x_train.shape[1]

    # Get the training tensors and also shuffle them
    X_tensor = torch.tensor(x_train.values, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).squeeze(1)

    # Validation Tensors
    X_valid_tensor = torch.tensor(x_valid.values, dtype=torch.float32).unsqueeze(1)
    y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.float32).squeeze(1)

    # Testing Tensors
    X_test_tensor = torch.tensor(x_test.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).squeeze(1)

    print("Number of NaNs in X_tensor:", torch.isnan(X_tensor).sum().item())


    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    # define the CNN
    class CNN(nn.Module):
        def __init__(self, num_features):
            super().__init__()

            self.num_features = num_features
            
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

            self.pool = nn.MaxPool1d(kernel_size=3, stride=3)  # downsamples by 3x

            self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

            # POOLING (k=3, s=3) wih PCA
            self.fc1 = nn.Linear(32, 64)

            # POOLING without PCA
            # self.fc1 = nn.Linear(32 * 9, 64)

            self.fc2 = nn.Linear(64, 1)  # 2 classes

        def forward(self, x):

            x = F.relu(self.conv1(x))  # conv1 -> ReLU -> Pool
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)

            x = torch.flatten(x, 1) # flatten all dimensions except batch size

            x = F.relu(self.fc1(x))

            x = self.fc2(x).squeeze(1)

            return x

    model = CNN(num_features)

    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    y_test = y_test.to_numpy()
    y_train = y_train.to_numpy()
    y_valid = y_valid.to_numpy()

    training_accuracies = []

    # Training loop
    epochs = 10
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

        train_accuracy = correct / len(y_train)

        print("Train Accuracy is :", train_accuracy, " after epoch ", epoch)

        training_accuracies.append(train_accuracy)
    

    # Plotting Training Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs + 1), training_accuracies, marker='o', linestyle='-', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy Over Epochs")
    plt.grid(True)
    plt.show()

    # Evaluate the model on validation dataset to get the best hyperparameters
    model.eval()
    with torch.no_grad():
        predictions = model(X_valid_tensor).squeeze()
        predictions = (predictions > 0.5).float()  # Convert to binary predictions

    correct = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == y_valid[i]:
            correct += 1

    print("Validation Accuracy is:", correct / len(y_valid), "after epoch", epoch)
    print("Size of Validation Dataset:", len(y_valid))

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

    # generate confusion matrix for test dataset
    cm = confusion_matrix(y_test, predictions)
    print("CONFUSION MATRIX", cm)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset')

    args = parser.parse_args()

    main(args.dataset)
