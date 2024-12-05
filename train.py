import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse
# from torch.optim.lr_scheduler import StepLR


def main(dataset):

    # Prepare the data
    X = pd.read_csv(dataset)
    y_labels = pd.read_csv('data/processed_y.csv')

    # Split Features and Predictions into train, test, validation
    x_train = X.loc[0:61262]
    x_test = X.loc[61262:61421]
    x_valid = X.loc[61421:61473]
    y_train = y_labels.loc[0:61262]
    y_test = y_labels.loc[61262:61421]
    y_valid = y_labels.loc[61421:61473]

    num_features = x_train.shape[1]

    # Training Set Tensors
    X_tensor = torch.tensor(x_train.values, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).squeeze(1)

    # Validation Set Tensors
    X_valid_tensor = torch.tensor(x_valid.values, dtype=torch.float32).unsqueeze(1)
    y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.float32).squeeze(1)

    # Testing Set Tensors
    X_test_tensor = torch.tensor(x_test.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).squeeze(1)

    # Make sure to shuffle dataset
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

            # pooling reduces dimensions
            # self.fc1 = nn.Linear(32 * num_features - 64, 64)
            self.fc1 = nn.Linear(32 * num_features, 64)

            self.fc2 = nn.Linear(64, 1)  # 2 classes

        def forward(self, x):

            x = F.relu(self.conv1(x))  # conv1 -> ReLU -> Pool
            # x = self.pool(x)
            x = F.relu(self.conv2(x))
            # x = self.pool(x)

            x = torch.flatten(x, 1) # flatten all dimensions except batch size

            x = F.relu(self.fc1(x))

            x = self.fc2(x).squeeze(1)

            return x

    model = CNN(num_features)

    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = StepLR(optimizer, step_size=3, gamma=0.1)  # Reduce LR by 0.1 every 5 epochs

    y_test = y_test.to_numpy()
    y_train = y_train.to_numpy()
    y_valid = y_valid.to_numpy()

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
        
        # scheduler.step()
        
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

    # TODO: Evaluate model on test dataset
    # # Evaluate the model on test dataset
    # model.eval()
    # with torch.no_grad():
    #     predictions = model(X_test_tensor).squeeze()
    #     predictions = (predictions > 0.5).float()  # Convert to binary predictions
    #     print("Predictions:", predictions.numpy())

    # correct = 0
    # for i in range(len(predictions)):
    #     if int(predictions[i]) == y_test[i]:
    #         correct += 1

    # print("Test Accuracy is :", correct / len(y_test), " after epoch ", epoch)

    # Evaluate the model on validation dataset to get the best hyperparameters
    model.eval()
    with torch.no_grad():
        predictions = model(X_valid_tensor).squeeze()
        predictions = (predictions > 0.5).float()  # Convert to binary predictions

    correct = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == y_valid[i]:
            correct += 1

    print("Validation Accuracy is :", correct / len(y_valid), " after epoch ", epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    args = parser.parse_args()
    main(args.dataset)
