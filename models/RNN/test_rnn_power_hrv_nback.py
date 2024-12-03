# update from test_rnn_power_hrv_nback.py: target changed from pvtb scores to nback scores

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob2
import random2
import pandas as pd
import os
import re
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


# matplotlib.use('Agg')
matplotlib.use('TkAgg')


class EEGDataset(Dataset):
    def __init__(self, X_power, X_hrv_t, X_hrv_f, y):
        """
        Args:
            X_power (numpy array or torch tensor): EEG power features of shape (num_samples, seq_len_power, feature_dim_power)
            X_hrv_t (numpy array or torch tensor): HRV time-domain features of shape (num_samples, seq_len_hrv_t, feature_dim_hrv_t)
            X_hrv_f (numpy array or torch tensor): HRV frequency-domain features of shape (num_samples, seq_len_hrv_f, feature_dim_hrv_f)
            y (numpy array or torch tensor): Labels of shape (num_samples,)
        """
        # Convert input arrays to PyTorch tensors
        self.X_power = torch.tensor(X_power, dtype=torch.float32)
        self.X_hrv_t = torch.tensor(X_hrv_t, dtype=torch.float32)
        self.X_hrv_f = torch.tensor(X_hrv_f, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # Binary classification labels

    def __len__(self):
        return len(self.X_power)  # Assumes all inputs have the same number of samples

    def __getitem__(self, idx):
        return {
            'power': self.X_power[idx],
            'hrv_time': self.X_hrv_t[idx],
            'hrv_freq': self.X_hrv_f[idx],
            'label': self.y[idx]
        }


class MultiPathLSTMClassifier(nn.Module):
    def __init__(self, input_dims, hidden_dim, num_layers, output_dim, dropout=0.45):
        """
        Args:
            input_dims (tuple): A tuple of input dimensions for each feature type (power, hrv_t, hrv_f).
            hidden_dim (int): Number of features in the hidden state for each LSTM.
            num_layers (int): Number of recurrent layers for each LSTM.
            output_dim (int): Number of output features (e.g., 1 for binary classification).
            dropout (float): Dropout probability for LSTM layers.
        """
        super(MultiPathLSTMClassifier, self).__init__()

        # Separate LSTMs for each feature type
        self.lstm_power = nn.LSTM(input_dims[0], hidden_dim, num_layers,
                                  batch_first=True, dropout=dropout, bidirectional=True)
        self.lstm_hrv_t = nn.LSTM(input_dims[1], hidden_dim, num_layers,
                                  batch_first=True, dropout=dropout, bidirectional=True)
        self.lstm_hrv_f = nn.LSTM(input_dims[2], hidden_dim, num_layers,
                                  batch_first=True, dropout=dropout, bidirectional=True)

        # Batch normalization layers for each LSTM's output
        self.bn_power = nn.BatchNorm1d(hidden_dim)
        self.bn_hrv_t = nn.BatchNorm1d(hidden_dim)
        self.bn_hrv_f = nn.BatchNorm1d(hidden_dim)

        # Fully connected layer after concatenation
        self.fc = nn.Linear(hidden_dim * 3, output_dim)

        # Additional dropout layer before the fully connected layer
        self.fc_dropout = nn.Dropout(dropout)

    def forward(self, x_power, x_hrv_t, x_hrv_f):
        # Process each feature type through its respective LSTM
        _, (hn_power, _) = self.lstm_power(x_power)
        _, (hn_hrv_t, _) = self.lstm_hrv_t(x_hrv_t)
        _, (hn_hrv_f, _) = self.lstm_hrv_f(x_hrv_f)

        # Take the last hidden state from each LSTM and apply batch normalization
        hn_power = self.bn_power(hn_power[-1])  # (batch_size, hidden_dim)
        hn_hrv_t = self.bn_hrv_t(hn_hrv_t[-1])  # (batch_size, hidden_dim)
        hn_hrv_f = self.bn_hrv_f(hn_hrv_f[-1])  # (batch_size, hidden_dim)

        # Concatenate the hidden states from all LSTM paths
        combined_features = torch.cat((hn_power, hn_hrv_t, hn_hrv_f), dim=1)  # (batch_size, hidden_dim * 3)

        # Apply dropout before the fully connected layer
        combined_features = self.fc_dropout(combined_features)

        # Pass through the fully connected layer
        out = self.fc(combined_features)

        # Output without sigmoid to be used with BCEWithLogitsLoss for better numerical stability
        return out


#
# class MultiPathLSTMClassifier(nn.Module):
#     def __init__(self, input_dims, hidden_dim, num_layers, output_dim, dropout=0.3):
#         """
#         Args:
#             input_dims (tuple): A tuple of input dimensions for each feature type (power, hrv_t, hrv_f)
#             hidden_dim (int): Number of features in the hidden state for each LSTM.
#             num_layers (int): Number of recurrent layers for each LSTM.
#             output_dim (int): Number of output features (e.g., 1 for binary classification).
#             dropout (float): Dropout probability for LSTM layers.
#         """
#         super(MultiPathLSTMClassifier, self).__init__()
#
#         # Separate LSTMs for each feature type
#         self.lstm_power = nn.LSTM(input_dims[0], hidden_dim, num_layers, batch_first=True, dropout=dropout)
#         self.lstm_hrv_t = nn.LSTM(input_dims[1], hidden_dim, num_layers, batch_first=True, dropout=dropout)
#         self.lstm_hrv_f = nn.LSTM(input_dims[2], hidden_dim, num_layers, batch_first=True, dropout=dropout)
#
#         # Fully connected layer after concatenation
#         self.fc = nn.Linear(hidden_dim * 3, output_dim)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x_power, x_hrv_t, x_hrv_f):
#         # Process each feature type through its respective LSTM
#         _, (hn_power, _) = self.lstm_power(x_power)
#         _, (hn_hrv_t, _) = self.lstm_hrv_t(x_hrv_t)
#         _, (hn_hrv_f, _) = self.lstm_hrv_f(x_hrv_f)
#
#         # Take the last hidden state from each LSTM
#         hn_power = hn_power[-1]  # (batch_size, hidden_dim)
#         hn_hrv_t = hn_hrv_t[-1]  # (batch_size, hidden_dim)
#         hn_hrv_f = hn_hrv_f[-1]  # (batch_size, hidden_dim)
#
#         # Concatenate the hidden states from all LSTM paths
#         combined_features = torch.cat((hn_power, hn_hrv_t, hn_hrv_f), dim=1)  # (batch_size, hidden_dim * 3)
#
#         # Pass through the fully connected layer and activation
#         out = self.fc(combined_features)
#         return out


def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        correct_train = 0
        total_train = 0
        running_train_loss = 0.0

        # Training loop
        for batch in train_loader:
            power_features = batch['power'].to(device)
            hrv_time_features = batch['hrv_time'].to(device)
            hrv_freq_features = batch['hrv_freq'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            # Forward pass
            outputs = model(power_features, hrv_time_features, hrv_freq_features)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            running_train_loss += loss.item() * hrv_time_features.size(0)

            # Calculate training accuracy
            predicted = (outputs >= 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Calculate average training loss and accuracy for the epoch
        epoch_train_loss = running_train_loss / total_train
        train_losses.append(epoch_train_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Evaluate on the validation set
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            power_features = batch['power'].to(device)
            hrv_time_features = batch['hrv_time'].to(device)
            hrv_freq_features = batch['hrv_freq'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            # Forward pass
            outputs = model(power_features, hrv_time_features, hrv_freq_features)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * hrv_time_features.size(0)

            # Calculate accuracy
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_val_loss / total
    accuracy = 100 * correct / total
    return val_loss, accuracy


def test_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            power_features = batch['power'].to(device)
            hrv_time_features = batch['hrv_time'].to(device)
            hrv_freq_features = batch['hrv_freq'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            outputs = model(power_features, hrv_time_features, hrv_freq_features)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


def test_model_batchwise(model, test_loader, device):
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            power_features = batch['power'].to(device)
            hrv_time_features = batch['hrv_time'].to(device)
            hrv_freq_features = batch['hrv_freq'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            outputs = model(power_features, hrv_time_features, hrv_freq_features)
            predicted = (outputs >= 0.5).float()

    return labels, predicted


def read_input(path_save, names_ret, names_ret_nback):

    # Read the saved organized input data
    X_power = np.load(path_save + "/yasa_c3_eeg_rel_powers.npy")
    X_hrv_t = np.load(path_save + "/neurokit_hrv_params_t.npy")
    X_hrv_f = np.load(path_save + "/neurokit_hrv_params_f.npy")

    # Remove the subjects without n-back scores
    sub_id_retain = [index for index, subject in enumerate(names_ret) if subject in names_ret_nback]
    X_power = X_power[sub_id_retain]
    X_hrv_t = X_hrv_t[sub_id_retain]
    X_hrv_f = X_hrv_f[sub_id_retain]

    # You can save the updated retained subjects for later use

    return X_power, X_hrv_t, X_hrv_f


def read_output(dir_targets, task):

        targets = pd.read_csv(dir_targets)
        target = targets[task].to_numpy()

        return target


def standardization(X_train, X_val, X_test, method='all_timesteps'):

    if method == 'all_timesteps':

         scaler = StandardScaler()

         # Reshape to 2D (combine all timesteps)
         num_samples_train, num_timesteps, num_features = X_train.shape
         X_train_reshaped = X_train.reshape(-1, num_features)

         # Fit the scaler on the training data and transform it
         X_train_reshaped = scaler.fit_transform(X_train_reshaped)

         # Reshape back to 3D
         X_train_standardized = X_train_reshaped.reshape(num_samples_train, num_timesteps, num_features)

         # Apply the same transformation to validation and test data without fitting again
         num_samples_val = X_val.shape[0]
         X_val_reshaped = X_val.reshape(-1, num_features)
         X_val_reshaped = scaler.transform(X_val_reshaped)
         X_val_standardized = X_val_reshaped.reshape(num_samples_val, num_timesteps, num_features)

         num_samples_test = X_test.shape[0]
         X_test_reshaped = X_test.reshape(-1, num_features)
         X_test_reshaped = scaler.transform(X_test_reshaped)
         X_test_standardized = X_test_reshaped.reshape(num_samples_test, num_timesteps, num_features)

    elif method == "per_sample":

        def standardize_per_sample(X):

            # X is a 3D array of shape (num_samples, num_timesteps, num_features)
            num_samples, num_timesteps, num_features = X.shape
            X_standardized = np.empty_like(X)

            for i in range(num_samples):
                # For each sample, compute mean and std along the time axis for each feature
                mean = X[i].mean(axis=0)  # Mean of shape (num_features,)
                std = X[i].std(axis=0)  # Standard deviation of shape (num_features,)

                # Avoid division by zero by replacing zero std with 1 (safe operation)
                std[std == 0] = 1

                # Standardize the sample
                X_standardized[i] = (X[i] - mean) / std

            return X_standardized

        # Standardize train, validation, and test sets separately
        X_train_standardized = standardize_per_sample(X=X_train)
        X_val_standardized = standardize_per_sample(X=X_val)
        X_test_standardized = standardize_per_sample(X=X_test)

    else:
        raise ValueError("Invalid method. Choose 'all_timesteps' or 'per_sample'.")

    return X_train_standardized, X_val_standardized, X_test_standardized


# 5. Hyperparameters and Data Preparation
input_dim_power = 6  # Number of PSD features per 30-second segment (rel power of 6 frequency bands, without overlap)
input_dim_hrv_t = 6  # Number of time-domain HRV features per 2-minute segment (with 50% overlap)
input_dim_hrv_f = 5  # Number of frequency-domain HRV features per 5-minute segment (with 50% overlap)
hidden_dim = 96
num_layers = 8
output_dim = 1  # Binary classification
num_epochs = 600
batch_size = 16
learning_rate = 0.000005
task = 'impulsivity'

# last
# input_dim_power = 6  # Number of PSD features per 30-second segment (rel power of 6 frequency bands, without overlap)
# input_dim_hrv_t = 6  # Number of time-domain HRV features per 2-minute segment (with 50% overlap)
# input_dim_hrv_f = 5  # Number of frequency-domain HRV features per 5-minute segment (with 50% overlap)
# hidden_dim = 64
# num_layers = 6
# output_dim = 1  # Binary classification
# num_epochs = 150
# batch_size = 16
# learning_rate = 0.00005
# task = 'impulsivity'

# hidden_dim = 64
# num_layers = 6
# output_dim = 1  # Binary classification
# num_epochs = 800
# batch_size = 16
# learning_rate = 0.00007
# task = 'impulsivity'

# hidden_dim = 32
# num_layers = 4
# output_dim = 1  # Binary classification
# num_epochs = 250
# batch_size = 16
# learning_rate = 0.00004
# task = 'impulsivity'

path_file = "/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs"
path_save = "/media/livia/Elements/public_sleep_data/stages/stages/original/yasa_eeg_powers"
dir_targets = path_save + "/nback_scores_classification_for_yasa_c3_eeg_rel_power_analysis.csv"

# found in "test_rnn_power.py"
path_file2 = r'/media/livia/Elements/public_sleep_data/stages/stages/original/'
subj_retained_for_power_analysis = pd.read_csv(path_file2 + "yasa_eeg_powers/subjects_retained_for_power_analysis.csv", header=None)
subj_retained_for_power_analysis = subj_retained_for_power_analysis.values.flatten().tolist()
names_ret = []
for num_subj in range(len(subj_retained_for_power_analysis)):
    m = re.search('.*/([^/]+)$', subj_retained_for_power_analysis[num_subj])
    if m:
        name = m.group(1)
    names_ret.append(name)
names_ret = np.array(names_ret)
subj_retained_for_nback = pd.read_csv(path_file2 + "yasa_eeg_powers/subjects_retained_for_power_and_nback_analysis.csv", header=None)
subj_retained_for_nback = subj_retained_for_nback.values.flatten().tolist()[1:]  # Get the values and remove the header

# Read the data and retain only the subjects included in the n-back analysis
X_power, X_hrv_t, X_hrv_f = read_input(path_save, names_ret, subj_retained_for_nback)

# Read the targets and only retain the desired nback score (column)
y = read_output(dir_targets, task=task)

# Find the indexes of the train, val, and test data
fold = 1
cohort_size = len(y)
all_subjs = list(range(cohort_size))
test_subj = [fold-1]  # works only for the 1st fold, modify for other folds
# Update the outlier list (remove potential test subjects)
subj_indexes1 = [element for element in all_subjs if element not in test_subj]

# 80% of the subjects used for train
train_size = int(80*len(subj_indexes1)/100)
# the rest 20% of the subjects used for validation
val_size = len(subj_indexes1) - train_size

# Extract the train data subject indexes
random2.seed(123)
train_subj = random2.sample(subj_indexes1, train_size)
# Extract the validation data subject indexes (remaining subjects)
val_subj = [element for element in subj_indexes1 if element not in train_subj]

# X should have shape (num_samples, seq_len, feature_dim), y should have shape (num_samples,)
X_power_train = X_power[train_subj, :, :]
X_hrv_t_train = X_hrv_t[train_subj, :, :]
X_hrv_f_train = X_hrv_f[train_subj, :, :]
y_train = y[train_subj, ]
X_power_val = X_power[val_subj, :, :]
X_hrv_t_val = X_hrv_t[val_subj, :, :]
X_hrv_f_val = X_hrv_f[val_subj, :, :]
y_val = y[val_subj, ]
X_power_test = X_power[test_subj, :, :]
X_hrv_t_test = X_hrv_t[test_subj, :, :]
X_hrv_f_test = X_hrv_f[test_subj, :, :]
y_test = y[test_subj, ]

# Standardize using training set statistics
X_power_train_norm, X_power_val_norm, X_power_test_norm = standardization(X_power_train, X_power_val, X_power_test,
                                                                          method="all_timesteps")
X_hrv_t_train_norm, X_hrv_t_val_norm, X_hrv_t_test_norm = standardization(X_hrv_t_train, X_hrv_t_val, X_hrv_t_test,
                                                                          method="all_timesteps")
X_hrv_f_train_norm, X_hrv_f_val_norm, X_hrv_f_test_norm = standardization(X_hrv_f_train, X_hrv_f_val, X_hrv_f_test,
                                                                          method="all_timesteps")

# Create datasets and data loaders
train_dataset = EEGDataset(X_power_train_norm, X_hrv_t_train_norm, X_hrv_f_train_norm, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataset = EEGDataset(X_power_val_norm, X_hrv_t_val_norm, X_hrv_f_val_norm, y_val)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
test_dataset = EEGDataset(X_power_test_norm, X_hrv_t_test_norm, X_hrv_f_test_norm, y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiPathLSTMClassifier(input_dims=(input_dim_power, input_dim_hrv_t, input_dim_hrv_f),
                                hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Train and Evaluate the Model
train_losses, val_losses, \
    train_accuracies, val_accuracies = train_and_validate_model(model, train_loader, val_loader, criterion,
                                                                optimizer, num_epochs, device)
# train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device)
# evaluate_model(model, val_loader, device=device)

# Evaluate the model on the test set
test_accuracy = test_model(model, test_loader, device=device)

val_accuracy = test_model(model, val_loader, device=device)

# Test the model batchwise to get predictions and true labels and visualize the results
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
labels1, predictions1 = test_model_batchwise(model, train_loader, device=device)
labels2, predictions2 = test_model_batchwise(model, val_loader, device=device)

# Scatter Plot for Class Comparison for validation set
plt.figure()
plt.scatter(range(len(labels2)), labels2.to('cpu'), color='blue', marker='o', label='True Labels')
plt.scatter(range(len(predictions2)), predictions2.to('cpu'), color='red', marker='x', label='Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Predictions vs. True Labels')
plt.legend(loc="best")
plt.show()

# ROC Curve
fpr1, tpr1, _ = roc_curve(labels1.to('cpu'), predictions1.to('cpu'))
roc_auc1 = auc(fpr1, tpr1)
fpr2, tpr2, _ = roc_curve(labels2.to('cpu'), predictions2.to('cpu'))
roc_auc2 = auc(fpr2, tpr2)
plt.figure()
plt.plot(fpr1, tpr1, color='darkorange', lw=2, label=f'ROC curve train (area = {roc_auc1:.2f})')
plt.plot(fpr2, tpr2, color='green', lw=2, label=f'ROC curve val (area = {roc_auc2:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix for validation set
cm = confusion_matrix(labels2.to('cpu'), predictions2.to('cpu'))
ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
