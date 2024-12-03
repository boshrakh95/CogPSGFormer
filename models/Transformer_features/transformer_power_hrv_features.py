#
# fixed transformer arch, EEG(c3-m2) powers of 6 bands + ECG time/freq HRV features over time
# Dataset: neurokit_hrv_params_f.npy, neurokit_hrv_params_t.npy, yasa_c3_eeg_rel_powers.npy
# : created in "test_rnn_power_hrv.py" from
#                                processed in "process_augmented1_dataset.py"
# Data formed from STAGES data (clinics with similar channels and subjects with nback results and valid signals),
# 735 individuals,
# possible targets: nback impulsivity (false positive)
# no demographics (should be prepared)
# no sleep stages (insensible sleep stage score and synchronization problems should be resolved)
# attention mask: don't use, artifact mask already applied
# 10fold-CV: Fold1 (note: train, val, and test data are separated based on subjects)
# original data of one subject: test data

# 14 Nov 2024

from typing import Optional, Any
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import numpy as np
import glob2
from torch.nn import Linear, ReLU, MSELoss,Module, Dropout
from torch.optim import Adam
import random2
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')

torch.cuda.is_available()


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBatchNormEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu", norm="BatchNorm"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if norm == "BatchNorm":
            self.norm1 = nn.BatchNorm1d(d_model, eps=1e-5)
            self.norm2 = nn.BatchNorm1d(d_model, eps=1e-5)
        elif norm == "LayerNorm":
            self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
            self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        else:
            raise ValueError("norm should be 'BatchNorm' or 'LayerNorm'")

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        # Apply normalization layer based on chosen type (BatchNorm/LayerNorm)
        if isinstance(self.norm1, nn.BatchNorm1d):
            src = self.norm1(src.permute(1, 2, 0)).permute(2, 0, 1)  # BatchNorm needs (batch_size, d_model, seq_len)
        else:
            src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        if isinstance(self.norm2, nn.BatchNorm1d):
            src = self.norm2(src.permute(1, 2, 0)).permute(2, 0, 1)
        else:
            src = self.norm2(src)

        return src


# MultiPath Transformer Classifier
class MultiPathTransformerClassifier(nn.Module):
    def __init__(self, input_dims, d_model, nhead, num_layers, dim_feedforward, output_dim,
                 dropout=0.1, activation="relu", norm="BatchNorm", freeze=False, task_type="classification"):
        """
                Args:
                    input_dims (tuple): Feature dimensions for each input stream.
                    d_model (int): Target feature dimension for each stream after projection.
                    nhead (int): Number of attention heads in the transformer.
                    num_layers (int): Number of transformer layers for each stream.
                    dim_feedforward (int): Dimension of the feedforward network.
                    output_dim (int): Number of output classes for classification or 1 for regression.
                    dropout (float): Dropout rate.
                    pos_encoding (str): Type of positional encoding ('fixed').
                    activation (str): Activation function ('relu' or 'gelu').
                    norm (str): Type of normalization ('BatchNorm' or 'LayerNorm').
                    freeze (bool): Whether to freeze the dropout for fine-tuning.
                    task_type (str): Task type - 'classification' or 'regression'.
                """
        super(MultiPathTransformerClassifier, self).__init__()

        self.task_type = task_type
        self.output_dim = output_dim

        # Linear projection layers for each stream to match d_model
        self.project_time_hrv = nn.Linear(input_dims[0], d_model)
        self.project_freq_hrv = nn.Linear(input_dims[1], d_model)
        self.project_power = nn.Linear(input_dims[2], d_model)

        # Positional encoding
        self.pos_enc = FixedPositionalEncoding(d_model, dropout=dropout * (1.0 - freeze))

        # Transformer encoder layers for each feature type
        self.transformer_time_hrv = nn.ModuleList([
            TransformerBatchNormEncoderLayer(d_model, nhead, dim_feedforward, dropout * (1.0 - freeze),
                                             activation=activation, norm=norm)
            for _ in range(num_layers)
        ])
        self.transformer_freq_hrv = nn.ModuleList([
            TransformerBatchNormEncoderLayer(d_model, nhead, dim_feedforward, dropout * (1.0 - freeze),
                                             activation=activation, norm=norm)
            for _ in range(num_layers)
        ])
        self.transformer_power = nn.ModuleList([
            TransformerBatchNormEncoderLayer(d_model, nhead, dim_feedforward, dropout * (1.0 - freeze),
                                             activation=activation, norm=norm)
            for _ in range(num_layers)
        ])

        # Fully connected layer after concatenation of all streams
        self.fc = nn.Linear(d_model * 3, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_time_hrv, x_freq_hrv, x_power):
        # Apply linear projection to match d_model
        x_time_hrv = self.project_time_hrv(x_time_hrv)
        x_freq_hrv = self.project_freq_hrv(x_freq_hrv)
        x_power = self.project_power(x_power)

        # Apply positional encoding
        x_time_hrv = self.pos_enc(x_time_hrv.transpose(0, 1))
        x_freq_hrv = self.pos_enc(x_freq_hrv.transpose(0, 1))
        x_power = self.pos_enc(x_power.transpose(0, 1))

        # Process each feature type through its respective transformer encoder
        for layer in self.transformer_time_hrv:
            x_time_hrv = layer(x_time_hrv)
        for layer in self.transformer_freq_hrv:
            x_freq_hrv = layer(x_freq_hrv)
        for layer in self.transformer_power:
            x_power = layer(x_power)

        # Take the last output from each transformer encoder
        x_time_hrv = x_time_hrv[-1]  # (batch_size, d_model)
        x_freq_hrv = x_freq_hrv[-1]  # (batch_size, d_model)
        x_power = x_power[-1]  # (batch_size, d_model)

        # Concatenate the outputs from each transformer path
        combined_features = torch.cat((x_time_hrv, x_freq_hrv, x_power), dim=1)  # (batch_size, d_model * 3)

        # Apply dropout and pass through the fully connected layer for classification
        combined_features = self.dropout(combined_features)
        out = self.fc(combined_features)

        # Apply final activation based on task type
        if self.task_type == "classification":
            if self.output_dim == 1:
                # For binary classification, raw logits without activation (for BCEWithLogitsLoss)
                return out  # Output shape should be (batch_size, 1)
            else:
                # For multi-class classification, use logits for CrossEntropyLoss
                return out
        elif self.task_type == "regression":
            return out  # Raw output for regression
        else:
            raise ValueError("task_type should be 'classification' or 'regression'")


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


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device, task_type="classification"):

    model.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = [] if task_type == "classification" else None
    val_accuracies = [] if task_type == "classification" else None
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            x_power = batch['power'].to(device)
            x_time_hrv = batch['hrv_time'].to(device)
            x_freq_hrv = batch['hrv_freq'].to(device)
            labels = batch['label'].to(device)

            outputs = model(x_time_hrv, x_freq_hrv, x_power)

            if task_type == "classification":
                if model.output_dim == 1:
                    loss = criterion(outputs, labels.float().unsqueeze(1))  # Binary classification
                    preds = torch.round(torch.sigmoid(outputs)).squeeze(1)  # Sigmoid + threshold at 0.5
                else:
                    loss = criterion(outputs, labels.long())  # Multi-class classification
                    preds = torch.argmax(outputs, dim=1)
            elif task_type == "regression":
                loss = criterion(outputs.squeeze(), labels.float())  # Regression
                preds = outputs.squeeze()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            if task_type == "classification":
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        if task_type == "classification":
            train_accuracy = 100 * train_correct / train_total
            train_accuracies.append(train_accuracy)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        else:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                x_power = batch['power'].to(device)
                x_time_hrv = batch['hrv_time'].to(device)
                x_freq_hrv = batch['hrv_freq'].to(device)
                labels = batch['label'].to(device)

                outputs = model(x_time_hrv, x_freq_hrv, x_power)

                if task_type == "classification":
                    if model.output_dim == 1:
                        loss = criterion(outputs, labels.float().unsqueeze(1))
                        preds = torch.round(torch.sigmoid(outputs)).squeeze(1)
                    else:
                        loss = criterion(outputs, labels.long())
                        preds = torch.argmax(outputs, dim=1)
                elif task_type == "regression":
                    loss = criterion(outputs.squeeze(), labels.float())
                    preds = outputs.squeeze()

                val_loss += loss.item() * labels.size(0)
                if task_type == "classification":
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if task_type == "classification":
            val_accuracy = 100 * val_correct / val_total
            val_accuracies.append(val_accuracy)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        else:
            print(f"Validation Loss: {val_loss:.4f}")

    # Plot training/validation loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    ax1.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    ax1.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    # Plot Accuracy (only if classification task)
    if task_type == "classification":
        ax2.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
        ax2.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()

    plt.tight_layout()
    plt.show()

    if task_type == "classification":
        return train_losses, val_losses, train_accuracies, val_accuracies
    else:
        return train_losses, val_losses


def test_model(model, test_loader, device, task_type="classification"):

    model.eval()
    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            power_features = batch['power'].to(device)
            hrv_time_features = batch['hrv_time'].to(device)
            hrv_freq_features = batch['hrv_freq'].to(device)
            labels = batch['label'].to(device)

            outputs = model(power_features, hrv_time_features, hrv_freq_features)

            if task_type == "classification":
                if model.output_dim == 1:
                    # Binary classification
                    predicted = (torch.sigmoid(outputs) >= 0.5).float()
                else:
                    # Multi-class classification
                    _, predicted = torch.max(outputs, 1)  # Get the index of the max logit
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            elif task_type == "regression":
                # For regression, you can compute test loss if a criterion is provided
                test_loss += F.mse_loss(outputs.squeeze(), labels.float(), reduction='sum').item()  # Example loss calculation

    if task_type == "classification":
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy
    elif task_type == "regression":
        # Compute Mean Squared Error for regression
        mean_squared_error = test_loss / total
        print(f'Test Mean Squared Error: {mean_squared_error:.4f}')
        return mean_squared_error


def test_model_batchwise(model, test_loader, device, task_type="classification"):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            power_features = batch['power'].to(device)
            hrv_time_features = batch['hrv_time'].to(device)
            hrv_freq_features = batch['hrv_freq'].to(device)
            labels = batch['label'].to(device)

            outputs = model(power_features, hrv_time_features, hrv_freq_features)

            if task_type == "classification":
                if model.output_dim == 1:
                    # Binary classification
                    predicted = (torch.sigmoid(outputs) >= 0.5).float()  # Apply sigmoid and threshold at 0.5
                else:
                    # Multi-class classification
                    _, predicted = torch.max(outputs, 1)  # Get the index of the max logit
            elif task_type == "regression":
                # For regression, output raw values as predictions
                predicted = outputs.squeeze()

            # Append labels and predictions for all batches
            all_labels.append(labels)
            all_predictions.append(predicted)

    # Concatenate all batches to form final tensors
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    return all_labels, all_predictions


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
input_dims = (6, 5, 6)  # Number of time-domain HRV features per 2-minute segment (with 50% overlap)
                        # Number of frequency-domain HRV features per 5-minute segment (with 50% overlap)
                        # Number of PSD features per 30-second segment (rel power of 6 frequency bands, without overlap)
d_model = 64
nhead = 8
num_layers = 3
dim_feedforward = 64
output_dim = 1  # Set to 1 for binary classification or regression; set to number of classes for multi-class classification
dropout = 0.1
task_type = "classification"

num_epochs = 150
batch_size_train = 16
batch_size_val = 64
batch_size_test = 1
learning_rate = 0.0001
task = 'impulsivity'

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
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
val_dataset = EEGDataset(X_power_val_norm, X_hrv_t_val_norm, X_hrv_f_val_norm, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
test_dataset = EEGDataset(X_power_test_norm, X_hrv_t_test_norm, X_hrv_f_test_norm, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if task_type == "classification":
    if output_dim == 1:
        criterion = nn.BCEWithLogitsLoss()  # Binary classification
    else:
        criterion = nn.CrossEntropyLoss()  # Multi-class classification
elif task_type == "regression":
    criterion = nn.MSELoss()  # Mean Squared Error for regression
else:
    raise ValueError("task_type should be 'classification' or 'regression'")

model = MultiPathTransformerClassifier(
    input_dims=input_dims,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    output_dim=output_dim,
    dropout=dropout,
    activation="relu",
    norm="BatchNorm",
    freeze=False,
    task_type=task_type
)

optimizer = Adam(model.parameters(), lr=learning_rate)

if task_type == "classification":
    train_losses, val_losses, train_accuracies, val_accuracies = \
        train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device, task_type=task_type)
else:
    train_losses, val_losses = \
        train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device, task_type=task_type)


if task_type == "classification":
    test_accuracy = test_model(model, test_loader, device, task_type=task_type)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
else:
    test_loss = test_model(model, test_loader, device, task_type=task_type)
    print(f"Test Loss: {test_loss:.4f}")

a = 0

