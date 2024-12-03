#
# fixed transformer arch, EEG(c3-m2) powers of 6 bands + ECG time/freq HRV features over time + raw ECG/EEG signals
# Dataset: neurokit_hrv_params_f.npy, neurokit_hrv_params_t.npy, yasa_c3_eeg_rel_powers.npy
# : created in "test_rnn_power_hrv.py" from
#                                processed in "process_augmented1_dataset.py"
#
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
import os
import shutil
import csv
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


class MultiPathTransformerClassifier(nn.Module):
    def __init__(self, feat_dims, raw_dims, d_model_feat, d_model_raw, nhead, num_layers_feat, num_layers_raw,
                 dim_feedforward_feat, dim_feedforward_raw, output_dim,
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
        self.project_time_hrv = nn.Linear(feat_dims[0], d_model_feat)
        self.project_freq_hrv = nn.Linear(feat_dims[1], d_model_feat)
        self.project_power = nn.Linear(feat_dims[2], d_model_feat)
        self.project_ecg = nn.Linear(raw_dims[0], d_model_raw)
        self.project_eeg = nn.Linear(raw_dims[1], d_model_raw)

        # Positional encoding
        self.pos_enc1 = FixedPositionalEncoding(d_model_feat, dropout=dropout * (1.0 - freeze))
        self.pos_enc2 = FixedPositionalEncoding(d_model_raw, dropout=dropout * (1.0 - freeze))

        # Transformer encoder layers for each feature type
        self.transformer_time_hrv = nn.ModuleList([
            TransformerBatchNormEncoderLayer(d_model_feat, nhead, dim_feedforward_feat, dropout * (1.0 - freeze),
                                             activation=activation, norm=norm)
            for _ in range(num_layers_feat)
        ])
        self.transformer_freq_hrv = nn.ModuleList([
            TransformerBatchNormEncoderLayer(d_model_feat, nhead, dim_feedforward_feat, dropout * (1.0 - freeze),
                                             activation=activation, norm=norm)
            for _ in range(num_layers_feat)
        ])
        self.transformer_power = nn.ModuleList([
            TransformerBatchNormEncoderLayer(d_model_feat, nhead, dim_feedforward_feat, dropout * (1.0 - freeze),
                                             activation=activation, norm=norm)
            for _ in range(num_layers_feat)
        ])
        self.transformer_ecg = nn.ModuleList([
            TransformerBatchNormEncoderLayer(d_model_raw, nhead, dim_feedforward_raw, dropout * (1.0 - freeze),
                                             activation=activation, norm=norm)
            for _ in range(num_layers_raw)
        ])
        self.transformer_eeg = nn.ModuleList([
            TransformerBatchNormEncoderLayer(d_model_raw, nhead, dim_feedforward_raw, dropout * (1.0 - freeze),
                                             activation=activation, norm=norm)
            for _ in range(num_layers_raw)
        ])

        # Fully connected layer after concatenation of all streams
        self.fc = nn.Linear((d_model_feat * 3) + (d_model_raw * 2), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_time_hrv, x_freq_hrv, x_power, x_ecg, x_eeg):

        # Apply linear projection to match d_model
        x_time_hrv = self.project_time_hrv(x_time_hrv)
        x_freq_hrv = self.project_freq_hrv(x_freq_hrv)
        x_power = self.project_power(x_power)
        x_ecg = self.project_ecg(x_ecg)
        x_eeg = self.project_eeg(x_eeg)

        # Apply positional encoding
        x_time_hrv = self.pos_enc1(x_time_hrv.transpose(0, 1))
        x_freq_hrv = self.pos_enc1(x_freq_hrv.transpose(0, 1))
        x_power = self.pos_enc1(x_power.transpose(0, 1))
        x_ecg = self.pos_enc2(x_ecg.transpose(0, 1))  # CHECK
        x_eeg = self.pos_enc2(x_eeg.transpose(0, 1))

        # Process each feature type through its respective transformer encoder
        for layer in self.transformer_time_hrv:
            x_time_hrv = layer(x_time_hrv)
        for layer in self.transformer_freq_hrv:
            x_freq_hrv = layer(x_freq_hrv)
        for layer in self.transformer_power:
            x_power = layer(x_power)
        for layer in self.transformer_ecg:
            x_ecg = layer(x_ecg)
        for layer in self.transformer_eeg:
            x_eeg = layer(x_eeg)

        # Take the last output from each transformer encoder
        x_time_hrv = x_time_hrv[-1]  # (batch_size, d_model)
        x_freq_hrv = x_freq_hrv[-1]  # (batch_size, d_model)
        x_power = x_power[-1]  # (batch_size, d_model)
        x_ecg = x_ecg[-1]  # (batch_size, d_model)
        x_eeg = x_eeg[-1]  # (batch_size, d_model)

        # Concatenate the outputs from each transformer path
        combined = torch.cat((x_time_hrv, x_freq_hrv, x_power, x_ecg, x_eeg), dim=1)
        del x_time_hrv, x_freq_hrv, x_power, x_ecg, x_eeg

        # Apply dropout and pass through the fully connected layer for classification
        combined = self.dropout(combined)
        out = self.fc(combined)

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


class MyDataset(Dataset):

    def __init__(self, dir_raw, dir_features, targets, valid_indexes, subj_indexes, subset, stats=None):

        self.dir_power, self.dir_hrv_t, self.dir_hrv_f = dir_features
        self.dir_ecg, self.dir_eeg = dir_raw
        self.valid_indexes = valid_indexes
        self.subj_indexes = subj_indexes
        self.subset = subset
        self.targets = targets
        self.stats = stats
        print('number of subjects in the ', subset, 'set: ', str(len(self.subj_indexes)))

        # Read the feature files
        self.X_power = torch.tensor(np.load(self.dir_power), dtype=torch.float32)
        self.X_hrv_t = torch.tensor(np.load(self.dir_hrv_t), dtype=torch.float32)
        self.X_hrv_f = torch.tensor(np.load(self.dir_hrv_f), dtype=torch.float32)
        # Retain only the valid subjects
        self.X_power = self.X_power[self.valid_indexes]
        self.X_hrv_t = self.X_hrv_t[self.valid_indexes]
        self.X_hrv_f = self.X_hrv_f[self.valid_indexes]

    def __len__(self):
        # if self.subset == 'train':
        #     len1 = len(self.subj_indexes)*self.augmentation_rate
        # else:
        len1 = len(self.subj_indexes)  # check
        return len1

    def __getitem__(self, idx):

        samp_index = self.subj_indexes[idx]

        # Get target items corresponding to the current subset
        target = self.targets[samp_index]

        # Extract subject ID
        samp_dir_eeg = self.dir_eeg[samp_index]
        samp_dir_ecg = self.dir_ecg[samp_index]
        # subject_id = os.path.basename(os.path.dirname(samp_dir))
        # print(subject_id)

        # Load the EEG and ECG data and standardize them
        eeg = torch.tensor(np.load(samp_dir_eeg+"/eeg_C3-M2_segmented_30sec.npy"), dtype=torch.float32)
        ecg = torch.tensor(np.load(samp_dir_ecg+"/ecg_segmented_2min.npy"), dtype=torch.float32)
        eeg = self.truncate_pad(eeg, type='eeg')
        ecg = self.truncate_pad(ecg, type='ecg')
        power = self.X_power[samp_index]
        hrv_time = self.X_hrv_t[samp_index]
        hrv_freq = self.X_hrv_f[samp_index]
        if self.stats:
            eeg = (eeg - self.stats['eeg_mean']) / self.stats['eeg_std']
            ecg = (ecg - self.stats['ecg_mean']) / self.stats['ecg_std']
            power = (power - self.stats['power_mean']) / self.stats['power_std']
            hrv_time = (hrv_time - self.stats['hrv_t_mean']) / self.stats['hrv_t_std']
            hrv_freq = (hrv_freq - self.stats['hrv_f_mean']) / self.stats['hrv_f_std']

        # if self.subset == 'train':
        #     eeg = standardize_raw(eeg, method='all_timesteps', subset='train')
        #     ecg = standardize_raw(ecg, method='all_timesteps', subset='train')
        # else:
        #     eeg = standardize_raw(eeg, method = 'all_timesteps', subset='val')
        #     ecg = standardize_raw(ecg, method = 'all_timesteps', subset='val')

        return {
            'power': power,
            'hrv_time': hrv_time,
            'hrv_freq': hrv_freq,
            'ecg': ecg,
            'eeg': eeg,
            'label': torch.tensor(target, dtype=torch.float32)}

    @staticmethod
    def truncate_pad(data, type, seq_len_t=5):

        # Retain the last five hours
        if type == 'eeg':
            feature_freq = 30  # duration of the windows in sec
        elif type == 'ecg':
            feature_freq = 120
        duration = seq_len_t * 60 * 60  # duration of the signal to retain
        seq_len = int(duration / feature_freq)
        # Cut or zero-pad
        if data.shape[0] < seq_len:
            # Pad the array along the first axis (segments) to have shape (seq_len, num_features)
            # data = np.pad(data, ((0, 0), (0, seq_len - data.shape[0])), mode='constant', constant_values=0)
            pad = torch.zeros(seq_len - data.shape[0], data.shape[1])
            data = torch.cat((data, pad), dim=0)
        else:
            data = data[-seq_len:, :]

        return data


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device,
                task_output_dir, task, task_type="classification"):

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
            x_ecg = batch['ecg'].to(device)
            x_eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)

            # Ensure all inputs are Float tensors
            x_time_hrv = x_time_hrv.float()
            x_freq_hrv = x_freq_hrv.float()
            x_power = x_power.float()
            x_ecg = x_ecg.float()
            x_eeg = x_eeg.float()

            model = model.float()
            outputs = model(x_time_hrv, x_freq_hrv, x_power, x_ecg, x_eeg)

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
                x_ecg = batch['ecg'].to(device)
                x_eeg = batch['eeg'].to(device)
                labels = batch['label'].to(device)

                # Ensure all inputs are Float tensors
                x_time_hrv = x_time_hrv.float()
                x_freq_hrv = x_freq_hrv.float()
                x_power = x_power.float()
                x_ecg = x_ecg.float()
                x_eeg = x_eeg.float()

                outputs = model(x_time_hrv, x_freq_hrv, x_power, x_ecg, x_eeg)

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
    # plt.show()
    learning_curve_path = os.path.join(task_output_dir, "learning_curves_task_" + task + ".png")
    plt.savefig(learning_curve_path)
    plt.close()

    if task_type == "classification":
        return train_losses, val_losses, train_accuracies, val_accuracies
    else:
        return train_losses, val_losses


# EDIT for multiple tasks and added raw data
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


# EDIT for multiple tasks and added raw data
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


def compute_zscore_stat(train_dataset):
    """
    Compute the mean and standard deviation incrementally for large datasets.
    Args:
        dataset (Dataset): Dataset object that provides samples in __getitem__.
    Returns:
        stats (dict): Mean and standard deviation for each feature (single value per feature).
    """
    n_samples = 0

    # Initialize accumulators for mean and variance
    ecg_sum, ecg_sum_sq = 0.0, 0.0
    eeg_sum, eeg_sum_sq = 0.0, 0.0
    power_list, hrv_t_list, hrv_f_list = [], [], []

    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]  # [0]  # Access the sample dictionary

        # Extract features
        power_list.append(sample['power'])
        hrv_t_list.append(sample['hrv_time'])
        hrv_f_list.append(sample['hrv_freq'])
        ecg = sample['ecg'].numpy()  # 2D array (time x channel)
        eeg = sample['eeg'].numpy()  # 2D array (time x channel)

        # Sum and sum of squares for ECG and EEG (global)
        ecg_sum += ecg.sum()
        ecg_sum_sq += (ecg ** 2).sum()

        eeg_sum += eeg.sum()
        eeg_sum_sq += (eeg ** 2).sum()

        n_samples += 1

    # Compute mean and std
    ecg_mean = ecg_sum / (n_samples * ecg.size)
    ecg_std = np.sqrt((ecg_sum_sq / (n_samples * ecg.size)) - ecg_mean ** 2)

    eeg_mean = eeg_sum / (n_samples * eeg.size)
    eeg_std = np.sqrt((eeg_sum_sq / (n_samples * eeg.size)) - eeg_mean ** 2)

    power = np.array(power_list)
    hrv_t = np.array(hrv_t_list)
    hrv_f = np.array(hrv_f_list)
    power = power.reshape(-1, power.shape[-1])  # Shape: (num_samples_train * num_timesteps, num_features)
    hrv_t = hrv_t.reshape(-1, hrv_t.shape[-1])
    hrv_f = hrv_f.reshape(-1, hrv_f.shape[-1])

    # Step 2: Fit the scaler on the reshaped training data
    scaler_power = StandardScaler()
    scaler_hrv_t = StandardScaler()
    scaler_hrv_f = StandardScaler()
    power = scaler_power.fit_transform(power)  # Standardize the data
    hrv_t = scaler_hrv_t.fit_transform(hrv_t)
    hrv_f = scaler_hrv_f.fit_transform(hrv_f)

    # Step 3: Reshape back to 3D
    # skipped bc we don't need to use the standardized data and just need the mean and std to use later

    # Step 4: Retrieve the mean and std for each feature
    power_means = scaler_power.mean_  # Mean for each feature
    power_stds = scaler_power.scale_  # Standard deviation for each feature
    hrv_t_means = scaler_hrv_t.mean_
    hrv_t_stds = scaler_hrv_t.scale_
    hrv_f_means = scaler_hrv_f.mean_
    hrv_f_stds = scaler_hrv_f.scale_

    stats = {
        'power_mean': power_means,
        'power_std': power_stds,
        'hrv_t_mean': hrv_t_means,
        'hrv_t_std': hrv_t_stds,
        'hrv_f_mean': hrv_f_means,
        'hrv_f_std': hrv_f_stds,
        'ecg_mean': ecg_mean,
        'ecg_std': ecg_std,
        'eeg_mean': eeg_mean,
        'eeg_std': eeg_std,
    }

    return stats


def train_model_multiple_tasks(dir_features, dir_raw, names_input, target_file, model_class, output_dir, device,
                               num_epochs=300, batch_size_train=8, batch_size_val=128, batch_size_test=1,
                               learning_rate=0.00005, num_layers_feat=2,
                               num_layers_raw=2, d_model_feat=128, d_model_raw=128, nhead=4,
                               dim_feedforward_feat=512, dim_feedforward_raw=512, output_dim=1, dropout=0.1,
                               task_type="classification", freeze=False, activation="relu", norm="BatchNorm"):
    """
    Train the model for each column in the target file.
    """
    # Read the target file
    targets = pd.read_csv(target_file)
    # tasks = targets.columns.tolist()[1:]  # Exclude the 'test_sessions.subid' column
    tasks = ['pcet_concept_level_responses', 'pvtb_errors_commission']
    names_target = targets['subject_id'].to_numpy()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each task (column)
    for task in tasks:
        print(f"Training for task: {task}")

        # Get ids of subjects without nan for this task
        names_this_task = names_target[~np.isnan(targets[task])]
        # Ids of subjects that are in both the input and target data
        subj_to_ret = np.intersect1d(names_input, names_this_task)
        # Get target values for the subjects retained
        targets1 = targets[targets['subject_id'].isin(subj_to_ret)]
        # Extract the target values for the current task
        y_task = targets1[task].to_numpy()
        # Indices of subjects retained for this task in the input data
        valid_indices = np.where(np.isin(names_input, subj_to_ret))[0]

        # compare subjs in target file and raw data dir (to see if you can use "valid_indices" for raw data)

        # Modify raw data directories to include only the subjects retained for this task
        dir_raw_mod = tuple([[lst[i] for i in valid_indices] for lst in dir_raw])

        # Split data into train/val/test sets (1 subject test, ~80% train, ~20% val)
        # Find the indexes of the train, val, and test data
        fold = 1
        cohort_size = len(y_task)
        all_subjs = list(range(cohort_size))
        test_subj = [fold - 1]  # works only for the 1st fold, modify for other folds
        # Update the outlier list (remove potential test subjects)
        subj_indexes1 = [element for element in all_subjs if element not in test_subj]

        # 80% of the subjects used for train
        train_size = int(80 * len(subj_indexes1) / 100)
        # the rest 20% of the subjects used for validation
        val_size = len(subj_indexes1) - train_size

        # Extract the train data subject indexes
        random2.seed(123)
        train_subj = random2.sample(subj_indexes1, train_size)
        # Extract the validation data subject indexes (remaining subjects)
        val_subj = [element for element in subj_indexes1 if element not in train_subj]

        # Create PyTorch datasets and loaders
        train_dataset = MyDataset(dir_raw_mod, dir_features, y_task, valid_indices, train_subj, 'train')
        # Extract feature and raw signal dimensions
        feat_dims = (train_dataset.X_hrv_t.shape[-1], train_dataset.X_hrv_f.shape[-1], train_dataset.X_power.shape[-1])
        raw_dims = (train_dataset[0]['ecg'].shape[-1], train_dataset[0]['eeg'].shape[-1])
        # stats = compute_zscore_stat(train_dataset)
        # with open("stats/stats.csv", "w", newline="") as csv_file:
        #     writer = csv.writer(csv_file)
        #     for key, value in stats.items():
        #         if isinstance(value, (np.ndarray, list)):  # Check if array or list
        #             writer.writerow([key] + value.tolist())
        #         else:  # For scalar values, wrap them in a list
        #             writer.writerow([key, value])
        # Load stats.csv (computed and saved in the previous step)
        stats = {}
        with open("stats/stats.csv", "r") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                stats[row[0]] = list(map(float, row[1:]))
        stats = {key: value[0] for key, value in stats.items()}
        train_dataset = MyDataset(dir_raw_mod, dir_features, y_task, valid_indices, train_subj, 'train', stats)
        # a = train_dataset[0]
        val_dataset = MyDataset(dir_raw_mod, dir_features, y_task, valid_indices, val_subj, 'val', stats)
        test_dataset = MyDataset(dir_raw_mod, dir_features, y_task, valid_indices, test_subj, 'test', stats)

        # X_power_train, X_hrv_t_train, X_hrv_f_train, dir_ecg_mod, dir_eeg_mod, y_train)
        # val_dataset = MyDataset(X_power_val, X_hrv_t_val, X_hrv_f_val, dir_ecg_mod, dir_eeg_mod, y_val)
        # test_dataset = MyDataset(X_power_test, X_hrv_t_test, X_hrv_f_test, dir_ecg_mod, dir_eeg_mod, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

        # Initialize model, optimizer, and loss function
        # feat_dims = (X_power_train.shape[-1], X_hrv_t_train.shape[-1], X_hrv_f_train.shape[-1])
        # raw_dims = (X_ecg_train.shape[-1], X_eeg_train.shape[-1])
        model = model_class(feat_dims=feat_dims, raw_dims=raw_dims, d_model_feat=d_model_feat, d_model_raw=d_model_raw,
                            nhead=nhead, num_layers_feat=num_layers_feat, num_layers_raw=num_layers_raw,
                            dim_feedforward_feat=dim_feedforward_feat, dim_feedforward_raw=dim_feedforward_raw,
                            output_dim=output_dim, dropout=dropout, activation=activation, norm=norm, freeze=freeze,
                            task_type=task_type).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if task_type == "classification":
            if output_dim == 1:
                criterion = nn.BCEWithLogitsLoss()  # Binary classification
            else:
                criterion = nn.CrossEntropyLoss()  # Multi-class classification
        elif task_type == "regression":
            criterion = nn.MSELoss()  # Mean Squared Error for regression
        else:
            raise ValueError("task_type should be 'classification' or 'regression'")

        # Save results
        task_output_dir = os.path.join(output_dir, task)
        os.makedirs(task_output_dir, exist_ok=True)

        # Train and validate the model
        if task_type == "classification":
            train_losses, val_losses, train_accuracies, val_accuracies = \
                train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device,
                                         task_output_dir, task, task_type)
        else:
            train_losses, val_losses = \
                train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device,
                                         task_output_dir, task, task_type)

        # Save metrics
        if task_type == "classification":
            metrics = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accuracies": train_accuracies,
                "val_accuracies": val_accuracies
            }
        else:
            metrics = {
                "train_losses": train_losses,
                "val_losses": val_losses
            }
        pd.DataFrame(metrics).to_csv(os.path.join(task_output_dir, "metrics.csv"), index=False)

    print(f"Training completed for all tasks. Results saved in {output_dir}.")
    a = 0


# 5. Hyperparameters and Data Preparation
d_model_raw = 128
d_model_feat = 8  # 15-64
nhead = 8
num_layers_feat = 2
num_layers_raw = 2

dim_feedforward_feat = 16  # 64-128
dim_feedforward_raw = 92  #
output_dim = 1  # Set to 1 for binary classification or regression; set to number of classes for multi-class classific
dropout = 0.1
task_type = "classification"

num_epochs = 20
batch_size_train = 16
batch_size_val = 64
batch_size_test = 1
learning_rate = 0.0001

path_file = "/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs"
path_save = "/media/livia/Elements/public_sleep_data/stages/stages/original/yasa_eeg_powers"
dir_targets = path_save + "/all_scores_classification_for_yasa_c3_eeg_rel_power_analysis.csv"
dir_ecg = sorted(glob2.glob(path_file + '/[!p]*/yasa_outputs/ecg_segmented_2min/*'))
dir_eeg = sorted(glob2.glob(path_file + '/[!p]*/yasa_outputs/eeg_segmented_30sec/*'))

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

# Read the extracted EEG/ECG features
dir_power = path_save + "/yasa_c3_eeg_rel_powers.npy"
dir_hrv_t = path_save + "/neurokit_hrv_params_t.npy"
dir_hrv_f = path_save + "/neurokit_hrv_params_f.npy"

# ID of subjects with proper data (found in "test_rnn_power.py" read_input)
path_file2 = r'/media/livia/Elements/public_sleep_data/stages/stages/original/'
subj_retained_for_power_analysis = pd.read_csv(path_file2 + "yasa_eeg_powers/subjects_retained_for_power_analysis.csv", header=None)
subj_retained_for_power_analysis = subj_retained_for_power_analysis.values.flatten().tolist()
names_input = []
for num_subj in range(len(subj_retained_for_power_analysis)):
    m = re.search('.*/([^/]+)$', subj_retained_for_power_analysis[num_subj])
    if m:
        name = m.group(1)
    names_input.append(name)
names_input = np.array(names_input)

# # Names ecg
# names_ecg = []
# for num_subj in range(len(dir_ecg)):
#     m = re.search('.*/([^/]+)$', dir_ecg[num_subj])
#     if m:
#         name = m.group(1)
#     names_ecg.append(name)
# names_ecg = np.array(names_ecg)

# # Names eeg
# names_eeg = []
# for num_subj in range(len(dir_eeg)):
#     m = re.search('.*/([^/]+)$', dir_eeg[num_subj])
#     if m:
#         name = m.group(1)
#     names_eeg.append(name)
# names_eeg = np.array(names_eeg)
# print(sum(names_ecg == names_input)==len(names_ecg))

# Note: names_ecg and names_input are the same, names_eeg has a few more bc some subjs were removed in yasa data processing

# # Remove the extra eeg directories from PC (Ran once, no need to run again)
# set1 = set(names_input)
# set2 = set(names_eeg)
# noncommon_ids = set2.symmetric_difference(set1)
# noncommon_ids_list = sorted(list(noncommon_ids))

# # Find the directories to delete
# directories_to_delete = [
#     dir_path for dir_path in dir_eeg if any(sub_id in dir_path for sub_id in noncommon_ids_list)]
# # Remove the directories and their contents
# for dir_path in directories_to_delete:
#     if os.path.exists(dir_path):
#         print(f"Removing directory and its contents: {dir_path}")
#         shutil.rmtree(dir_path)  # Deletes the directory and all its contents
#     else:
#         print(f"Directory not found: {dir_path}")
# # Rerun the dir_eeg load and names_eeg creation and then run the below line to check if the names match
# print(sum(names_ecg == names_eeg)==len(names_ecg))  # Should be True
# Note: did all these to make sure the subjects of eeg, ecg, and feature data are the same

output_dir = "/home/livia/PycharmProjects/YASA_proj/results/transformer_power_hrv_features_raw_v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model_multiple_tasks(
    dir_features=(dir_power, dir_hrv_t, dir_hrv_f),
    dir_raw=(dir_ecg, dir_eeg),
    names_input=names_input,
    target_file=dir_targets,
    model_class=MultiPathTransformerClassifier,
    output_dir=output_dir,
    device=device,
    num_epochs=num_epochs,  # Reduced epochs for faster testing
    batch_size_train=8,
    batch_size_val=128,
    batch_size_test=1,
    learning_rate=0.00001,
    d_model_feat=d_model_feat,
    d_model_raw=d_model_raw,
    nhead=nhead,
    num_layers_feat=num_layers_feat,
    num_layers_raw=num_layers_raw,
    dim_feedforward_feat=dim_feedforward_feat,
    dim_feedforward_raw=dim_feedforward_raw,
    output_dim=output_dim,
    dropout=dropout,
    task_type=task_type,
    freeze=False,
    activation="relu",
    norm="BatchNorm"
)

