# update from test_rnn_power.py: Time/Freq HRV params added and the model is updated to multi-path LSTM

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
    def __init__(self, input_dims, hidden_dim, num_layers, output_dim, dropout=0.3):
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
        self.lstm_power = nn.LSTM(input_dims[0], hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.lstm_hrv_t = nn.LSTM(input_dims[1], hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.lstm_hrv_f = nn.LSTM(input_dims[2], hidden_dim, num_layers, batch_first=True, dropout=dropout)

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


# def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
#     model.train()
#     for epoch in range(num_epochs):
#         correct = 0
#         total = 0
#         running_loss = 0.0
#
#         for batch in train_loader:
#             # Extract each feature type and label from the batch
#             power_features = batch['power'].to(device)
#             hrv_time_features = batch['hrv_time'].to(device)
#             hrv_freq_features = batch['hrv_freq'].to(device)
#             labels = batch['label'].to(device).unsqueeze(1)  # Ensure labels have the shape (batch_size, 1)
#
#             # Forward pass with multi-path inputs
#             outputs = model(power_features, hrv_time_features, hrv_freq_features)
#
#             # Compute the loss
#             loss = criterion(outputs, labels)
#
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # Calculate running loss
#             running_loss += loss.item() * power_features.size(0)  # Use batch size for total loss calculation
#
#             # Calculate predictions and accuracy
#             predicted = (outputs >= 0.5).float()  # Threshold at 0.5 for binary classification
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)
#
#         # Calculate average loss for the epoch
#         epoch_loss = running_loss / total
#
#         # Calculate accuracy for the epoch
#         accuracy = 100 * correct / total
#
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
#
#
# def evaluate_model(model, val_loader, device):
#     model.eval()  # Set model to evaluation mode
#     correct = 0
#     total = 0
#
#     with torch.no_grad():  # Disable gradient calculation for evaluation
#         for batch in val_loader:
#             # Extract each feature type and label from the batch
#             power_features = batch['power'].to(device)
#             hrv_time_features = batch['hrv_time'].to(device)
#             hrv_freq_features = batch['hrv_freq'].to(device)
#             labels = batch['label'].to(device).unsqueeze(1)  # Ensure labels have the shape (batch_size, 1)
#
#             # Forward pass with multi-path inputs
#             outputs = model(power_features, hrv_time_features, hrv_freq_features)
#
#             # Calculate predictions
#             predicted = (outputs >= 0.5).float()
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     # Calculate accuracy
#     accuracy = 100 * correct / total
#     print(f'Validation Accuracy: {accuracy:.2f}%')
#     return accuracy


def read_input(dir_subjs_power, dir_subjs_hrv, bands, hrv_t_names, hrv_f_names, seq_len_t, path_save):

    all_subj_power = []
    all_subj_hrv_t = []
    all_subj_hrv_f = []
    subjects_with_too_many_nans = []
    subjects_to_retain = []
    print('Processing power files...')
    for num in range(len(dir_subjs_power)):
        # num = num + 53

        if num % 100 == 0:
            print(f'subject {num} from {len(dir_subjs_power)}...')
        all_bands = []
        for band in bands:
            band_data = pd.read_csv(os.path.join(dir_subjs_power[num], band+"_rel_power_all_eeg_channels_30sec.csv"))

            # Filter the DataFrame to get the row where column 0 matches "C3-M2"
            # Assuming "C3-M2" is in the first column, which is accessible using band_data.iloc[:, 0]
            row_data = band_data[band_data.iloc[:, 0] == "C3-M2"]
            if not row_data.empty:
                # Extract all values except the first column (which contains the label)
                all_bands.append(row_data.iloc[:, 1:].to_numpy())
        all_bands = np.vstack(all_bands)

        # # Retain the last five hours
        # duration = seq_len_t*60*60  # duration of the signal to retain
        # seq_len = int(duration/feature_freq)
        # # Cut or zero-pad
        # if all_bands.shape[1] < seq_len:
        #     # Pad the array along the second axis (columns) to have a shape of (6, 300)
        #     all_bands = np.pad(all_bands, ((0, 0), (0, seq_len-all_bands.shape[1])), mode='constant', constant_values=0)
        # else:
        #     all_bands = all_bands[:, -seq_len:]

        # Check if there are too many nans (original signal has been mostly zero)
        num_nans = np.isnan(all_bands[0, :]).sum()
        threshold = len(all_bands[0, :]) / 2  # Calculate 1/10th of the data length
        if num_nans > threshold:  # Check if the number of NaNs exceeds the threshold
            subjects_with_too_many_nans.append(dir_subjs_power[num])
            print(f"{dir_subjs_power[num]} has too many NaNs: {num_nans} out of {len(all_bands[0, :])}")
        else:
            all_bands = np.nan_to_num(all_bands, nan=0.0)  # Replace NaNs with 0 if the count is within acceptable limit
            # subjects_data[subject_name] = data  # Save the updated data back
            all_subj_power.append(all_bands)
            subjects_to_retain.append(dir_subjs_power[num])
    # all_subj_power = np.array(all_subj_power)

    # # squeeze
    # all_subj_power = np.squeeze(all_subj_power, axis=0)
    # transpose
    # all_subj_power = all_subj_power.transpose((0, 2, 1))

    print('Processing HRV files...')
    for num in range(len(dir_subjs_hrv)):

        if num % 100 == 0:
            print(f'subject {num} from {len(dir_subjs_hrv)}...')
        hrv_f = pd.read_csv(os.path.join(dir_subjs_hrv[num], "frequency_hrv_params_5min_50%overlap_2.csv"))
        hrv_t = pd.read_csv(os.path.join(dir_subjs_hrv[num], "time_hrv_params_2min_50%overlap_2.csv"))
        # Only keep the desired columns (features)
        hrv_f = hrv_f[hrv_f_names]
        hrv_t = hrv_t[hrv_t_names]

        # Only keep the non-nan rows (timesteps)
        # Step 1: Find indices of rows with NaNs in hrv_t and hrv_f
        nan_indices_t = np.where(np.isnan(hrv_t).any(axis=1))[0]
        nan_indices_f = np.where(np.isnan(hrv_f).any(axis=1))[0]

        # Step 2: Map NaN indices to time windows in power_features

        # For each NaN in `hrv_t_features`, mark 4 corresponding 30-sec segments (2 min with 50% overlap)
        power_nan_indices = set()
        for idx in nan_indices_t:
            start_idx = idx * 2  # Each 2-min segment starts every 1 minute
            power_nan_indices.update(range(start_idx, start_idx + 4))
        # For each NaN in `hrv_f_features`, mark 10 corresponding 30-sec segments (5 min with 50% overlap)
        for idx in nan_indices_f:
            start_idx = idx * 5  # Each 5-min segment starts every 2.5 minutes
            power_nan_indices.update(range(start_idx, start_idx + 10))

        # power_nan_indices = sorted([idx for idx in power_nan_indices if idx < len(all_subj_power)])

        # Step 3: Remove NaN time windows from all arrays
        power_nan_indices = sorted(power_nan_indices)
        all_subj_power[num] = np.delete(all_subj_power[num].transpose((1, 0)), power_nan_indices, axis=0)
        hrv_t_features = hrv_t[~np.isnan(hrv_t).any(axis=1)]
        hrv_f_features = hrv_f[~np.isnan(hrv_f).any(axis=1)]

        # Step 4: Retain or Pad to Cover 5 Hours with Overlap Considered
        # Calculate required epochs for 5 hours for each array, accounting for 50% overlap
        target_duration_sec = seq_len_t * 60 * 60  # hours to retain (in seconds), 5 hours

        # Power features (no overlap): 30-second segments
        power_target_epochs = target_duration_sec // 30  # Each segment is 30 seconds, no overlap

        # Time-domain HRV features with 50% overlap: 2-minute segments
        # With 50% overlap, each 2-minute segment starts every 1 minute (60 seconds)
        hrv_t_target_epochs = target_duration_sec // 60  # Effective epoch length is 60 seconds

        # Frequency-domain HRV features with 50% overlap: 5-minute segments
        # With 50% overlap, each 5-minute segment starts every 2.5 minutes (150 seconds)
        hrv_f_target_epochs = target_duration_sec // 150  # Effective epoch length is 150 seconds

        # Zero-pad each array if shorter than needed
        all_subj_power[num] = np.pad(all_subj_power[num], ((0, max(0, power_target_epochs - len(all_subj_power[num]))), (0, 0)),
                                     mode='constant')
        hrv_t_features = np.pad(hrv_t_features, ((0, max(0, hrv_t_target_epochs - len(hrv_t_features))), (0, 0)),
                                mode='constant')
        hrv_f_features = np.pad(hrv_f_features, ((0, max(0, hrv_f_target_epochs - len(hrv_f_features))), (0, 0)),
                                mode='constant')

        # Truncate each array to exactly 5 hours if they exceed the target
        all_subj_power[num] = all_subj_power[num][:power_target_epochs]
        all_subj_hrv_t.append(hrv_t_features[:hrv_t_target_epochs])
        all_subj_hrv_f.append(hrv_f_features[:hrv_f_target_epochs])

    all_subj_power = np.array(all_subj_power)
    all_subj_hrv_f = np.array(all_subj_hrv_f)
    all_subj_hrv_t = np.array(all_subj_hrv_t)

    # save the data as one file
    np.save(path_save+"/yasa_c3_eeg_rel_powers.npy", all_subj_power)  # (subj, seq_len1, band), rel power of C3-M2 in 6 bands
    np.save(path_save+"/neurokit_hrv_params_t.npy", all_subj_hrv_t)  # (subj, seq_len2, hrv_param_t), time domain HRV params
    np.save(path_save+"/neurokit_hrv_params_f.npy", all_subj_hrv_f)  # (subj, seq_len3, hrv_param_f), freq domain HRV params
    # df1 = pd.DataFrame(subjects_to_retain)
    # df1.to_csv(path_save + '/subjects_retained_for_power_analysis.csv', index=False, header=False)

    return all_subj_power, all_subj_hrv_t, all_subj_hrv_f, subjects_with_too_many_nans, subjects_to_retain

###################
# check dim of model output and if it matches the target shape (num_samples, 1) or (num_samples,)
###################

# def create_output(dir_target, path_file, path_save, subj_to_ret):
#
#     y = torch.load(dir_target)
#     dir1 = sorted(glob2.glob(path_file + '/[!p]*/processed_luna/*.edf'))
#
#     # extract all the subject IDs
#     names_all = []
#     for num_subj in range(len(dir1)):
#         m = re.search('luna/(.+?)-luna', dir1[num_subj])
#         if m:
#             name = m.group(1)
#         # name = name.replace('/', '')
#         names_all.append(name)
#     names_all = np.array(names_all)
#
#     names_ret = []
#     for num_subj in range(len(subj_to_ret)):
#         m = re.search('.*/([^/]+)$', subj_to_ret[num_subj])
#         if m:
#             name = m.group(1)
#         # name = name.replace('/', '')
#         names_ret.append(name)
#     names_ret = np.array(names_ret)
#
#     # Find indexes of subjects that should be retained (CHECK SOME EXAMPLES TO MAKE SURE THEY ARE CORRECT)
#     retain_indexes = [index for index, subject in enumerate(names_all) if subject in names_ret]
#     y_new = y[retain_indexes]
#     np.save(path_save + "/pvtb_rrt_values_classification_threshold_median_for_yasa_c3_eeg_rel_power_analysis.npy", y_new)  # (subj, 1)
#
#     return y_new


def create_output(dir_targets, subj_to_ret):

    pvtb_rrt, pvtb_lap_num, pvtb_lap_prob, subj_ids_all = np.array([]), np.array([]), np.array([]), np.array([])
    for tar in range(len(dir_targets)):
        targets = pd.read_excel(dir_targets[tar])
        # Extract the desired scores
        rrt = targets['PVTB.PVTB_MEAN_RRT'].to_numpy()
        lap_num = targets['PVTB.PVTB_500MS_LAP'].to_numpy()
        lap_prob = targets['PVTB.PVTB_500MS_LAPPROB'].to_numpy()
        print("Number of subjects in ", dir_targets[tar], "is: ", len(rrt))
        subj_ids = targets['test_sessions.subid'].to_numpy()

        # Concatenate the values from all the subsets
        pvtb_rrt = np.concatenate((pvtb_rrt, rrt))
        pvtb_lap_num = np.concatenate((pvtb_lap_num, lap_num))
        pvtb_lap_prob = np.concatenate((pvtb_lap_prob, lap_prob))
        subj_ids_all = np.concatenate((subj_ids_all, subj_ids))

    names_ret = []
    for num_subj in range(len(subj_to_ret)):
        m = re.search('.*/([^/]+)$', subj_to_ret[num_subj])
        if m:
            name = m.group(1)
        # name = name.replace('/', '')
        names_ret.append(name)
    names_ret = np.array(names_ret)

    sub_id_retain = [index for index, subject in enumerate(subj_ids_all) if subject in names_ret]
    # missing_subjects = [subject for subject in names_ret if subject not in subj_ids_all]

    # Convert regression to binary classification
    # Find the class thresholds (median of the values)
    thresh_rrt = np.median(pvtb_rrt)
    print(np.sum(pvtb_rrt >= np.median(pvtb_rrt)), "subj with high rrt")
    print(np.sum(pvtb_rrt < np.median(pvtb_rrt)), "subj with low rrt")
    thresh_lap_num = np.median(pvtb_lap_num)
    print(np.sum(pvtb_lap_num >= np.median(pvtb_lap_num)), "subj with high lap_num")
    print(np.sum(pvtb_lap_num < np.median(pvtb_lap_num)), "subj with low lap_num")
    thresh_lap_prob = np.median(pvtb_lap_prob)
    print(np.sum(pvtb_lap_prob >= np.median(pvtb_lap_prob)), "subj with high lap_prob")
    print(np.sum(pvtb_lap_prob < np.median(pvtb_lap_prob)), "subj with low lap_prob")
    # Convert
    pvtb_rrt_class = (pvtb_rrt >= thresh_rrt).astype(int)  # high: class 1, low: class 0
    pvtb_lap_prob_class = (pvtb_lap_prob >= thresh_lap_prob).astype(int)
    pvtb_lap_num_class = (pvtb_lap_num >= thresh_lap_num).astype(int)

    pvtb_rrt_new = pvtb_rrt_class[sub_id_retain]
    pvtb_lap_num_new = pvtb_lap_num_class[sub_id_retain]
    pvtb_lap_prob_new = pvtb_lap_prob_class[sub_id_retain]

    # np.save(path_save + "/pvtb_rrt_values_classification_threshold_median_for_yasa_c3_eeg_rel_power_analysis.npy",
    #         pvtb_rrt_new)  # (subj, 1) --> (936,)
    # np.save(path_save + "/pvtb_lap_num_values_classification_threshold_median_for_yasa_c3_eeg_rel_power_analysis.npy",
    #         pvtb_lap_num_new)
    # np.save(path_save + "/pvtb_lap_prob_values_classification_threshold_median_for_yasa_c3_eeg_rel_power_analysis.npy",
    #         pvtb_lap_prob_new)

    return pvtb_rrt_new, pvtb_lap_num_new, pvtb_lap_prob_new


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
num_layers = 4
output_dim = 1  # Binary classification
num_epochs = 400
batch_size = 16
learning_rate = 0.00001
bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma']  # input_dim = len(bands)
hrv_f_names = ['HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_TP']
hrv_t_names = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_CVNN', 'HRV_SDRMSSD', 'HRV_pNN20']
# feature_freq = 30  # duration of the windows used to extract features in sec
seq_len_t = 5  # number of hours to retain (from the end of the signal, after removing unreliable segments)

path_file = "/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs"
path_save = "/media/livia/Elements/public_sleep_data/stages/stages/original/yasa_eeg_powers"
# print(path_file)

# Directories of the subjects' folders
dir_subjs_power = sorted(glob2.glob(path_file + '/[!p]*/yasa_outputs/eeg_bandpowers_30sec/*'))  # because the target files that start with "pvtb" are also in
                                                         # that directory, and shouldn't be in this list
                                                         # Note: the files that start with the string after ! will not
                                                         # be considered in the glob reading
# dir_subjs = [f for f in dir_subjs if not f.split('/')[-1].startswith('w')]
dir_subjs_hrv = sorted(glob2.glob(path_file + '/[!p]*/yasa_outputs/ecg_hrv_params/*'))

dir_targets = sorted([file for file in glob2.glob(path_file + "/*/Modified_*.xlsx") if not file.endswith("subset.xlsx")])
# dir_targets = sorted(glob2.glob(path_file + "/*/usable/Modified_*.xlsx"))
# dir_target = path_file + "/pvtb_rrt_values_classification_threshold_median.pt"

# found in "test_rnn_power.py"
path_file2 = r'/media/livia/Elements/public_sleep_data/stages/stages/original/'
subj_retained_for_power_analysis = pd.read_csv(path_file2 + "yasa_eeg_powers/subjects_retained_for_power_analysis.csv", header=None)
subj_retained_for_power_analysis = subj_retained_for_power_analysis.values.flatten().tolist()

# Read the data, extract and organize rel power of 6 bands for C3-M2, remove subjs with too many nans,
# extract and organize main HRV params, remove nan rows (time-steps) (segments with unreliable R-peak counts),
# retain features of the last 5 hours of the remaining data, save the results as nparray
# ((Ran it once, no need to run again unless you want to check something))
# X_power, X_hrv_t, X_hrv_f, subj_to_rem, subj_to_ret = read_input(subj_retained_for_power_analysis, dir_subjs_hrv, bands,
#                                                                  hrv_t_names, hrv_f_names, seq_len_t, path_save)

# Read the saved organized input data
X_power = np.load(path_save+"/yasa_c3_eeg_rel_powers.npy")
X_hrv_t = np.load(path_save+"/neurokit_hrv_params_t.npy")
X_hrv_f = np.load(path_save+"/neurokit_hrv_params_f.npy")

# Read the targets and modify to only retain the subjects included in the yasa analysis and without too many nans (937)
# pvtb_rrt_new, pvtb_lap_num_new, pvtb_lap_prob_new = create_output(dir_targets, subj_retained_for_power_analysis)
# y = pvtb_lap_prob_new  # for example here the target is pvtb_lap_prob

# Read the saved organized targets
y = np.load(path_save + "/pvtb_rrt_values_classification_threshold_median_for_yasa_c3_eeg_rel_power_analysis.npy")

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
                                                                          method="per_sample")
X_hrv_t_train_norm, X_hrv_t_val_norm, X_hrv_t_test_norm = standardization(X_hrv_t_train, X_hrv_t_val, X_hrv_t_test,
                                                                          method="per_sample")
X_hrv_f_train_norm, X_hrv_f_val_norm, X_hrv_f_test_norm = standardization(X_hrv_f_train, X_hrv_f_val, X_hrv_f_test,
                                                                          method="per_sample")

# Create datasets and data loaders
train_dataset = EEGDataset(X_power_train_norm, X_hrv_t_train_norm, X_hrv_f_train_norm, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataset = EEGDataset(X_power_val_norm, X_hrv_t_val_norm, X_hrv_f_val_norm, y_val)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
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

a = 0
