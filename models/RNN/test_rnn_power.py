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


class EEGDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (numpy array or torch tensor): EEG PSD features of shape (num_samples, seq_len, feature_dim)
            y (numpy array or torch tensor): Labels of shape (num_samples,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert to PyTorch tensor
        self.y = torch.tensor(y, dtype=torch.float32)  # Binary classification labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM output: (batch_size, seq_length, hidden_dim)
        _, (hn, _) = self.lstm(x)
        hn = hn[-1]  # Take the last layer's hidden state (many-to-one classification)
        out = self.fc(hn)
        return self.sigmoid(out)


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate running loss
            running_loss += loss.item() * inputs.size(0)  # Multiply by batch size to get total loss for the batch

            # Calculate predictions and accuracy
            predicted = (outputs >= 0.5).float()  # Threshold at 0.5 for binary classification
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Calculate average loss for the epoch
        epoch_loss = running_loss / total

        # Calculate accuracy for the epoch
        accuracy = 100 * correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")


# def train_model(model, train_loader, criterion, optimizer, num_epochs):
#     model.train()
#     for epoch in range(num_epochs):
#         correct = 0
#         total = 0
#         running_loss = 0.0
#
#         for batch_index, (inputs, labels) in enumerate(train_loader):
#             inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
#
#             # Forward pass
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#
#             # Check gradients for each parameter
#             for name, param in model.named_parameters():
#                 if param.requires_grad:
#                     if param.grad is not None:
#                         grad_norm = param.grad.norm().item()
#                         print(f"Gradient norm for {name} in batch {batch_index + 1}: {grad_norm:.4f}")
#                     else:
#                         print(f"No gradient computed for {name} in batch {batch_index + 1}")
#
#             # Perform optimization step
#             optimizer.step()
#
#             # Calculate running loss
#             running_loss += loss.item() * inputs.size(0)
#
#             # Calculate predictions and accuracy
#             predicted = (outputs >= 0.5).float()
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)
#
#         # Calculate average loss for the epoch
#         epoch_loss = running_loss / total
#         accuracy = 100 * correct / total
#
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")


def evaluate_model(model, val_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')
    return accuracy


def read_input(dir_subjs, bands, feature_freq, seq_len_t, path_save):

    all_subj = []
    subjects_with_too_many_nans = []
    subjects_to_retain = []
    for num in range(len(dir_subjs)):
        # num = num + 53

        if num % 100 == 0:
            print(f'subject {num} from {len(dir_subjs)}...')
        all_bands = []
        for band in bands:
            band_data = pd.read_csv(os.path.join(dir_subjs[num], band+"_rel_power_all_eeg_channels_30sec.csv"))

            # Filter the DataFrame to get the row where column 0 matches "C3-M2"
            # Assuming "C3-M2" is in the first column, which is accessible using band_data.iloc[:, 0]
            row_data = band_data[band_data.iloc[:, 0] == "C3-M2"]
            if not row_data.empty:
                # Extract all values except the first column (which contains the label)
                all_bands.append(row_data.iloc[:, 1:].to_numpy())
        all_bands = np.vstack(all_bands)

        # Retain the last five hours
        duration = seq_len_t*60*60  # duration of the signal to retain
        seq_len = int(duration/feature_freq)
        # Cut or zero-pad
        if all_bands.shape[1] < seq_len:
            # Pad the array along the second axis (columns) to have a shape of (6, 300)
            all_bands = np.pad(all_bands, ((0, 0), (0, seq_len-all_bands.shape[1])), mode='constant', constant_values=0)
        else:
            all_bands = all_bands[:, -seq_len:]

        # Check if there are too many nans (original signal has been mostly zero)
        num_nans = np.isnan(all_bands[0, :]).sum()
        threshold = len(all_bands[0, :]) / 10  # Calculate 1/10th of the data length
        if num_nans > threshold:  # Check if the number of NaNs exceeds the threshold
            subjects_with_too_many_nans.append(dir_subjs[num])
            print(f"{dir_subjs[num]} has too many NaNs: {num_nans} out of {len(all_bands[0, :])}")
        else:
            all_bands = np.nan_to_num(all_bands, nan=0.0)  # Replace NaNs with 0 if the count is within acceptable limit
            # subjects_data[subject_name] = data  # Save the updated data back
            all_subj.append(all_bands)
            subjects_to_retain.append(dir_subjs[num])

    all_subj = np.array(all_subj)

    # save the data as one file
    np.save(path_save+"/yasa_c3_eeg_rel_powers.npy", all_subj)  # (subj, band, seq_len), rel power of C3-M2 in 6 bands
    df1 = pd.DataFrame(subjects_to_retain)
    df1.to_csv(path_save + '/subjects_retained_for_power_analysis.csv', index=False, header=False)

    return all_subj, subjects_with_too_many_nans, subjects_to_retain


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

    np.save(path_save + "/pvtb_rrt_values_classification_threshold_median_for_yasa_c3_eeg_rel_power_analysis.npy",
            pvtb_rrt_new)  # (subj, 1) --> (936,)
    np.save(path_save + "/pvtb_lap_num_values_classification_threshold_median_for_yasa_c3_eeg_rel_power_analysis.npy",
            pvtb_lap_num_new)
    np.save(path_save + "/pvtb_lap_prob_values_classification_threshold_median_for_yasa_c3_eeg_rel_power_analysis.npy",
            pvtb_lap_prob_new)

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
            X_standardized = np.zeros_like(X)

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
        X_train_standardized = standardize_per_sample(X_train)
        X_val_standardized = standardize_per_sample(X_val)
        X_test_standardized = standardize_per_sample(X_test)

    return X_train_standardized, X_val_standardized, X_test_standardized


# 5. Hyperparameters and Data Preparation
input_dim = 6  # Number of PSD features per 30-second segment (rel power of 6 frequency bands)
hidden_dim = 32
num_layers = 3
output_dim = 1  # Binary classification
num_epochs = 20
batch_size = 16
learning_rate = 0.0001
bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma']  # input_dim = len(bands)
feature_freq = 30  # duration of the windows used to extract features in sec
seq_len_t = 5  # number of hours to retain (from the end of the signal)

path_file = "/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs"
path_save = "/media/livia/Elements/public_sleep_data/stages/stages/original/yasa_eeg_powers"
# print(path_file)

# Directories of the subjects' folders
dir_subjs = sorted(glob2.glob(path_file + '/[!p]*/yasa_outputs/eeg_bandpowers_30sec/*'))  # because the target files that start with "pvtb" are also in
                                                         # that directory, and shouldn't be in this list
                                                         # Note: the files that start with the string after ! will not
                                                         # be considered in the glob reading
# dir_subjs = [f for f in dir_subjs if not f.split('/')[-1].startswith('w')]

dir_targets = sorted([file for file in glob2.glob(path_file + "/*/Modified_*.xlsx") if not file.endswith("subset.xlsx")])
# dir_targets = sorted(glob2.glob(path_file + "/*/usable/Modified_*.xlsx"))
# dir_target = path_file + "/pvtb_rrt_values_classification_threshold_median.pt"

# Read the data, extract and organize rel power of 6 bands for C3-M2, remove subjs with too many nans, save the results as one array
# Ran it once, no need to run again unless you want to check something
X, subj_to_rem, subj_to_ret = read_input(dir_subjs, bands, feature_freq, seq_len_t, path_save)

# Read the saved organized input data
X = np.load(path_save+"/yasa_c3_eeg_rel_powers.npy")
X = X.transpose((0, 2, 1))

# Read the targets and modify to only retain the subjects included in the yasa analysis and without too many nans (937)
# pvtb_rrt_new, pvtb_lap_num_new, pvtb_lap_prob_new = create_output(dir_targets, subj_to_ret)
# y = pvtb_lap_prob_new  # for example here the target is pvtb_lap_prob

# Read the saved organized targets
y = np.load(path_save + "/pvtb_lap_prob_values_classification_threshold_median_for_yasa_c3_eeg_rel_power_analysis.npy")

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
X_train = X[train_subj, :, :]
y_train = y[train_subj, ]
X_val = X[val_subj, :, :]
y_val = y[val_subj, ]
X_test = X[test_subj, :, :]
y_test = y[test_subj, ]

# Standardize using training set statistics
X_train_norm, X_val_norm, X_test_norm = standardization(X_train, X_val, X_test, method="per_sample")

# Create datasets and data loaders
train_dataset = EEGDataset(X_train_norm, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = EEGDataset(X_val_norm, y_val)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_dataset = EEGDataset(X_test_norm, y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.0001
model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. Train and Evaluate the Model
train_model(model, train_loader, criterion, optimizer, num_epochs)

evaluate_model(model, val_loader)


