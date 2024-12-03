import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim
import torch.nn as nn
import re
import random as random2

# Note: update it to accept both classification and regression tasks (get help from "transformer_power_hrv_features.py")


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

        # # Take the last hidden state from each LSTM and apply batch normalization
        # hn_power = self.bn_power(hn_power[-1])  # (batch_size, hidden_dim)
        # hn_hrv_t = self.bn_hrv_t(hn_hrv_t[-1])  # (batch_size, hidden_dim)
        # hn_hrv_f = self.bn_hrv_f(hn_hrv_f[-1])  # (batch_size, hidden_dim)

        # Take the last hidden state from each LSTM and conditionally apply batch normalization
        if hn_power.size(1) > 1:  # Check if batch size > 1
            hn_power = self.bn_power(hn_power[-1])  # (batch_size, hidden_dim)
        else:
            hn_power = hn_power[-1]  # Skip BatchNorm for single-sample batch

        if hn_hrv_t.size(1) > 1:
            hn_hrv_t = self.bn_hrv_t(hn_hrv_t[-1])
        else:
            hn_hrv_t = hn_hrv_t[-1]

        if hn_hrv_f.size(1) > 1:
            hn_hrv_f = self.bn_hrv_f(hn_hrv_f[-1])
        else:
            hn_hrv_f = hn_hrv_f[-1]

        # Concatenate the hidden states from all LSTM paths
        combined_features = torch.cat((hn_power, hn_hrv_t, hn_hrv_f), dim=1)  # (batch_size, hidden_dim * 3)

        # Apply dropout before the fully connected layer
        combined_features = self.fc_dropout(combined_features)

        # Pass through the fully connected layer
        out = self.fc(combined_features)

        # Output without sigmoid to be used with BCEWithLogitsLoss for better numerical stability
        return out


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


def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, output_dir, task):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
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

    # plt.tight_layout()
    # plt.show()
    # Save the plot
    learning_curve_path = os.path.join(output_dir, "learning_curves_task_" + task + ".png")
    plt.tight_layout()
    plt.savefig(learning_curve_path)
    plt.close()

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


def train_model_multiple_tasks(input_data, names_input, target_file, model_class, output_dir, device, num_epochs=300,
                               batch_size_train=8, batch_size_val=128, batch_size_test=1,
                               hidden_dim=128, num_layers=3, learning_rate=0.00005):
    """
    Train the model for each column in the target file.

    Args:
    - input_data: A tuple (X_power, X_hrv_t, X_hrv_f) of input feature arrays.
    - target_file: Path to the CSV file containing multiple targets.
    - model_class: The model class to use for training.
    - output_dir: Directory to save results for each task.
    - device: PyTorch device ('cuda' or 'cpu').
    - num_epochs: Number of epochs for training.
    - batch_size: Batch size for DataLoader.
    - learning_rate: Learning rate for optimizer.
    """
    # Read the target file
    targets = pd.read_csv(target_file)
    tasks = targets.columns.tolist()[1:]  # Exclude the 'test_sessions.subid' column
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

        # Filter input data for valid subjects
        X_power, X_hrv_t, X_hrv_f = input_data
        X_power_task = X_power[valid_indices]
        X_hrv_t_task = X_hrv_t[valid_indices]
        X_hrv_f_task = X_hrv_f[valid_indices]

        # # Split data into train/val sets (80% train, 20% val)
        # num_samples = len(y_task)
        # train_size = int(0.8 * num_samples)
        # indices = np.arange(num_samples)
        # np.random.shuffle(indices)
        # train_indices = indices[:train_size]
        # val_indices = indices[train_size:]

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

        X_power_train, X_power_val, X_power_test = X_power_task[train_subj], X_power_task[val_subj], X_power_task[test_subj]
        X_hrv_t_train, X_hrv_t_val, X_hrv_t_test = X_hrv_t_task[train_subj], X_hrv_t_task[val_subj], X_hrv_t_task[test_subj]
        X_hrv_f_train, X_hrv_f_val, X_hrv_f_test = X_hrv_f_task[train_subj], X_hrv_f_task[val_subj], X_hrv_f_task[test_subj]
        y_train, y_val, y_test = y_task[train_subj], y_task[val_subj], y_task[test_subj]

        # Standardize the data
        X_power_train, X_power_val, X_power_test = standardization(X_power_train, X_power_val, X_power_test, method='all_timesteps')
        X_hrv_t_train, X_hrv_t_val, X_hrv_t_test = standardization(X_hrv_t_train, X_hrv_t_val, X_hrv_t_test, method='all_timesteps')
        X_hrv_f_train, X_hrv_f_val, X_hrv_f_test = standardization(X_hrv_f_train, X_hrv_f_val, X_hrv_f_test, method='all_timesteps')

        # Create PyTorch datasets and loaders
        train_dataset = EEGDataset(X_power_train, X_hrv_t_train, X_hrv_f_train, y_train)
        val_dataset = EEGDataset(X_power_val, X_hrv_t_val, X_hrv_f_val, y_val)
        test_dataset = EEGDataset(X_power_test, X_hrv_t_test, X_hrv_f_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

        # Initialize model, optimizer, and loss function
        input_dims = (X_power_train.shape[-1], X_hrv_t_train.shape[-1], X_hrv_f_train.shape[-1])
        model = model_class(input_dims=input_dims, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Save results
        task_output_dir = os.path.join(output_dir, task)
        os.makedirs(task_output_dir, exist_ok=True)

        # Train and validate the model
        train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate_model(
            model, train_loader, val_loader, criterion, optimizer, num_epochs, device, task_output_dir, task
        )

        # Save metrics
        metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }
        pd.DataFrame(metrics).to_csv(os.path.join(task_output_dir, "metrics.csv"), index=False)

        # Save learning curves
        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # plt.plot(train_losses, label='Train Loss')
        # plt.plot(val_losses, label='Val Loss')
        # plt.title(f"Loss for {task}")
        # plt.legend()
        # plt.subplot(1, 2, 2)
        # plt.plot(train_accuracies, label='Train Accuracy')
        # plt.plot(val_accuracies, label='Val Accuracy')
        # plt.title(f"Accuracy for {task}")
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(task_output_dir, "learning_curves.png"))
        # plt.close()

    print(f"Training completed for all tasks. Results saved in {output_dir}.")
    a = 0


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


path_file = "/media/livia/Elements/public_sleep_data/stages/stages/original/STAGES_PSGs"
path_save = "/media/livia/Elements/public_sleep_data/stages/stages/original/yasa_eeg_powers"
dir_targets = path_save + "/all_scores_classification_for_yasa_c3_eeg_rel_power_analysis.csv"

X_power = np.load(path_save + "/yasa_c3_eeg_rel_powers.npy")
X_hrv_t = np.load(path_save + "/neurokit_hrv_params_t.npy")
X_hrv_f = np.load(path_save + "/neurokit_hrv_params_f.npy")

# found in "test_rnn_power.py"
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

output_dir = "/home/livia/PycharmProjects/YASA_proj/results/comparison1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model_multiple_tasks(
    input_data=(X_power, X_hrv_t, X_hrv_f),
    names_input=names_input,
    target_file=dir_targets,
    model_class=MultiPathLSTMClassifier,
    output_dir=output_dir,
    device=device,
    num_epochs=800,  # Reduced epochs for faster testing
    batch_size_train=8,
    batch_size_val=128,
    batch_size_test=1,
    learning_rate=0.00003,
    hidden_dim=128,
    num_layers=3
)

a = 0

