import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Example model initialization
input_dims = (6, 6, 6)  # Adjust based on actual feature dimensions for each stream
d_model = 64
nhead = 8
num_layers = 2
dim_feedforward = 256
output_dim = 1  # Set to 1 for binary classification or regression; set to number of classes for multi-class classification
dropout = 0.1

# Define task type
task_type = "classification"  # Set to "classification" or "regression"

# Instantiate model
model = MultiPathTransformerClassifier(
    input_dims=input_dims,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    output_dim=output_dim,
    dropout=dropout,
    task_type=task_type
)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define loss function
if task_type == "classification":
    if output_dim == 1:
        criterion = nn.BCEWithLogitsLoss()  # Binary classification
    else:
        criterion = nn.CrossEntropyLoss()  # Multi-class classification
elif task_type == "regression":
    criterion = nn.MSELoss()  # Mean Squared Error for regression
else:
    raise ValueError("task_type should be 'classification' or 'regression'")

# Training loop
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            x_time_hrv, x_freq_hrv, x_power, labels = batch
            x_time_hrv = x_time_hrv.to(device)
            x_freq_hrv = x_freq_hrv.to(device)
            x_power = x_power.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(x_time_hrv, x_freq_hrv, x_power)

            # Calculate loss
            if task_type == "classification":
                if output_dim == 1:
                    loss = criterion(outputs, labels.float().unsqueeze(1))  # Binary classification
                    preds = torch.round(torch.sigmoid(outputs)).squeeze(1)  # Sigmoid + threshold at 0.5
                else:
                    loss = criterion(outputs, labels.long())  # Multi-class classification
                    preds = torch.argmax(outputs, dim=1)
            elif task_type == "regression":
                loss = criterion(outputs.squeeze(), labels.float())  # Regression
                preds = outputs.squeeze()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            if task_type == "classification":
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        train_loss /= len(train_loader.dataset)
        if task_type == "classification":
            train_accuracy = 100 * train_correct / train_total
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

        # Validation step (optional)
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                x_time_hrv, x_freq_hrv, x_power, labels = batch
                x_time_hrv = x_time_hrv.to(device)
                x_freq_hrv = x_freq_hrv.to(device)
                x_power = x_power.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(x_time_hrv, x_freq_hrv, x_power)

                # Calculate validation loss
                if task_type == "classification":
                    if output_dim == 1:
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

        # Calculate average validation loss and accuracy for the epoch
        val_loss /= len(val_loader.dataset)
        if task_type == "classification":
            val_accuracy = 100 * val_correct / val_total
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        else:
            print(f"Validation Loss: {val_loss:.4f}")

