import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import time
# -------------------------
# 1. Load and preprocess data


# Load Digits dataset
digits = load_digits()
X = digits.data  # 64 features (8x8 image pixels)
y = digits.target  # 10 classes (digits 0-9)


# -------------------------
#Lad MNIST dataset

# Load MNIST dataset from OpenML
# mnist = fetch_openml('mnist_784')



# # Convert to numpy arrays
# X = np.array(mnist.data)  # 784 features (28x28 image pixels flattened)
# y = np.array(mnist.target)  # 10 classes (digits 0-9)

# # Optionally, convert `y` to integers if it's loaded as strings
# y = y.astype(int)


# -------------------------
# 2. Create a two-layer network
# -------------------------
class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # First layer: z_1 = W_1 x + b_1
        z1 = self.fc1(x)
        # Activation: h_1 = ReLU(z_1)
        h1 = self.relu(z1)
        # Second layer: z_2 = W_2 h_1 + b_2
        z2 = self.fc2(h1)
        return z2  # We will apply nn.CrossEntropyLoss, which expects logits

# Hyperparameters
input_dim = X.shape[1]  # features
hidden_dim = 128               # you can choose any reasonable size
output_dim = 10                # 

model = TwoLayerNet(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
N_exp = 10
num_epochs = 100

batch_size = 64
learning_rate = 1e-2

all_history = {
    'loss': np.zeros((N_exp, num_epochs)),
    'train_acc': np.zeros((N_exp, num_epochs)),
    'val_acc': np.zeros((N_exp, num_epochs)),
    'test_acc': np.zeros((N_exp, num_epochs)),
    'time': np.zeros((N_exp, num_epochs))
}

for e in range(N_exp):
    # Split data
    # First, split the dataset into train+valid and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Then, split the train+valid set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )


    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    X_val_torch = torch.tensor(X_val, dtype=torch.float32)
    y_val_torch = torch.tensor(y_val, dtype=torch.long)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.long)

    history = {
        'loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': [],
        'time': []
    }
    t0 = time.time()
    
    for epoch in range(num_epochs):
        # Shuffle training data (optional but common)
        perm = torch.randperm(X_train_torch.size(0))
        X_train_torch = X_train_torch[perm]
        y_train_torch = y_train_torch[perm]

        # Mini-batch updates
        for i in range(0, X_train_torch.size(0), batch_size):
            x_batch = X_train_torch[i:i+batch_size]
            y_batch = y_train_torch[i:i+batch_size]
            
            # Forward pass
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            # Backpropagation
            model.zero_grad()
            loss.backward()

            # Manual parameter update for each parameter in the model
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad

        # Evaluate on training and validation sets
        with torch.no_grad():
            # Training accuracy
            train_logits = model(X_train_torch)
            _, train_preds = torch.max(train_logits, 1)
            train_accuracy = (train_preds == y_train_torch).float().mean()

            # Validation accuracy
            val_logits = model(X_val_torch)
            _, val_preds = torch.max(val_logits, 1)
            val_accuracy = (val_preds == y_val_torch).float().mean()

            test_logits = model(X_test_torch)
            _, test_preds = torch.max(test_logits, 1)
            test_accuracy = (test_preds == y_test_torch).float().mean()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Loss: {loss.item():.4f}, "
            f"Train Acc: {train_accuracy.item():.4f}, "
            f"Val Acc: {val_accuracy.item():.4f}", 
            f"Test Acc: {test_accuracy.item():.4f}")




        # save the history of the training process
        history['loss'].append(loss.item())
        history['train_acc'].append(train_accuracy.item())
        history['val_acc'].append(val_accuracy.item())
        history['test_acc'].append(test_accuracy.item())
        history['time'].append(time.time()-t0)


    all_history['loss'][e] = history['loss']
    all_history['train_acc'][e] = history['train_acc']
    all_history['val_acc'][e] = history['val_acc']
    all_history['test_acc'][e] = history['test_acc']
    all_history['time'][e] = history['time']
    
# Flatten the data for CSV
flat_data = []
for metric_name, values in all_history.items():
    for e, row in enumerate(values):
        flat_data.append({"metric": metric_name, "index": e, "values": row})

# Convert to DataFrame and save as CSV
df = pd.DataFrame(flat_data)
df.to_csv("baseline_history.csv", index=False)




