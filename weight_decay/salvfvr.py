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


N_exp = 10
num_epochs = 50

all_history = {
    'loss_outer': np.zeros((N_exp, num_epochs)),
    'train_acc_outer': np.zeros((N_exp, num_epochs)),
    'val_acc_outer': np.zeros((N_exp, num_epochs)),
    'test_acc_outer': np.zeros((N_exp, num_epochs)),
    'loss_inner': np.zeros((N_exp, num_epochs)),
    'train_acc_inner': np.zeros((N_exp, num_epochs)),
    'val_acc_inner': np.zeros((N_exp, num_epochs)),
    'test_acc_inner': np.zeros((N_exp, num_epochs)),
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


    
    model_inner = TwoLayerNet(input_dim, hidden_dim, output_dim)
    model_outer = TwoLayerNet(input_dim, hidden_dim, output_dim)
    model_outer_last = TwoLayerNet(input_dim, hidden_dim, output_dim)

    # -------------------------
    # 3. Define loss and optimizer
    # -------------------------
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # 4. Training loop
    # -------------------------
    batch_size = 64

    gamma1 = 1
    gamma2 = 0.01
    C = torch.ones(1).requires_grad_(True)
    C.data.fill_(20.0)
    eta = 0.1
    rho = 0.001
    c1 = 1
    c2 = 0.2
    alpha = 0.1
    beta = 0.9
    vr = True

    z = torch.tensor(0.0, requires_grad=True)
    lamda = torch.tensor(1.0, requires_grad=True)


    history = {
        'loss_outer': [],
        'train_acc_outer': [],
        'val_acc_outer': [],
        'test_acc_outer': [],
        'loss_inner': [],
        'train_acc_inner': [],
        'val_acc_inner': [],
        'test_acc_inner': [],
        'time': []
    }
    # Simple mini-batch training

    dparam_last = {}
    dparam = {}
    t0 = time.time()

    for epoch in range(num_epochs):
        # Shuffle training data (optional but common)
        perm = torch.randperm(X_train_torch.size(0))
        X_train_torch = X_train_torch[perm]
        y_train_torch = y_train_torch[perm]


        # Mini-batch updates
        C.requires_grad = False
        z.requires_grad = False
        lamda.requires_grad = True
        model_inner.requires_grad = True
        model_outer.requires_grad = False
        model_outer_last.requires_grad = False
        with torch.no_grad():
            lamda.fill_(0)  # Reset lamda to 1.0 at the beginning of each epoch
        for i in range(0, X_train_torch.size(0), batch_size):
            x_batch = X_train_torch[i:i+batch_size]
            y_batch = y_train_torch[i:i+batch_size]
            
            # Forward pass
            logits = model_inner(x_batch)
            loss_inner = criterion(logits, y_batch)

            constraint = sum(p.pow(2).sum() for p in model_inner.parameters()) - C ** 2

            loss_inner = loss_inner + gamma1 * torch.maximum( lamda + constraint / gamma1, torch.tensor(0., requires_grad=True)) ** 2 - gamma1 * lamda ** 2 - 0.5 * gamma2 * (lamda - z) ** 2
            
            # Backpropagation
            model_inner.zero_grad()
            loss_inner.backward()

            # Manual parameter update for each parameter in the model
            learning_rate = eta 
            with torch.no_grad():
                for param in model_inner.parameters():
                    param -= learning_rate * param.grad

                # Update lamda in place
                lamda += rho * lamda.grad

                # Print diagnostics
                # print("lamda: ", lamda.item(), 
                    # "constraint violation: ", torch.maximum(constraint, torch.tensor(0.0)).item(), 
                    # "lamda.grad: ", lamda.grad)
                

                lamda.grad.zero_()  # Reset gradients for lamda

                # Enforce lower bound for lamda
                lamda.clamp_(min=1e-4)  # In-place operation to ensure lamda >= 1e-4


        if epoch == 0:
            model_outer.load_state_dict(model_inner.state_dict())


        # Evaluate on validation sets
        # Mini-batch updates
        C.requires_grad = True
        z.requires_grad = True
        lamda.requires_grad = False
        model_inner.requires_grad = False
        model_outer.requires_grad = True
        model_outer_last.requires_grad = False
        for i in range(0, X_val_torch.size(0), batch_size):
            x_val_batch = X_val_torch[i:i+batch_size]
            y_val_batch = y_val_torch[i:i+batch_size]
            x_train_batch = X_train_torch[i:i+batch_size]
            y_train_batch = y_train_torch[i:i+batch_size]

            # Forward pass
            logits_val = model_outer(x_val_batch)
            logits_train = model_outer(x_train_batch)
            constraint_outer = sum(p.pow(2).sum() for p in model_outer.parameters()) - C ** 2
            constraint_inner = sum(p.pow(2).sum() for p in model_inner.parameters()) - C ** 2
            loss_outer = (  criterion(logits_val, y_val_batch) +  
                c1 * criterion(logits_val, y_val_batch) - c1 * criterion(logits_train, y_train_batch) - 0.5 * c1 * gamma1 * torch.maximum(lamda + constraint_inner / gamma1, torch.tensor(0., requires_grad=True)) ** 2 - 0.5 * gamma1 * lamda ** 2 + 0.5 * c1 * gamma2 * (lamda - z) ** 2 + 
                0.5 * c2 * torch.maximum(constraint_outer, torch.tensor(0.0, requires_grad=True)) ** 2
            )
            
            # Backpropagation
            model_outer.zero_grad()
            loss_outer.backward(retain_graph=True)
               # print("C.grad: ", C.grad)
            with torch.no_grad():
                for name, param in model_outer.named_parameters():
                    dparam[name] = param.grad
                dC = C.grad
                dz = z.grad
                


            if epoch + i > 0:
                C_last.requires_grad = False
                model_outer.requires_grad = False
                model_inner.requires_grad = False
                model_outer_last.requires_grad = True
                constraint_outer_last = sum(p.pow(2).sum() for p in model_outer_last.parameters()) - C_last ** 2
                constraint_inner_last = sum(p.pow(2).sum() for p in model_inner.parameters()) - C_last ** 2
                loss_outer_last = (  criterion(logits_val, y_val_batch) +  
                    c1 * criterion(logits_val, y_val_batch) - c1 * criterion(logits_train, y_train_batch) - 0.5 * c1 * gamma1 * torch.maximum(lamda + constraint_inner_last / gamma1, torch.tensor(0., requires_grad=True)) ** 2 - 0.5 * gamma1 * lamda ** 2 + 0.5 * c1 * gamma2 * (lamda - z_last) ** 2 + 
                    0.5 * c2 * torch.maximum(constraint_outer_last, torch.tensor(0.0, requires_grad=True)) ** 2
                )
                model_outer_last.zero_grad()
                loss_outer_last.backward()
                # print("C.grad: ", C.grad)
                with torch.no_grad():
                    for name, param in model_outer_last.named_parameters():
                        dparam[name] += (1-beta) * (dparam_last[name] - param.grad)
                    dC += (1-beta) * (dC_last - C.grad)
                    dz += (1-beta) * (dz_last - z.grad)


            # print("C.grad: ", C.grad)
            with torch.no_grad():
                model_outer_last.load_state_dict(model_outer.state_dict())
                for name, param in model_outer.named_parameters():
                    # print(name, dparam[name].shape)
                    param -= alpha * dparam[name]
                    dparam_last[name] = dparam[name]
                C_last = C.clone().requires_grad_(False)
                z_last = z.clone().requires_grad_(False)
                C -= alpha * dC
                C.grad.zero_()
                if z.grad is not None:
                    z -= 1e-3 * dz
                    z.grad.zero_()

                C.clamp_(min=1e-4)
                z.clamp_(min=1e-4)
                dC_last = dC.clone()
                dz_last = dz.clone()

        # Evaluate on test sets using both model_outer and model_inner
        with torch.no_grad():
            # model_outer
            # Training accuracy
            train_logits = model_outer(X_train_torch)
            _, train_preds = torch.max(train_logits, 1)
            train_accuracy = (train_preds == y_train_torch).float().mean()

            # Validation accuracy
            val_logits = model_outer(X_val_torch)
            _, val_preds = torch.max(val_logits, 1)
            val_accuracy = (val_preds == y_val_torch).float().mean()

            # Test accuracy
            test_logits = model_outer(X_test_torch)
            _, test_preds = torch.max(test_logits, 1)
            test_accuracy = (test_preds == y_test_torch).float().mean()

            # model_inner
            # Training accuracy
            train_logits_inner = model_inner(X_train_torch)
            _, train_preds_inner = torch.max(train_logits_inner, 1)
            train_accuracy_inner = (train_preds_inner == y_train_torch).float().mean()

            # Validation accuracy
            val_logits_inner = model_inner(X_val_torch)
            _, val_preds_inner = torch.max(val_logits_inner, 1)
            val_accuracy_inner = (val_preds_inner == y_val_torch).float().mean()

            # Test accuracy
            test_logits_inner = model_inner(X_test_torch)
            _, test_preds_inner = torch.max(test_logits_inner, 1)
            test_accuracy_inner = (test_preds_inner == y_test_torch).float().mean()




        # print(f"Epoch [{epoch+1}/{num_epochs}], "
        #     f"model_outer: "
        #         f"Loss: {loss_outer.item():.4f}, "
        #         f"Train Acc: {train_accuracy.item():.4f}, "
        #         f"Val Acc: {val_accuracy.item():.4f}, "
        #         f"Test Acc: {test_accuracy.item():.4f}\n"
        #         f"model_inner: "
        #         f"Loss: {loss_inner.item():.4f}, "
        #         f"Train Acc: {train_accuracy_inner.item():.4f}, "
        #         f"Val Acc: {val_accuracy_inner.item():.4f}, "
        #         f"Test Acc: {test_accuracy_inner.item():.4f}"
        #         )


        # save the history of the training process
        history['loss_outer'].append(loss_outer.item())
        history['train_acc_outer'].append(train_accuracy.item())
        history['val_acc_outer'].append(val_accuracy.item())
        history['test_acc_outer'].append(test_accuracy.item())
        history['loss_inner'].append(loss_inner.item())
        history['train_acc_inner'].append(train_accuracy_inner.item())
        history['val_acc_inner'].append(val_accuracy_inner.item())
        history['test_acc_inner'].append(test_accuracy_inner.item())
        history['time'].append(time.time()-t0)

    print("C: ", C.item())

    all_history['loss_outer'][e, :] = history['loss_outer']
    all_history['train_acc_outer'][e, :] = history['train_acc_outer']
    all_history['val_acc_outer'][e, :] = history['val_acc_outer']
    all_history['test_acc_outer'][e, :] = history['test_acc_outer']
    all_history['loss_inner'][e, :] = history['loss_inner']
    all_history['train_acc_inner'][e, :] = history['train_acc_inner']
    all_history['val_acc_inner'][e, :] = history['val_acc_inner']
    all_history['test_acc_inner'][e, :] = history['test_acc_inner']
    all_history['time'][e, :] = history['time']
    
# Flatten the data for CSV
flat_data = []
for metric_name, values in all_history.items():
    for e, row in enumerate(values):
        flat_data.append({"metric": metric_name, "index": e, "values": row})

# Convert to DataFrame and save as CSV
df = pd.DataFrame(flat_data)
df.to_csv("SALVFVR_history.csv", index=False)




