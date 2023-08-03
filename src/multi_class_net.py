import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Setting up hyperparams
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# Data creation
X_blobs, y_blobs = make_blobs(n_samples= 1000, 
                              n_features= NUM_FEATURES,
                              centers=NUM_CLASSES,
                              cluster_std=1.5,
                              random_state=RANDOM_SEED)


X_blob_tensor = torch.from_numpy(X_blobs).type(torch.float)
y_blob_tensor = torch.from_numpy(y_blobs).type(torch.long)

# Splits
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob_tensor, y_blob_tensor, test_size=0.2)

device = 'cpu'

class Multi_Class_Neural_Network(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.layer_stack(x)

model_01 = Multi_Class_Neural_Network(input_features=NUM_FEATURES, 
                                      output_features=NUM_CLASSES, 
                                      hidden_units=8).to(device)

def accuracy(y_true, y_predictions):
    correct = torch.eq(y_true, y_predictions).sum().item()
    acc = (correct/len(y_predictions)) * 100
    return acc

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(
        np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.03
epochs = 1000
optimizer = torch.optim.Adam(model_01.parameters(), lr=learning_rate)

X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

current_epoch = []
live_train_loss = []
live_train_acc = []
live_test_loss = []
live_test_acc = []

for epoch in range(epochs + 1):
    current_epoch.append(epoch)
    model_01.train()
    y_logits = model_01(X_blob_train)
    y_softmax = torch.softmax(y_logits, dim=1).argmax(dim=1)
    loss = loss_fn(y_logits, y_blob_train)
    live_train_loss.append(loss.detach().numpy())
    acc =  accuracy(y_blob_train, y_softmax)
    live_train_acc.append(acc)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_01.eval()
    with torch.inference_mode():
        test_logits = model_01.forward(X_blob_test)
        test_softmax = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_blob_test)
        live_test_loss.append(test_loss.detach().numpy())
        test_acc = accuracy(y_blob_test, test_softmax)
        live_test_acc.append(test_acc)
    
    print(f'Epoch {epoch} \n TR_CELoss: {loss} TR_Acc: {acc} \n TE_CELoss: {test_loss} TE_Acc: {test_acc}')


plt.subplot(2, 2, 1)
plt.title('Train Decision Map')
plot_decision_boundary(model_01, X_blob_train, y_blob_train)
plt.subplot(2, 2, 2)
plt.title('Test Decision Map')
plot_decision_boundary(model_01, X_blob_test, y_blob_test)
plt.subplot(2, 2, 3)
plt.title('Cross Entropy Loss Over Epoch')
plt.plot(current_epoch, live_train_loss, c='r', label='Train')
plt.plot(current_epoch, live_test_loss, c='b', label='Test')
plt.legend()
plt.subplot(2, 2, 4)
plt.title('Accuracy Metric Over Epoch')
plt.plot(current_epoch, live_train_acc, c='r', label='Train')
plt.plot(current_epoch, live_test_acc, c='b', label='Test')
plt.legend()
plt.show()