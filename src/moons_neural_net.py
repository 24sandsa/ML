import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

X_crescent, y_crescent = make_moons(n_samples=1000,
                                     noise=0.2,
                                    random_state=42)

X_tensor = torch.from_numpy(X_crescent).type(torch.float)
y_tensor = torch.from_numpy(y_crescent).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=RANDOM_SEED)


class Crescent_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1)
        )
    def forward(self, x):
        return self.layer_stack(x)

luna = Crescent_Model()

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

epochs = 5000
learning_rate = 0.03
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(luna.parameters(), lr=learning_rate)

current_epoch = []
live_train_loss = []
live_train_acc = []
live_test_loss = []
live_test_acc = []

for epoch in range(epochs+1):
    current_epoch.append(epoch)
    luna.train()
    train_logits = luna(X_train).squeeze()
    train_sigmoids = torch.round(torch.sigmoid(train_logits))
    train_loss = loss_fn(train_logits, y_train)
    live_train_loss.append(train_loss.detach().numpy())
    train_acc = accuracy(y_train, train_sigmoids)
    live_train_acc.append(train_acc)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    luna.eval()
    with torch.inference_mode():
        test_logits = luna(X_test).squeeze()
        test_sigmoids = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        live_test_loss.append(test_loss.detach().numpy())
        test_acc = accuracy(y_test, test_sigmoids)
        live_test_acc.append(test_acc)

    print(f'Epoch {epoch} \n TR_BCELoss: {train_loss} TR_Acc: {train_acc} \n TE_BCELoss {train_loss}, TE_Acc {test_acc}')

plt.subplot(2, 2, 1)
plt.title('Train Decision Map')
plot_decision_boundary(luna, X_train, y_train)
plt.subplot(2, 2, 2)
plt.title('Test Decision Map')
plot_decision_boundary(luna, X_test, y_test)
plt.subplot(2, 2, 3)
plt.title('Binary Cross Entropy Loss Over Epoch')
plt.plot(current_epoch, live_train_loss, c='r', label='Train')
plt.plot(current_epoch, live_test_loss, c='b', label='Test')
plt.legend()
plt.subplot(2, 2, 4)
plt.title('Accuracy Metric Over Epoch')
plt.plot(current_epoch, live_train_acc, c='r', label='Train')
plt.plot(current_epoch, live_test_acc, c='b', label='Test')
plt.legend()
plt.show()
plt.show()