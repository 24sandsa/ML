import torch
from torch import nn
from torchvision import datasets
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from pathlib import Path

# Generate data from sklearn
X_gen, y_gen = datasets.make_regression(n_samples=300, n_features=1, noise=10, random_state=1)

# Convert numpy arrays into tensors
# Tensors are just matrices that can have any number of dims
X = torch.from_numpy(X_gen.astype(np.float32)).squeeze(dim=1)
Y = torch.from_numpy(y_gen.astype(np.float32))

train_test_split = int(len(X) * 0.8)

X_train = X[:train_test_split]
X_test = X[train_test_split:]
y_train = Y[:train_test_split]
y_test = Y[train_test_split:]

# y = mx + b
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

model_0 = LinearRegression()
loss_fn = nn.MSELoss()
learning_rate = 2000
epochs = 2000
optimizer = torch.optim.Adam(model_0.parameters(), lr=learning_rate)

# Training Loop

active_train_loss = []
active_test_loss = []
active_epoch = []

for epoch in range(epochs + 1):
    model_0.train()
    active_epoch.append(epoch)
    y_pred = model_0.forward(X_train)
    loss = loss_fn(y_pred, y_train)
    active_train_loss.append(loss.detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.inference_mode():
        y_test_preds = model_0.forward(X_test)
        test_loss = loss_fn(y_test_preds, y_test)
        active_test_loss.append(test_loss.detach().numpy())

    if epoch % 10 == 0:
        print(f'Epoch: {epoch} \n Train loss: {loss} \n Test loss: {test_loss}')
        print(model_0.state_dict())

figure, axis = plt.subplots(2, 1)

y_pred = y_pred.detach().numpy()
axis[0].scatter(X_train, y_train, label = 'test data')
axis[0].plot(X_train, y_pred, c='r', label = 'regression line')
axis[0].legend()
axis[1].plot(active_epoch, active_train_loss, label='training loss')
axis[1].plot(active_epoch, active_test_loss, label = 'testing loss')
axis[1].legend()
plt.show()

# Saving the model
path = '/Users/aiden/Projects/Python/ML/src/models'
model_name = '01_LinearRegression.pth'

model_path = Path('models')
model_path.mkdir(parents=True, exist_ok=True)
model_save_path = model_path / model_name

print(f'Saving model to {model_save_path}')
torch.save(obj=model_0, f=model_save_path)