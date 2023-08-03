import torch
from torch import nn
from torch import backends
from torch import optim
import torchviz
from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# The creation of our classification model will have several hyperparams
# - Input layer shape: same as the number of features
# - Hidden layers: We can have as many or as little as we like
# - Neurons per hidden layer: generally we have 10 - 512
# - Output layer shape: 1 in binary classification and 1 per class for multiclass
# - Hidden layer activation: ReLU usually
# - Output activation: Sigmoid for binary, softmax for multiclass
# - Loss function - Binary crossentropy  or cross entropy
# - Optimizer can be SGD or Adam in this case

# Settings
pd.options.display.max_rows = 500


# Let's get some data!!! (and make some tensors)
X, y = datasets.make_circles(n_samples=3000, noise=0.05, random_state=42)

# For visualizing our data
circle_data_vis = pd.DataFrame({'X1': X[:, 0],
                                'X2': X[:, 1],
                                'label': y})

# Generating some tensors from numpy using the torch API
X_tens = torch.from_numpy(X).type(torch.float)
y_tens = torch.from_numpy(y).type(torch.float)


# Splitting test and train sets
X_train, x_test, y_train, y_test = model_selection.train_test_split(X_tens,
                                                                    y_tens,
                                                                    test_size=0.2,
                                                                    random_state=42)

# Now it is time to build our actual model, we need to:
# 1. Set up agnostic code such as Metal Acceleration etc
# 2. Construct a model using nn.Module
# 3. Define our loss function and optimizer
# 4. Create a training and testing loop

# Defining MPS
device = 'cpu'

# Model class


class CircleModel(nn.Module):
    # Some boilerplate
    def __init__(self):
        super().__init__()
        # Here we are defining the hidden layers of our NN
        # Linear layers as well as one ReLU layer that can be used.
        self.layer1 = nn.Linear(in_features=2, out_features=50)
        self.layer2 = nn.Linear(in_features=50, out_features=50)
        self.layer3 = nn.Linear(in_features=50, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # This is the calculation that is done at every loop
        # We make sure to call ReLU on every single linear layer
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))


# This is the same model with an automatically defined forward method
# This is best for linear models because we cant define a custom forward method here
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1),
    nn.ReLU()
).to(device)


# Send our model to cpu as well as deine lr and epochs
model_01 = CircleModel().to(device)
learning_rate = 0.03
epochs = 10000

# Loss function and Optimizer
# BCE returns a number between zero and one for loss, we want closest to zero
# BCE also expects raw output logits from the model rather than processed values
loss_fn = nn.BCEWithLogitsLoss()
optim = optim.SGD(params=model_01.parameters(),
                  lr=0.03)

# Calculate Accuracy (maybe percision?)


def accuracy(y_true, y_predictions):
    correct = torch.eq(y_true, y_predictions).sum().item()
    acc = (correct/len(y_predictions)) * 100
    return acc

def precision(y_true, y_predictions):
    tp = torch.eq(y_true == 1, y_predictions == 1).sum().item()
    fp = torch.eq(y_true == 0, y_predictions == 1).sum().item()
    prec = tp/(tp + fp)
    return prec

# Imported function for graphing decision boundary


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


# Training and testing
# We need to convert raw logits to probabilities to actual labels
X_train, y_train = X_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

# This is for tracking the active counts for graphing later on
active_epoch = []
train_active_loss = []
test_active_loss = []
train_active_accuracy = []
test_active_accuracy = []

# Here is the training loop of this NN
# Open up the loop and input the desired number of epochs
# Begin tracking the epochs and place the model into training mode
# Calculate the logits (outputs of the forward pass) and squeeze them to remove an extra dim
# Pass the logits into a sigmoid function and then round them for a threshold of 0.5
# calculate the loss between the Logits and the Y_train data
# Begin to track active loss
# Calculate the accuracy metric and then begin to track active accuracy'
# Clear all gradients from the optimizer for the next loop
# Perform backpropagation and send the old data back to the model
# Perform gradient descent with the optimizer (Thank god for Adam)
# Evaluate the model
# Perform all necessary calculations within inference mode for test data
# Output any relevent data
for epoch in range(epochs + 1):
    active_epoch.append(epoch)
    model_01.train()
    y_logits = model_01(X_train).squeeze()
    # Logits to pred probs to pred labels
    y_pred = torch.round(torch.sigmoid(y_logits))
    # nn.BCELogits expects logit inputs to calculate its loss
    loss = loss_fn(y_logits, y_train)
    train_active_loss.append(loss.detach().numpy())
    acc = accuracy(y_true=y_train, y_predictions=y_pred)
    train_active_accuracy.append(acc)
    prec = precision(y_train, y_pred)
    optim.zero_grad()
    loss.backward()
    optim.step()
    model_01.eval()

    with torch.inference_mode():
        test_logits = model_01(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_active_loss.append(test_loss.detach().numpy())
        test_acc = accuracy(y_true=y_test, y_predictions=test_pred)
        test_active_accuracy.append(test_acc)
        test_precision = precision(y_test, test_pred)
    if epoch % 10 == 0:
        print(
            f'Epoch: {epoch} \n TR_BCELoss: {loss} TR_Accuracy: {acc}% \n TE_BCELoss: {test_loss} TE_Accuracy: {test_acc}%')


plt.subplot(2, 2, 1)
plt.title('Train Data')
plot_decision_boundary(model_01, X_train, y_train)
plt.subplot(2, 2, 2)
plt.title('Test Data')
plot_decision_boundary(model_01, x_test, y_test)
plt.subplot(2, 2, 3)
plt.title('Binary Cross Entropy Loss')
plt.plot(active_epoch, train_active_loss, label='Train BCELoss')
plt.plot(active_epoch, test_active_loss, label='Test BCELoss')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.legend()
plt.subplot(2, 2, 4)
plt.title('Accuracy Metric')
plt.plot(active_epoch, train_active_accuracy, label='Train Accuracy')
plt.plot(active_epoch, test_active_accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

path = '/Users/aiden/Projects/Python/ML/src/models'
model_name = '01_Classification.pt'

model_path = Path('models')
model_path.mkdir(parents=True, exist_ok=True)
model_save_path = model_path / model_name

print(f'Saving model to {model_save_path}')
torch.save(obj=model_01, f=model_save_path)

plot_path = '/Users/aiden/Projects/Python/ML/figs/results_panel.png'
plt.savefig(plot_path)
