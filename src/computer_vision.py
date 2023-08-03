# Conventional imports
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
# Some utils that make our work easy
from torchvision import datasets
from torchvision.transforms import ToTensor
# Some utils for taking metrics
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from timeit import default_timer as timer 
# Data science libs (they are friends not foes)
import matplotlib.pyplot as plt
import random
from tqdm.auto import tqdm


train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)


BATCH_SIZE = 32
CLASS_NAMES = train_data.classes
DEVICE = 'mps'
epochs = 10


train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class LinearNetwork(nn.Module):
    def __init__(self, input_features, hidden_units, output_features):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    def forward(self, x):
        return self.layer_stack(x)
    
class ReLUNetwork(nn.Module):
    def __init__(self, input_features, hidden_units, output_features):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features= hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer_stack(x)

class ConvNeuralNetwork(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x


def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = DEVICE):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        if batch % 100 == 0:
            print(f'Processed {batch * len(X)} of {len(data_loader.dataset)} samples')

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%\n")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = DEVICE):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

def make_predictions(model:nn.Module,
                     data:list,
                     device:torch.device = DEVICE):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            logits = model.forward(sample)
            probs = torch.softmax(logits.squeeze(), dim=0)
            pred_probs.append(probs.cpu())
    return torch.stack(pred_probs)


model_2 = ConvNeuralNetwork(
    input_shape=1,
    hidden_units=10,
    output_shape=len(CLASS_NAMES)
).to(DEVICE)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_2.parameters(), lr=0.03)


time_start = timer()
for epoch in tqdm(range(epochs), position=0, leave=True):
    print(f'Epoch:{epoch}')
    train_step(
        model=model_2,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy,
        device=DEVICE
    )
    test_step(
        model=model_2,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy,
        device=DEVICE
    )
time_end = timer()
print_train_time(time_start, time_end, device=DEVICE)

testing_samples = []
testing_labels = []

for sample, label in random.sample(list(test_data), k=9):
    testing_samples.append(sample)
    testing_labels.append(label)

probability_dist_tensor = make_predictions(model=model_2,
                                           data=testing_samples)
distribution_key = probability_dist_tensor.argmax(dim=1)

plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3

for i, sample in enumerate(testing_samples):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(sample.squeeze(), cmap='gray')
    plt.axis('off')
    pred_label = CLASS_NAMES[distribution_key[i]]
    true_label = CLASS_NAMES[testing_labels[i]]
    title_text = f"Pred: {pred_label} | Truth: {true_label}"

    if pred_label == true_label:
        plt.title(title_text, c='g')
    else:
        plt.title(title_text, c='r')

plt.show()

y_preds = []
model_2.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc='Making Predictions...'):
        X, y = X.to(DEVICE), y.to(DEVICE)
        y_logits = model_2.forward(X)
        prob_dist = torch.softmax(y_logits.squeeze(), dim=0).argmax(dim=1)
        y_preds.append(prob_dist.cpu())

print(y_preds)
y_preds_tensor = torch.cat(y_preds)

plt.show()

print('Saving model ')
torch.save(model_2, '/Users/aiden/Projects/Python/ML/models/mnist_classifier.pt')