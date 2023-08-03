import torch

x = torch.tensor([1, 2])
y = torch.tensor([1, 2])


test = torch.eq(x == 1, y == 1).sum().item()
print(test)