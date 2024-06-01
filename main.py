import torch
import torch.utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.Food101(
    root='data',
    download=True,
    split='train',
    transform=transform
)

test_data = datasets.Food101(
    root='data',
    download=True,
    split='test',
    transform=transform
)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=4)