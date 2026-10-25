# Transfer learning: pretrained VGG19 on MNIST
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import vgg19, VGG19_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Data: VGG19 expects RGB images sized for its pretrained features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# 2. Load pretrained model and freeze its existing weights
model = vgg19(weights=VGG19_Weights.DEFAULT)
for parameter in model.parameters():
    parameter.requires_grad = False
model.classifier[6] = nn.Linear(4096, 10)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier[6].parameters(), lr=0.001)

# 3. Train only the replacement final layer
for epoch in range(2):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/2 complete")

# 4. Test
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
print(f"Test accuracy: {100 * correct / total:.2f}%")
