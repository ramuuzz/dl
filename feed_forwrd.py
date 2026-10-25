# 1. Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 2. Device and data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 3. Model: CIFAR-10 image = 3 x 32 x 32 = 3072 values
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3072, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 10)
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5
loss_history = []

# 4. Train
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")

# 5. Test accuracy
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
print(f"Test accuracy: {100 * correct / total:.2f}%")

# 6. Plot loss
plt.plot(range(1, epochs + 1), loss_history, marker="o")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Feedforward Network Training Loss")
plt.grid(True); plt.show()
