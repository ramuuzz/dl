# Experiment: initialization, dropout and L2-style weight decay
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Data
torch.manual_seed(1)
train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# 2. A simple model; Sequential avoids a custom class
def make_model(init="xavier", dropout=0.0):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 10)
    )
    for layer in model:
        if isinstance(layer, nn.Linear):
            if init == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif init == "kaiming":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    return model

# 3. Train/evaluate function
def run_experiment(init="xavier", dropout=0.0, weight_decay=0.0):
    model = make_model(init, dropout)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    losses, accuracies = [], []

    for epoch in range(3):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                pred = model(images).argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        losses.append(loss_sum / len(train_loader))
        accuracies.append(100 * correct / total)
        print(f"{init}, dropout={dropout}, wd={weight_decay} | epoch {epoch+1}: accuracy={accuracies[-1]:.2f}%")
    return losses, accuracies

# 4. Compare settings
results = {
    "Xavier": run_experiment("xavier"),
    "Kaiming": run_experiment("kaiming"),
    "Dropout": run_experiment("kaiming", dropout=0.5),
    "Weight decay": run_experiment("kaiming", weight_decay=0.0001)
}

# 5. Plot results
for name, (losses, accuracies) in results.items():
    plt.plot(losses, label=name)
plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.legend(); plt.grid(True); plt.show()
for name, (losses, accuracies) in results.items():
    plt.plot(accuracies, marker="o", label=name)
plt.xlabel("Epoch"); plt.ylabel("Test accuracy (%)"); plt.legend(); plt.grid(True); plt.show()
