# 1. Imports
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 2. Load data: data.csv must contain columns X and Y
data = pd.read_csv("data.csv")
X = torch.tensor(data["X"].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(data["Y"].values, dtype=torch.float32).view(-1, 1)

# 3. Model, loss, optimizer
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 500

# 4. Train
for epoch in range(epochs):
    prediction = model(X)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")

# 5. Predict
x_test = torch.tensor([[6.0]], dtype=torch.float32)
with torch.no_grad():
    y_pred = model(x_test).item()
print("Weight:", model.weight.item())
print("Bias:", model.bias.item())
print("Prediction for x=6:", y_pred)

# 6. Plot data and fitted line
x_np, y_np = X.numpy().ravel(), y.numpy().ravel()
x_line = np.linspace(x_np.min(), max(x_np.max(), 7), 100)
y_line = model.weight.item() * x_line + model.bias.item()
plt.scatter(x_np, y_np, label="Actual data")
plt.plot(x_line, y_line, label="Regression line")
plt.scatter([6], [y_pred], marker="x", s=100, label="Prediction at x=6")
plt.xlabel("X"); plt.ylabel("Y"); plt.legend(); plt.grid(True); plt.show()
