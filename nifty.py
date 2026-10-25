# NIFTY-50 forecasting using a sliding window and LSTM
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yfinance as yf

# 1. Download and normalize closing prices
prices = yf.download("^NSEI", period="5y", auto_adjust=True, progress=False)["Close"].dropna().values.reshape(-1, 1)
minimum, maximum = prices.min(), prices.max()
data = (prices - minimum) / (maximum - minimum)

# 2. Make sequences: previous 30 days -> next day's price
X, y = [], []
for i in range(30, len(data)):
    X.append(data[i-30:i])
    y.append(data[i])
X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

# 3. LSTM model
class PriceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

model = PriceLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Train
for epoch in range(10):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch+1}/10, loss={loss.item():.5f}")

# 5. Forecast the next 10 days recursively
model.eval()
sequence = torch.tensor(data[-30:], dtype=torch.float32).unsqueeze(0)
predictions = []
with torch.no_grad():
    for _ in range(10):
        prediction = model(sequence)
        predictions.append(prediction.item())
        sequence = torch.cat((sequence[:, 1:, :], prediction.unsqueeze(1)), dim=1)
predictions = np.array(predictions) * (maximum - minimum) + minimum
print("Next 10 predicted values:", predictions)

# 6. Plot recent values and forecast
recent = prices[-100:].ravel()
plt.plot(range(100), recent, label="Recent actual")
plt.plot(range(100, 110), predictions, marker="o", label="Forecast")
plt.xlabel("Trading-day index"); plt.ylabel("NIFTY-50 close"); plt.legend(); plt.grid(True); plt.show()
