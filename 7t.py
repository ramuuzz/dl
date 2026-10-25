# IMDB movie-review sentiment using an LSTM
import torch
import torch.nn as nn
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# 1. Load and pad text sequences
MAX_WORDS, MAX_LEN = 10000, 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
x_train = torch.tensor(pad_sequences(x_train, maxlen=MAX_LEN), dtype=torch.long)
x_test = torch.tensor(pad_sequences(x_test, maxlen=MAX_LEN), dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. LSTM model needs a class to select its hidden state
class SentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(MAX_WORDS, 32)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.output = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.output(hidden[-1]).squeeze(-1)

model = SentimentModel().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. Train using mini-batches
batch_size, epochs = 64, 3
history = []
for epoch in range(epochs):
    model.train()
    for i in range(0, len(x_train), batch_size):
        x = x_train[i:i+batch_size].to(device)
        y = y_train[i:i+batch_size].to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(x_train), batch_size):
            x = x_train[i:i+batch_size].to(device)
            y = y_train[i:i+batch_size].to(device)
            pred = model(x) >= 0.5
            correct += (pred == y.bool()).sum().item()
            total += len(y)
    accuracy = correct / total
    history.append(accuracy)
    print(f"Epoch {epoch+1}/{epochs}, training accuracy={accuracy:.4f}")

# 4. Test accuracy
model.eval()
correct = total = 0
with torch.no_grad():
    for i in range(0, len(x_test), batch_size):
        x = x_test[i:i+batch_size].to(device)
        y = y_test[i:i+batch_size].to(device)
        correct += ((model(x) >= 0.5) == y.bool()).sum().item()
        total += len(y)
print(f"Test accuracy: {100 * correct / total:.2f}%")

# 5. Plot accuracy history
plt.plot(range(1, epochs+1), history, marker="o")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("IMDB LSTM Training Accuracy"); plt.grid(True); plt.show()
