# Compare RNN, GRU and LSTM for IMDB sentiment classification
import time
import torch
import torch.nn as nn
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# 1. Load data (use a subset to keep the experiment manageable)
MAX_WORDS, MAX_LEN = 10000, 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
x_train = torch.tensor(pad_sequences(x_train[:10000], maxlen=MAX_LEN), dtype=torch.long)
y_train = torch.tensor(y_train[:10000], dtype=torch.float32)
x_test = torch.tensor(pad_sequences(x_test[:2000], maxlen=MAX_LEN), dtype=torch.long)
y_test = torch.tensor(y_test[:2000], dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Model: the recurrent layer changes by type
class TextModel(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.embedding = nn.Embedding(MAX_WORDS, 64)
        if kind == "RNN":
            self.rnn = nn.RNN(64, 32, batch_first=True)
        elif kind == "GRU":
            self.rnn = nn.GRU(64, 32, batch_first=True)
        else:
            self.rnn = nn.LSTM(64, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return torch.sigmoid(self.fc(hidden[-1])).squeeze(-1)

# 3. Train/evaluate each type
results = {}
for kind in ["RNN", "GRU", "LSTM"]:
    model = TextModel(kind).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    start = time.time()

    for epoch in range(2):
        model.train()
        for i in range(0, len(x_train), 128):
            x = x_train[i:i+128].to(device)
            y = y_train[i:i+128].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(x_test), 128):
            x = x_test[i:i+128].to(device)
            y = y_test[i:i+128].to(device)
            correct += ((model(x) >= 0.5) == y.bool()).sum().item()
            total += len(y)
    results[kind] = {"accuracy": 100 * correct / total, "seconds": time.time() - start}
    print(f"{kind}: accuracy={results[kind]['accuracy']:.2f}%, time={results[kind]['seconds']:.1f}s")

# 4. Compare results
plt.bar(results.keys(), [v["accuracy"] for v in results.values()])
plt.ylabel("Test accuracy (%)"); plt.title("RNN vs GRU vs LSTM"); plt.show()
plt.bar(results.keys(), [v["seconds"] for v in results.values()])
plt.ylabel("Training time (seconds)"); plt.title("Training Time Comparison"); plt.show()
