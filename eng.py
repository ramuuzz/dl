# English-to-Hindi sequence-to-sequence translation using GRU
import pandas as pd
import torch
import torch.nn as nn

# 1. Load dataset: Dataset.csv needs English and Hindi columns
data = pd.read_csv("Dataset.csv").dropna().head(3000)

def make_vocab(sentences, max_words=5000):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for word in " ".join(sentences.astype(str)).lower().split():
        if word not in vocab and len(vocab) < max_words:
            vocab[word] = len(vocab)
    return vocab

def encode(sentence, vocab):
    return [vocab.get(word, 3) for word in str(sentence).lower().split()]

en_vocab = make_vocab(data["English"])
hi_vocab = make_vocab(data["Hindi"])
X, Y = [], []
for english, hindi in zip(data["English"], data["Hindi"]):
    source = encode(english, en_vocab)[:15]
    target = [1] + encode(hindi, hi_vocab)[:14] + [2]
    X.append(source + [0] * (15 - len(source)))
    Y.append(target + [0] * (16 - len(target)))
X, Y = torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Encoder-decoder model (custom class handles hidden state)
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_embedding = nn.Embedding(len(en_vocab), 64)
        self.hi_embedding = nn.Embedding(len(hi_vocab), 64)
        self.encoder = nn.GRU(64, 128, batch_first=True)
        self.decoder = nn.GRU(64, 128, batch_first=True)
        self.output = nn.Linear(128, len(hi_vocab))

    def forward(self, source, target):
        _, hidden = self.encoder(self.en_embedding(source))
        decoded, _ = self.decoder(self.hi_embedding(target[:, :-1]), hidden)
        return self.output(decoded)

model = Seq2Seq().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. Train
for epoch in range(5):
    model.train()
    total_loss = 0.0
    for i in range(0, len(X), 64):
        source, target = X[i:i+64].to(device), Y[i:i+64].to(device)
        optimizer.zero_grad()
        output = model(source, target)
        loss = criterion(output.reshape(-1, output.size(-1)), target[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/5, loss={total_loss / max(1, (len(X)+63)//64):.4f}")

# 4. Greedy translation for one input sentence
reverse_hi = {index: word for word, index in hi_vocab.items()}
def translate(sentence):
    source = encode(sentence, en_vocab)[:15]
    source = torch.tensor([source + [0] * (15-len(source))], dtype=torch.long).to(device)
    model.eval()
    result = []
    with torch.no_grad():
        _, hidden = model.encoder(model.en_embedding(source))
        token = torch.tensor([[1]], dtype=torch.long).to(device)
        for _ in range(15):
            decoded, hidden = model.decoder(model.hi_embedding(token), hidden)
            token = model.output(decoded[:, -1, :]).argmax(dim=1, keepdim=True)
            index = token.item()
            if index == 2:
                break
            result.append(reverse_hi.get(index, "<unk>"))
    print("English:", sentence)
    print("Hindi:", " ".join(result))

translate(input("Enter an English sentence: "))
