# English-Hindi sentence retrieval using TF-IDF and a small neural network
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load sentence pairs: CSV needs English and Hindi columns
data = pd.read_csv("Dataset_English_Hindi.csv", encoding="utf-8", on_bad_lines="skip").dropna().head(3000)
english = data["English"].astype(str).tolist()
hindi = data["Hindi"].astype(str).tolist()

# 2. Convert both languages into fixed-size vectors
N = 500
en_vec = TfidfVectorizer(max_features=N, stop_words="english")
hi_vec = TfidfVectorizer(max_features=N)
X = en_vec.fit_transform(english).toarray()
Y = hi_vec.fit_transform(hindi).toarray()

# 3. Simple encoder and decoder (no custom class needed)
encoder = nn.Sequential(nn.Linear(N, 128), nn.ReLU(), nn.Linear(128, 64))
decoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, N))
loader = DataLoader(
    TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)),
    batch_size=32, shuffle=True
)
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# 4. Train the English-to-Hindi vector mapping
for epoch in range(10):
    total_loss = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        z = encoder(x)
        output = decoder(z)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/10, loss={total_loss / len(loader):.4f}")

# 5. Encode the known English sentences once for similarity search
encoder.eval()
with torch.no_grad():
    known_vectors = encoder(torch.tensor(X, dtype=torch.float32)).numpy()

# 6. Retrieve the closest sentence pair for each query
print("Type 'exit' to stop.")
while True:
    query = input("English sentence: ").strip()
    if query.lower() == "exit":
        break
    if not query:
        continue
    query_x = en_vec.transform([query]).toarray()
    with torch.no_grad():
        query_z = encoder(torch.tensor(query_x, dtype=torch.float32)).numpy()
    scores = cosine_similarity(query_z, known_vectors)[0]
    index = int(np.argmax(scores))
    print("Closest English:", english[index])
    print("Hindi pair:", hindi[index])
    print(f"Similarity: {scores[index]:.3f}")
