# train.py

import torch
import torch.nn as nn
from model import TinyGPT
from data import load_text

# Load dataset and model
data, stoi, itos = load_text()
vocab_size = len(stoi)
block_size = 256  # Set block size
embed_dim = 128  # Set embedding dimension

model = TinyGPT(vocab_size, embed_dim=embed_dim, block_size=block_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 20000  # Adjust the number of epochs
batch_size = 32
seq_length = 128  # Adjust sequence length

for epoch in range(epochs):
   model.train()
   optimizer.zero_grad()

   # Create batches of sequences
   idx = data[torch.randint(0, data.size(0) - seq_length, (batch_size,))]
   target = data[torch.randint(0, data.size(0) - seq_length, (batch_size,))]

   output = model(idx)
   loss = nn.CrossEntropyLoss()(output.view(-1, vocab_size), target.view(-1))

   loss.backward()
   optimizer.step()

   if epoch % 10 == 0:
       print(f"Epoch {epoch}, Loss: {loss.item()}")

   # Save model checkpoints every 100 epochs
   if epoch % 100 == 0:
       torch.save(model.state_dict(), "tinygpt.pt")
