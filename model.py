import torch
import torch.nn as nn

class TinyGPT(nn.Module):
   def __init__(self, vocab_size, embed_dim, block_size):
       super().__init__()
       self.token_embedding = nn.Embedding(vocab_size, embed_dim)
       self.position_embedding = nn.Embedding(block_size, embed_dim)
       self.fc = nn.Linear(embed_dim, vocab_size)

       self.block_size = block_size

   def forward(self, idx):
       # Ensure idx is 2D (batch_size, seq_length)
       if idx.ndimension() == 1:
           idx = idx.unsqueeze(0)  # Add batch dimension if it's missing

       batch_size, seq_length = idx.size()  # Get batch size and sequence length
       token_embeddings = self.token_embedding(idx)  # (batch_size, seq_length, embed_dim)
      
       # Correct positional embeddings setup
       position_ids = torch.arange(seq_length, device=idx.device).unsqueeze(0).expand(batch_size, seq_length)  # (batch_size, seq_length)
       position_embeddings = self.position_embedding(position_ids)  # (batch_size, seq_length, embed_dim)

       x = token_embeddings + position_embeddings  # Add token and position embeddings
       x = torch.relu(x)  # Optional: apply activation

       logits = self.fc(x)  # Final logits
       return logits

   @torch.no_grad()
   def generate(self, idx, max_new_tokens, temperature=1.0, top_k=10):
       for _ in range(max_new_tokens):
           idx_cond = idx[:, -self.position_embedding.num_embeddings:]  # Ensure we only consider the latest tokens
           logits = self(idx_cond)
           logits = logits[:, -1, :]  # Focus on the last token in the sequence
           logits = logits / temperature  # Apply temperature for creativity

           # Top-k sampling (only the top k logits are considered)
           top_k_vals, top_k_indices = torch.topk(logits, top_k, dim=-1)
           probs = torch.softmax(top_k_vals, dim=-1)
          
           next_token = top_k_indices.gather(-1, torch.multinomial(probs, num_samples=1))
           idx = torch.cat((idx, next_token), dim=1)  # Append generated token

       return idx

