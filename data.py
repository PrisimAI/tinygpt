from datasets import load_dataset
import torch

def load_text(dataset_name="wikimedia/wikipedia", config_name="20231101.en", split="train", limit=None):
   print("🔄 Loading dataset...")
   dataset = load_dataset(dataset_name, config_name, split=split)
   # Extract just the text field
   texts = [item["text"] for item in dataset]
   if limit:
       texts = texts[:limit]
   text = "\n".join(texts)

   # Build vocabulary
   chars = sorted(set(text))
   stoi = {ch: i for i, ch in enumerate(chars)}
   itos = {i: ch for ch, i in stoi.items()}
   data_as_int = [stoi[c] for c in text]

   import torch
   return torch.tensor(data_as_int), stoi, itos
