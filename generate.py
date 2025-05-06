import torch
from model import TinyGPT
from data import load_text

# Load dataset and model
data, stoi, itos = load_text("facts_list.txt")
vocab_size = len(stoi)

# Match your model to the trained one
model = TinyGPT(vocab_size, embed_dim=128, block_size=256)  # adjust dims if needed
model.load_state_dict(torch.load("tinygpt.pt"))
model.eval()

# Ask user for a prompt
prompt = input("📝 Enter a prompt: ")

# Handle unknown characters gracefully
try:
   idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
except KeyError as e:
   print(f"❌ Error: Character {e} not in vocabulary.")
   print("Try using a prompt that only contains characters from your training data.")
   exit()

# Generate and decode
generated = model.generate(idx, max_new_tokens=100)[0].tolist()
text = ''.join([itos[i] for i in generated])

print("\n🧠 GPT says:\n" + text)
