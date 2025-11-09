import torch
from logicsponge.processmining.neural_networks import TransformerModel

# Minimal sanity test for forward pass after mask patch
vocab_size = 50
model = TransformerModel(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64, num_layers=2, attention_heads=4, use_one_hot=False, pos_encoding_type="rope")
model.eval()

# Create left-padded batch: two sequences of different lengths
seqs = [torch.tensor([5,6,7,8,9]), torch.tensor([10,11,12])]
# Left-pad manually to length 5
pad_len = 5
batch = []
for s in seqs:
    if s.numel() < pad_len:
        pad = torch.zeros(pad_len - s.numel(), dtype=torch.long)
        batch.append(torch.cat([pad, s]))
    else:
        batch.append(s)
inputs = torch.stack(batch)  # [2,5]

with torch.no_grad():
    out = model(inputs)
print('Output shape:', tuple(out.shape))
print('Any NaNs:', bool(torch.isnan(out).any().item()))
print('Sample logits first row last token:', out[0,-1,:5].tolist())
