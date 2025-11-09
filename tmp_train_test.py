import torch, torch.nn as nn
import torch.optim as optim
from logicsponge.processmining.neural_networks import TransformerModel, train_rnn, PreprocessData

# Tiny synthetic dataset (list of sequences of token indices)
# Use tokens >0; 0 is padding.
sequences = [
    [1,2,3,4],
    [2,3,4],
    [3,4,5,6,7],
    [1,3,5],
]

# Convert to padded tensor [B,L] left-padded
max_len = max(len(s) for s in sequences)
B = len(sequences)
x = torch.zeros((B, max_len), dtype=torch.long)
for i, s in enumerate(sequences):
    x[i, max_len-len(s):] = torch.tensor(s, dtype=torch.long)

# Split train/val
train_seqs = x[:3]
val_seqs = x[3:]

vocab_size = 20
model = TransformerModel(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64, num_layers=2, attention_heads=4, use_one_hot=False, pos_encoding_type="rope")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('Starting tiny train...')
model = train_rnn(model, train_seqs, val_seqs, criterion, optimizer, batch_size=2, epochs=1, patience=1, window_size=None, left_pad=True)
print('Done tiny train.')
