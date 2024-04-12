import torch
import numpy as np
import time
import sys
import json
sys.path.append('models')

from create_model import create_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
block_size = 128
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 8
n_layer = 6
dropout = 0.2


with open('Iliad_and_Odyssey.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    perplexity = []
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            if split == 'val':
              perplexity.append(torch.exp(loss).item())
        out[split] = losses.mean()
        if split == 'val':
              perplexity = np.mean(perplexity)
    model.train()
    return out, perplexity


model_name = sys.argv[1]
model = create_model(model_name, n_head, n_embd, block_size, dropout, vocab_size, n_layer, device)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

losses_dict = {'train':[], 'val':[]}
perplexities = []
times = []
start_time = time.time()
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses, perplexity = estimate_loss(model)
        losses_dict['train'].append(losses['train'].item())
        losses_dict['val'].append(losses['val'].item())
        perplexities.append(perplexity)
        end_time = time.time()
        diff_time = end_time - start_time
        times.append(diff_time)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        start_time = time.time()

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

path_to_save = f'models_save/model_{model_name}.pth'
torch.save(model.state_dict(), path_to_save)

data = {
    'losses_dict': losses_dict,
    'perplexities': perplexities,
    'times': times
}

file_name = f'training_results/{model_name}.json'

with open(file_name, 'w') as f:
    json.dump(data, f)