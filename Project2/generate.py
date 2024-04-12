from create_model import create_model
import torch
import sys

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

chars = sorted(list(set(text)))
vocab_size = len(chars)

model_name = sys.argv[1]
path_save = sys.argv[2]

assert f'model_{model_name}.pth' == path_save, 'model name and path save are different'

model = create_model(model_name, n_head, n_embd, block_size, dropout, vocab_size, n_layer, device)
model.load_state_dict(torch.load(f'models_save/{path_save}'))
model = model.to(device)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
