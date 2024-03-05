import torch
import sys
import torch.nn.functional as F
from colorama import Fore, Style, init

init()

def build_dataset(words, block_size=3):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y

args = sys.argv
words = open(args[2], 'r').read().splitlines()

parameter = torch.load(args[1])
C = parameter['C']
W1 = parameter['W1']
b1 = parameter['b1']
W2 = parameter['W2']
b2 = parameter['b2']

stoi = torch.load('stoi.pth')
itos = {i:s for s,i in stoi.items()}

X_test, Y_test = build_dataset(words)

emb = C[X_test]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y_test)


print(Style.BRIGHT + f'test loss: {loss.item():.2f}')
print('example:')

g = torch.Generator().manual_seed(2147483645 + 10)
for _ in range(5):
    out = []
    context = [0] * 3
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(Fore.CYAN + ''.join(itos[i] for i in out))