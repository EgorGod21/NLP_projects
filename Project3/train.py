import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from custom_dataset import CustomDataset
from fit_eval import fit, eval

MODEL_PATH = 'cointegrated/rubert-tiny'
TOKENIZER_PATH = 'cointegrated/rubert-tiny'
MODEL_SAVE_PATH = 'best_model.pt'
NUM_CLASSES = 2
EPOCHS = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = pd.read_csv('train.csv')
valid_data = pd.read_csv('valid.csv')

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

out_features = model.bert.encoder.layer[0].output.dense.out_features

model.classifier = torch.nn.Linear(out_features, NUM_CLASSES)
model.to(device)

train_dataset = CustomDataset(train_data['text'], train_data['labels'], tokenizer)
valid_dataset = CustomDataset(valid_data['text'], valid_data['labels'], tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=True)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * EPOCHS
    )
loss_fn = torch.nn.CrossEntropyLoss().to(device)

best_accuracy = 0

for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  train_acc, train_loss = fit(model, train_dataset, train_dataloader,
                              device, loss_fn, optimizer, scheduler)
  print(f'Train loss {round(train_loss, 4)} accuracy {round(train_acc, 4)}')

  val_acc, val_loss = eval(model, valid_dataset, valid_dataloader, device, loss_fn)
  print(f'Val loss {round(val_loss, 4)} accuracy {round(val_acc, 4)}')
  print('-' * 10)

  if val_acc > best_accuracy:
    torch.save(model, MODEL_SAVE_PATH)
    best_accuracy = val_acc

    model = torch.load(MODEL_SAVE_PATH)
