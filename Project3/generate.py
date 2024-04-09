import pandas as pd
import torch
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score

from pred import predict

TOKENIZER_PATH = 'cointegrated/rubert-tiny'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data = pd.read_csv('test.csv')

texts = list(test_data['text'])
labels = list(test_data['labels'])

model = torch.load('best_model.pt')
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

predictions = [predict(model, t, device, tokenizer) for t in texts]

accuracy = accuracy_score(labels, predictions)

print(f'Accuracy: {round(accuracy, 4)}')