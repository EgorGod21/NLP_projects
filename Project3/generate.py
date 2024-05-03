from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from colorama import Fore, Style

model_name = "cointegrated/rut5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
loaded_model = AutoModelForSeq2SeqLM.from_pretrained("./saved_model").to("cuda")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = load_dataset("RussianNLP/Mixed-Summarization-Dataset")
test_dataset = dataset['test'].shuffle(seed=42).select(range(200))

summaries = []

for text in tqdm(test_dataset['text']):
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        ).to(device)

    summary_text_ids = loaded_model.generate(
        input_ids=input_ids,
        bos_token_id=loaded_model.config.bos_token_id,
        eos_token_id=loaded_model.config.eos_token_id,
        max_length=142,
        min_length=56,
        num_beams=4,
    )

    decoded_text = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    summaries.append(decoded_text)

import evaluate

rouge = evaluate.load('rouge')

results = rouge.compute(
        predictions=summaries,
        references=test_dataset['summary']
    )

print(results)
print('Исходный текст:')
print(Fore.GREEN + Style.BRIGHT + test_dataset['text'][2])
print(Style.RESET_ALL + 'Пересказ текста:')
print(Fore.GREEN + Style.BRIGHT + summaries[2])