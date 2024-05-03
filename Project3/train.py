from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import transformers
from colorama import Fore, Style

model_name = "cointegrated/rut5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset("RussianNLP/Mixed-Summarization-Dataset")

train_dataset = dataset['train'].shuffle(seed=42).select(range(10_000))
val_dataset = dataset['train'].shuffle(seed=42).select(range(10_000, 12_000))

def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

data_collator = transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = transformers.Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
    )

trainer = transformers.Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

print(Fore.GREEN + Style.BRIGHT + 'модель сохранена')