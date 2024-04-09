import torch


def predict(model, text, device, tokenizer):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )

    out = {
        'text': text,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten()
    }

    input_ids = out["input_ids"].to(device)
    attention_mask = out["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0)
    )

    prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

    return prediction