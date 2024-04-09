import torch
import numpy as np

from tqdm import tqdm


def fit(model, train_dataset, train_dataloader, device, loss_fn, optimizer, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0

    for data in tqdm(train_dataloader, desc="Fit"):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    train_acc = correct_predictions.double() / len(train_dataset)
    train_loss = np.mean(losses)
    return train_acc, train_loss


def eval(model, valid_dataset, valid_dataloader, device, loss_fn):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for data in tqdm(valid_dataloader, desc="Eval"):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            loss = loss_fn(outputs.logits, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    val_acc = correct_predictions.double() / len(valid_dataset)
    val_loss = np.mean(losses)
    return val_acc, val_loss