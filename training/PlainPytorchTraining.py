import torch
import wandb
from datasets import load_metric, Dataset
from sklearn.metrics import normalized_mutual_info_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler, DataCollatorWithPadding

training_stats = []
accuracy = 0


def train_loop(model, device, train_dataloader: DataLoader):
    optimizer = AdamW(model.parameters(), lr=3e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader)
    )

    model.train()
    total_train_loss = 0

    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return total_train_loss / len(train_dataloader)


def eval_loop(model, device, eval_dataloader: DataLoader):
    global accuracy
    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    global accuracy
    accuracy = metric.compute()["accuracy"]


def start_training(clust_bert, train: Dataset, validation: Dataset, device):
    data_collator = DataCollatorWithPadding(tokenizer=clust_bert.tokenizer)
    eval_dataloader = DataLoader(validation, batch_size=8, collate_fn=data_collator)

    num_epochs = 16

    for epoch in range(num_epochs):
        print("Loop in Epoch: " + str(epoch))
        pseudo_label_data = clust_bert.cluster_and_generate(train)
        nmi = normalized_mutual_info_score(train["labels"], pseudo_label_data["labels"])

        train_dataloader = DataLoader(
            pseudo_label_data, shuffle=True, batch_size=8, collate_fn=data_collator
        )
        train_loop(clust_bert, device, train_dataloader)
        eval_loop(clust_bert, device, eval_dataloader)

        wandb.log({"NMI": nmi, "loss": avg_train_loss, "validation": accuracy})
