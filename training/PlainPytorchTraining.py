import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import wandb
from datasets import load_metric, Dataset
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler, DataCollatorWithPadding

training_stats = []
avg_train_loss = 0
avg_val_accuracy = 0
avg_val_loss = 0


def train_loop(clust_bert, device, train_dataloader: DataLoader):
    optimizer = AdamW(clust_bert.model.parameters(), lr=3e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader)
    )

    clust_bert.model.train()
    total_train_loss = 0

    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = clust_bert(**batch)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    global avg_train_loss
    avg_train_loss = total_train_loss / len(train_dataloader)


def eval_loop(clust_bert, device, eval_dataloader: DataLoader):
    global avg_val_loss
    metric = load_metric("accuracy")
    clust_bert.model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = clust_bert(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    global avg_val_loss
    avg_val_loss = metric.compute()["accuracy"]


def plot():
    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)

    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


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

        wandb.log({"NMI": nmi})
        wandb.log({"loss": avg_train_loss})
        wandb.log({"validation": avg_val_loss})
