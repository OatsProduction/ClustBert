import senteval
import torch
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler, BertTokenizer

from models.pytorch.ClustBERT import ClustBERT

training_stats = []
accuracy = 0


def prepare(params, samples):
    return


def batcher(params, batch):
    sentences = [" ".join(s).lower() for s in batch]

    with torch.no_grad():
        y = params['tokenizer'](sentences, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        y = params['model'](y)[0]
        y = y[:, 0, :].view(-1, 768)

    return y


def train_loop(model, train_dataloader: DataLoader, device, config=None):
    learning_rate = 3e-5 if config is None or config.learning_rate is None else config.learning_rate

    optimizer = AdamW(model.parameters(), learning_rate)
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


def generate_clustering_statistic(clust_bert: ClustBERT, dataset: Dataset) -> dict:
    labels = dataset["labels"]
    result = torch.bincount(labels)
    amount_in_max_cluster = torch.max(result)
    average_cluster_size = torch.mean(result)
    standard_deviation = torch.std(result)
    under_x_cluster = 0
    x = 15

    for value in result:
        if value.item() <= x:
            under_x_cluster += 1

    return {
        "standard_deviation": standard_deviation,
        "amount_in_max_cluster": amount_in_max_cluster,
        "average_cluster_size": average_cluster_size,
        "under_x_cluster": under_x_cluster,
    }


def get_normal_sample_pseudolabels(dataset: Dataset, num_labels: int, random_crop_size: int):
    dataset = dataset.select(range(1, random_crop_size))
    return dataset


def eval_loop(clust_bert, old_device):
    clust_bert.eval()
    for param in clust_bert.parameters():
        param.requires_grad = False
    clust_bert.to(torch.device("cpu"))

    params = {
        'model': clust_bert.model,
        'tokenizer': BertTokenizer.from_pretrained("bert-base-cased"),
        'task_path': "../SentEval/data",
        'usepytorch': True,
        'classifier': {
            'nhid': 0,
            'optim': 'adam',
            'batch_size': 64,
            'tenacity': 5,
            'epoch_size': 4
        }
    }
    se = senteval.engine.SE(params, batcher, prepare)
    result = se.eval(["CR"])

    clust_bert.train()
    for param in clust_bert.parameters():
        param.requires_grad = True
    clust_bert.to(old_device)

    return float(result["CR"]["acc"])
