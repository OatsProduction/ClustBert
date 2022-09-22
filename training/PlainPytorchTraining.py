import numpy as np
import senteval
import torch
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler
from transformers import get_scheduler, BertTokenizer

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

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
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
        model.zero_grad()

        b_input_ids = batch["input_ids"]
        b_input_mask = batch["attention_mask"]
        b_labels = batch["labels"]

        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return total_train_loss / len(train_dataloader)


def generate_clustering_statistic(dataset: Dataset) -> dict:
    labels = dataset["labels"]
    result = torch.bincount(labels)
    amount_in_max_cluster = torch.max(result)
    average_cluster_size = torch.mean(result.double())
    standard_deviation = torch.std(result.double())
    under_x_cluster = 0
    x = 15

    for value in result:
        if value.item() <= x:
            under_x_cluster += 1

    return {
        "standard_deviation": standard_deviation,
        "amount_in_max_cluster": amount_in_max_cluster,
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
        'model': clust_bert.model.bert,
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


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = [int(i) for i in res]
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)
