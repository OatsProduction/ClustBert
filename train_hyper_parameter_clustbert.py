import argparse

import senteval
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, BertTokenizer

from models.pytorch.ClustBERT import ClustBERT
from training import DataSetUtils
from training.PlainPytorchTraining import train_loop


def prepare(params, samples):
    return


def batcher(params, batch):
    sentences = [" ".join(s).lower() for s in batch]

    with torch.no_grad():
        y = params['tokenizer'](sentences, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        y = params['model'](y)[0]
        y = y[:, 0, :].view(-1, 768)

    return y


def start_train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        wandb.config.update(config)
        config = wandb.config
        wandb.run.name = "crop_" + str(config.random_crop_size) + "_lr" + str(config.learning_rate) + "_k" + str(
            config.k) + "_epoch" + str(config.epochs) + "_" + wandb.run.id

        device = torch.device("cuda:1")

        big_train_dataset = DataSetUtils.get_million_headlines()

        clust_bert = ClustBERT(config.k)
        clust_bert.to(device)
        wandb.watch(clust_bert)

        big_train_dataset = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, big_train_dataset)
        data_collator = DataCollatorWithPadding(tokenizer=clust_bert.tokenizer)

        for epoch in range(0, config.epochs):
            print("Loop in Epoch: " + str(epoch))
            big_train_dataset = big_train_dataset.shuffle(seed=epoch)
            train = big_train_dataset.select(range(1, config.random_crop_size))

            pseudo_label_data, silhouette = clust_bert.cluster_and_generate(train, device)
            # nmi = normalized_mutual_info_score(train["labels"], pseudo_label_data["labels"])

            train_dataloader = DataLoader(
                pseudo_label_data, batch_size=16, collate_fn=data_collator
            )
            loss = train_loop(clust_bert, train_dataloader, device, config)

            wandb.log({
                "loss": loss,
                "silhouette": silhouette
            })

            if epoch % 10 == 0:
                validate(clust_bert, device)

        print("Start with SentEval")

        clust_bert.to(torch.device("cpu"))
        clust_bert.eval()

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
        result = se.eval(["MR", "CR"])
        mr_cr = (float(result["MR"]["acc"]) + float(result["CR"]["acc"])) / 2
        wandb.log({"MR_CR_score": mr_cr})


def validate(clust_bert, old_device):
    clust_bert.to(torch.device("cpu"))
    clust_bert.eval()
    for param in clust_bert.parameters():
        param.requires_grad = False
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
    result = se.eval(["STS13"])
    wandb.log({"STS13": result["STS13"]["all"]["pearson"]["mean"]})

    clust_bert.train()
    for param in clust_bert.parameters():
        param.requires_grad = True
    clust_bert.to(old_device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, help="Nots about the training run")
    parser.add_argument("-k", "--k", type=int, help="Nots about the training run")
    parser.add_argument("-c", "--random_crop_size", type=int, help="Nots about the training run")
    parser.add_argument("-l", "--learning_rate", type=float, help="Nots about the training run")
    parser.add_argument("-s", "--senteval_path", type=str, help="Nots about the training run")

    args = parser.parse_args()
    start_train(args)
