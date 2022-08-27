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
        config = wandb.config
        wandb.run.name = "data_" + str(config.data) + "_lr" + str(config.learning_rate) + "_k" + str(
            config.k) + "_" + wandb.run.id

        device = torch.device("cuda:0")

        train = DataSetUtils.get_million_headlines()
        train = train.select(range(1, config.data))

        clust_bert = ClustBERT(config.k)
        clust_bert.to(device)
        wandb.watch(clust_bert)

        train = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, train)
        data_collator = DataCollatorWithPadding(tokenizer=clust_bert.tokenizer)

        for epoch in range(0, 12):
            print("Loop in Epoch: " + str(epoch))
            pseudo_label_data, silhouette = clust_bert.cluster_and_generate(train, device)
            # nmi = normalized_mutual_info_score(train["labels"], pseudo_label_data["labels"])

            train_dataloader = DataLoader(
                pseudo_label_data, shuffle=True, batch_size=8, collate_fn=data_collator
            )
            loss = train_loop(clust_bert, train_dataloader, device, config)

            wandb.log({
                "loss": loss,
                "silhouette": silhouette
            })

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


if __name__ == '__main__':
    sweep_config = {
        "name": "Cool-Sweep",
        "method": "random",
        "parameters": {
            "data": {
                "values": [500]
            },
            "senteval_path": {
                "values": ["../SentEval/data"]
            },
            "k": {
                "values": [5]
            },
            "learning_rate": {
                "values": [1e-05]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="test-project")
    wandb.agent(sweep_id, start_train, count=1)
