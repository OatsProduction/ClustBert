import torch
import wandb
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from evaluation.evaluate_model import evaluate_model
from models.pytorch.ClustBERT import ClustBERT
from training import DataSetUtils
from training.PlainPytorchTraining import train_loop


def start_train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        device = torch.device("cuda:0")

        train = DataSetUtils.get_million_headlines()
        train = train.select(range(1, 10))

        clust_bert = ClustBERT(config.k)
        clust_bert.to(device)
        wandb.watch(clust_bert)

        train = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, train)
        data_collator = DataCollatorWithPadding(tokenizer=clust_bert.tokenizer)

        for epoch in range(0, 1):
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

        clust_bert.to(torch.device("cpu"))
        result = evaluate_model(clust_bert.model, ["MR", "CR"], config)
        wandb.log("MR", result["MR"]["acc"])
        wandb.log("CR", result["CR"]["acc"])

        # clust_bert.save()


if __name__ == '__main__':
    sweep_config = {
        "name": "Cool-Sweep",
        "method": "random",
        "parameters": {
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
