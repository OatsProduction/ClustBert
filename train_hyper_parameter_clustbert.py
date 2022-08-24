import wandb
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from models.pytorch.ClustBERT import ClustBERT
from training import DataSetUtils
from training.PlainPytorchTraining import train_loop


def start_train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        train = DataSetUtils.get_million_headlines()
        train = train.select(range(1, 1000))

        clust_bert = ClustBERT(config.k)
        wandb.watch(clust_bert)

        train = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, train)
        data_collator = DataCollatorWithPadding(tokenizer=clust_bert.tokenizer)

        for epoch in range(0, 12):
            print("Loop in Epoch: " + str(epoch))
            pseudo_label_data, silhouette = clust_bert.cluster_and_generate(train)
            # nmi = normalized_mutual_info_score(train["labels"], pseudo_label_data["labels"])

            train_dataloader = DataLoader(
                pseudo_label_data, shuffle=True, batch_size=8, collate_fn=data_collator
            )
            loss = train_loop(clust_bert, train_dataloader, config)

            wandb.log({
                "loss": loss,
                "silhouette": silhouette
            })

        clust_bert.save()


if __name__ == '__main__':
    sweep_config = {
        "name": "Cool-Sweep",
        "method": "random",
        "parameters": {
            "k": {
                "values": [100]
            },
            "learning_rate": {
                "values": [1e-05]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="Test-ClustBert")
    wandb.agent(sweep_id, start_train, count=5)
