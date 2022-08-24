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
        train = train.select(range(1, 150))

        clust_bert = ClustBERT(config)
        train = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, train)

        wandb.init(project="test-project", entity="clustbert")
        wandb.watch(clust_bert)

        data_collator = DataCollatorWithPadding(tokenizer=clust_bert.tokenizer)

        for epoch in range(0, config.epochs):
            print("Loop in Epoch: " + str(epoch))
            pseudo_label_data, silhouette = clust_bert.cluster_and_generate(train)
            # nmi = normalized_mutual_info_score(train["labels"], pseudo_label_data["labels"])

            train_dataloader = DataLoader(
                pseudo_label_data, shuffle=True, batch_size=8, collate_fn=data_collator
            )
            loss = train_loop(clust_bert, train_dataloader)

            wandb.log({
                "loss": loss,
                "silhouette": silhouette
            })

        clust_bert.save()


if __name__ == '__main__':
    hyperparameter_defaults = dict(
        learning_rate=0.001,
        k=100,
        optimizer="adam",
        epochs=12,
    )

    sweep_id = wandb.sweep(hyperparameter_defaults, project="ClustBert")
    wandb.agent(sweep_id, start_train, count=5)
