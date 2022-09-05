import argparse

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from models.pytorch.ClustBERT import ClustBERT
from training import DataSetUtils
from training.PlainPytorchTraining import train_loop, eval_loop, get_normal_sample_pseudolabels, \
    generate_clustering_statistic


def start_train(config=None):
    with wandb.init(config=config):
        wandb.config.update(config)
        config = wandb.config
        wandb.run.name = "crop_" + str(config.random_crop_size) + "_lr" + str(config.learning_rate) + "_k" + str(
            config.k) + "_epoch" + str(config.epochs) + "_" + wandb.run.id

        device = torch.device("cuda:1")

        clust_bert = ClustBERT(config.k)
        clust_bert.to(device)
        wandb.watch(clust_bert)

        big_train_dataset = DataSetUtils.get_million_headlines()
        big_train_dataset = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, big_train_dataset)
        data_collator = DataCollatorWithPadding(tokenizer=clust_bert.tokenizer)

        score = eval_loop(clust_bert, device)
        wandb.log({
            "mr_cr_score": score
        })

        for epoch in range(0, config.epochs):
            print("Loop in Epoch: " + str(epoch))
            big_train_dataset = big_train_dataset.shuffle(seed=epoch)
            pseudo_label_data, silhouette = clust_bert.cluster_and_generate(big_train_dataset, device)
            amount_in_max_cluster, under_x_cluster = generate_clustering_statistic(clust_bert, pseudo_label_data)

            pseudo_label_data = get_normal_sample_pseudolabels(pseudo_label_data, config.k, config.random_crop_size)
            train_dataloader = DataLoader(
                pseudo_label_data, batch_size=16, collate_fn=data_collator
            )

            loss = train_loop(clust_bert, train_dataloader, device, config)
            score = eval_loop(clust_bert, device)

            wandb.log({
                "loss": loss,
                "silhouette": silhouette,
                "mr_cr_score": score,
                "amount_in_max_cluster": amount_in_max_cluster,
                "under_x_cluster": under_x_cluster,
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, help="Nots about the training run")
    parser.add_argument("-k", "--k", type=int, help="Nots about the training run")
    parser.add_argument("-c", "--random_crop_size", type=int, help="Nots about the training run")
    parser.add_argument("-l", "--learning_rate", type=float, help="Nots about the training run")
    parser.add_argument("-s", "--senteval_path", type=str, help="Nots about the training run")

    args = parser.parse_args()
    start_train(args)
