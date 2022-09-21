import argparse
import logging

import torch
import wandb
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from evaluation.evaluate_model import evaluate_model, sts, senteval_tasks
from evaluation.print_evaluation import get_senteval_from_json, get_sts_from_json
from models.pytorch.ClustBERT import ClustBERT
from training import DataSetUtils
from training.PlainPytorchTraining import train_loop, eval_loop, generate_clustering_statistic, UnifLabelSampler


def start_train(config=None):
    if not args.wandb:
        wandb.init(config=config)
        wandb.config.update(config)
        config = wandb.config
        wandb.run.name = "_lr" + str(config.learning_rate) + "_k" + str(
            config.k) + "_epoch" + str(config.epochs) + "_" + wandb.run.id

    device = torch.device("cuda:1")
    torch.cuda.empty_cache()

    clust_bert = ClustBERT(config.k)
    clust_bert.to(device)
    if not args.wandb:
        wandb.watch(clust_bert)

    big_train_dataset = DataSetUtils.get_imdb().shuffle(seed=525)

    if not args.wandb:
        score = eval_loop(clust_bert, device)
        wandb.log({
            "cr_score": score
        })

    for epoch in range(0, config.epochs):
        print("Loop in Epoch: " + str(epoch))
        big_train_dataset = big_train_dataset.shuffle(seed=epoch)
        pre_processed_dataset = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, big_train_dataset)

        pseudo_label_data, silhouette = clust_bert.cluster_and_generate(pre_processed_dataset, device)
        nmi = normalized_mutual_info_score(big_train_dataset["original_label"], pseudo_label_data["labels"])

        wandb_dic = generate_clustering_statistic(clust_bert, pseudo_label_data)
        clust_bert.classifier = None
        clust_bert.classifier = nn.Linear(768, clust_bert.num_labels)
        clust_bert.to(device)

        images_lists = [[] for _ in range(clust_bert.num_labels)]
        for i in range(len(pseudo_label_data)):
            images_lists[int(pseudo_label_data[i]["labels"])].append(i)

        sampler = UnifLabelSampler(int(len(pseudo_label_data)), images_lists)
        data_collator = DataCollatorWithPadding(tokenizer=clust_bert.tokenizer)

        train_dataloader = DataLoader(
            pseudo_label_data, batch_size=2, sampler=sampler, collate_fn=data_collator
        )

        loss = train_loop(clust_bert, train_dataloader, device, config)
        score = eval_loop(clust_bert, device)

        if not args.wandb:
            wandb_dic["loss"] = loss
            wandb_dic["silhouette"] = silhouette
            wandb_dic["cr_score"] = score
            wandb_dic["nmi"] = nmi
            wandb.log(wandb_dic)

    result = evaluate_model(clust_bert.model, sts + senteval_tasks, config.senteval_path)

    if not args.wandb:
        sts_result = [wandb.run.name] + get_sts_from_json(result)
        my_table = wandb.Table(columns=["Id"] + sts, data=[sts_result])
        wandb.log({"STS": my_table})

        senteval_result = [wandb.run.name] + get_senteval_from_json(result)
        my_table = wandb.Table(columns=["Id"] + senteval_tasks, data=[senteval_result])
        wandb.log({"SentEval": my_table})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, help="Nots about the training run")
    parser.add_argument("-k", "--k", type=int, help="Nots about the training run")
    parser.add_argument("-l", "--learning_rate", type=float, help="Nots about the training run")
    parser.add_argument("-s", "--senteval_path", type=str, help="Nots about the training run")
    parser.add_argument("-w", "--wandb", action="store_true", help="Should start the program with Weights & Biases.")

    logging.disable(logging.INFO)  # disable INFO and DEBUG logger everywhere

    args = parser.parse_args()
    start_train(args)
