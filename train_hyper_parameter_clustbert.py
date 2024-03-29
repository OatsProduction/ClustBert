import argparse
import logging

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from evaluate_model import evaluate_model, sts, senteval_tasks
from evaluation.print_evaluation import get_senteval_from_json, get_sts_from_json
from models.pytorch.ClustBERT import ClustBERT
from training import DataSetUtils
from training.PlainPytorchTraining import train_loop, eval_loop, UnifLabelSampler


def start_train(config=None):
    device = torch.device(config.device)
    torch.cuda.empty_cache()

    bert_model = "base" if config is None or config.model is None else config.model
    bert_embedding = "average" if config is None or config.embedding is None else config.embedding
    clust_bert = ClustBERT(config.k, state=bert_model, pooling=bert_embedding)
    clust_bert.to(device)

    if not config.wandb:
        wandb.init(config=config)
        wandb.config.update(config)
        config = wandb.config
        wandb.run.name = "lr" + str(config.learning_rate) + "_k" + str(config.k) + \
                         "_epoch" + str(config.epochs) + "_bert_" + bert_model + \
                         "_embedding_" + bert_embedding + "_" + wandb.run.id
        wandb.watch(clust_bert)

    if config.dataset == "trec":
        big_train_dataset = DataSetUtils.get_tec().shuffle(seed=525)
    elif config.dataset == "imdb":
        big_train_dataset = DataSetUtils.get_imdb().shuffle(seed=525)
    else:
        big_train_dataset = DataSetUtils.get_million_headlines().shuffle(seed=525)

    table = None
    name = None

    if not config.wandb:
        columns = ["Id", "Epoch", "Texts", "Cluster"]
        table = wandb.Table(columns=columns)
        name = wandb.run.name
        score = eval_loop(clust_bert, device)
        wandb.log({
            "cr_score": score
        })

    for epoch in range(0, config.epochs):
        print("Loop in Epoch: " + str(epoch))
        big_train_dataset = big_train_dataset.shuffle(seed=epoch)
        pre_processed_dataset = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, big_train_dataset)

        pseudo_label_data, wandb_dic = clust_bert.cluster_and_generate(pre_processed_dataset, device, table, epoch,
                                                                       name=name)

        clust_bert.classifier = None
        clust_bert.classifier = nn.Linear(768, clust_bert.num_labels)
        clust_bert.to(device)

        texts_lists = [[] for _ in range(clust_bert.num_labels)]
        for i in range(len(pseudo_label_data)):
            texts_lists[int(pseudo_label_data[i]["labels"])].append(i)

        sampler = UnifLabelSampler(int(len(pseudo_label_data)), texts_lists)
        data_collator = DataCollatorWithPadding(tokenizer=clust_bert.tokenizer)

        train_dataloader = DataLoader(
            pseudo_label_data, batch_size=8, sampler=sampler, collate_fn=data_collator
        )

        loss = train_loop(clust_bert, train_dataloader, device, config)
        score = eval_loop(clust_bert, device)

        if not config.wandb:
            wandb_dic["loss"] = loss
            wandb_dic["cr_score"] = score
            wandb.log(wandb_dic)

    result = evaluate_model(clust_bert, sts + senteval_tasks)
    sts_result = [wandb.run.name] + get_sts_from_json(result, sts)
    print("STS - Results")
    print(sts)
    print(sts_result)

    senteval_result = [wandb.run.name] + get_senteval_from_json(result, senteval_tasks)
    print("STS - Results")
    print(senteval_tasks)
    print(senteval_result)

    if not config.wandb:
        my_table = wandb.Table(columns=["Id"] + sts, data=[sts_result])
        wandb.log({"STS": my_table})

        my_table = wandb.Table(columns=["Id"] + senteval_tasks, data=[senteval_result])
        wandb.log({"SentEval": my_table})

        wandb.log({"Example-Texts": table})

    if config.save:
        clust_bert.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, help="Define the amount of Epochs to train with.")
    parser.add_argument("-k", "--k", type=int, help="Define the amount of clusters for the clustering algorithm.")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Define the device to use.")
    parser.add_argument("-l", "--learning_rate", type=float, help="Define the learning rate for the run.")
    parser.add_argument("-em", "--embedding", type=str, help="Define the pooling to generate sentence embeddings. "
                                                             "Either cls or average.")
    parser.add_argument("-m", "--model", type=str, help="Define the BERT model to use. Either base or random.")
    parser.add_argument("-ds", "--dataset", type=str, help="Define the dataset to use. Either trec, imdb or headlines.")
    parser.add_argument("-w", "--wandb", action="store_true", help="Should start the program without Weights & Biases.")
    parser.add_argument("-s", "--save", action="store_true", help="Save the model.")

    logger = logging.getLogger(__name__)
    logging.disable(logging.DEBUG)  # disable INFO and DEBUG logger everywhere

    args = parser.parse_args()
    start_train(args)
