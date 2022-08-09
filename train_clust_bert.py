import argparse

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from models.pytorch.ClustBert import ClustBERT
from training import DataSetUtils
from training.PlainPytorchTraining import train_loop, avg_train_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='the index of the cuda GPU. Default is cuda:0')
    parser.add_argument("--data", type=int, help="define the slice of training data to use. Default is whole dataset")
    parser.add_argument("-s", "--save", help="saves the trained model", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.device is None else args.device)
    print("Using device: " + str(device))

    train = DataSetUtils.get_million_headlines()
    if args.data is not None:
        train = train.select(range(1, args.data))

    clust_bert = ClustBERT(100, device)
    train = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, train)

    wandb.init(project="test-project", entity="clustbert")
    wandb.watch(clust_bert)

    data_collator = DataCollatorWithPadding(tokenizer=clust_bert.tokenizer)
    num_epochs = 12

    for epoch in range(num_epochs):
        print("Loop in Epoch: " + str(epoch))
        pseudo_label_data = clust_bert.cluster_and_generate(train)
        # nmi = normalized_mutual_info_score(train["labels"], pseudo_label_data["labels"])

        train_dataloader = DataLoader(
            pseudo_label_data, shuffle=True, batch_size=8, collate_fn=data_collator
        )
        train_loop(clust_bert, device, train_dataloader)

        wandb.log({"loss": avg_train_loss})

    if args.save:
        clust_bert.save()
