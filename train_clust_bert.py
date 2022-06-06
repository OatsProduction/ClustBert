import argparse

import torch
import wandb

from training import DataSetUtils, PlainPytorchTraining
from training.ClustBert import ClustBERT

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, help='the index of the cuda GPU. Default is 0')
parser.add_argument("--data", type=int, help="define the slice of training data to use. Default is whole dataset")
parser.add_argument("-p", "--plot", help="plot the loss and validation", action="store_true")
parser.add_argument("-s", "--save", help="saves the trained model", action="store_true")
args = parser.parse_args()

cuda_index = "0" if args.cuda is None else str(args.cuda)
device = torch.device('cuda:' + cuda_index if torch.cuda.is_available() else 'cpu')
print("Using device: " + str(device))

snli = DataSetUtils.get_snli_dataset()
if args.data is not None:
    snli = snli.select(range(1, args.data))

wandb.init(project="test-project", entity="clustbert")
clust_bert = ClustBERT(3, device)
wandb.watch(clust_bert)
dataset = clust_bert.preprocess_datasets(snli)

generated_dataset = clust_bert.cluster_and_generate(dataset)
PlainPytorchTraining.start_training(clust_bert, generated_dataset, device)

if args.save:
    clust_bert.save()
if args.plot:
    PlainPytorchTraining.plot()
