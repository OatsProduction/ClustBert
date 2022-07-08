import argparse

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import BertModel, DataCollatorWithPadding

from models.PlainBERT import ClassifierTransformer
from training import DataSetUtils, PlainPytorchTraining

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, help='the index of the cuda GPU. Default is 0')
parser.add_argument("--data", type=int, help="define the slice of training data to use. Default is whole dataset")
parser.add_argument("-s", "--save", help="saves the trained model", action="store_true")
args = parser.parse_args()

cuda_index = "0" if args.cuda is None else str(args.cuda)
device = torch.device('cuda:' + cuda_index if torch.cuda.is_available() else 'cpu')
print("Using device: " + str(device))

train, valid = DataSetUtils.get_snli_dataset()
if args.data is not None:
    train = train.select(range(1, args.data))
    valid = valid.select(range(1, args.data))

model = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
bert = ClassifierTransformer(model, device)
train = DataSetUtils.preprocess_datasets(bert.tokenizer, train)
valid = DataSetUtils.preprocess_datasets(bert.tokenizer, valid)

wandb.init(project="test-project", entity="clustbert")
wandb.watch(bert)

data_collator = DataCollatorWithPadding(tokenizer=bert.tokenizer)
train_dataloader = DataLoader(train, shuffle=True, batch_size=16, collate_fn=data_collator)
eval_dataloader = DataLoader(valid, batch_size=16, collate_fn=data_collator)
num_epochs = 14

for epoch in range(num_epochs):
    print("Start Epoch: " + str(epoch))
    PlainPytorchTraining.train_loop(bert, device, train_dataloader)
    PlainPytorchTraining.eval_loop(bert, device, eval_dataloader)

    wandb.log({"loss": PlainPytorchTraining.avg_train_loss, "validation": PlainPytorchTraining.avg_val_loss})

if args.save:
    bert.save()
