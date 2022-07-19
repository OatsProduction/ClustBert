import argparse

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from lightning.LightningBert import LightningBert
from training import DataSetUtils

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, help='the index of the cuda GPU. Default is 0')
parser.add_argument("--data", type=int, help="define the slice of training data to use. Default is whole dataset")
parser.add_argument("-s", "--save", help="saves the trained model", action="store_true")
args = parser.parse_args()

bert = LightningBert("bert-base-cased", 3)

wandb.init(project="test-project", entity="clustbert")
wandb_logger = WandbLogger()
wandb_logger.watch(bert)

train, valid = DataSetUtils.get_snli_dataset()
if args.data is not None:
    train = train.select(range(1, args.data))
    valid = valid.select(range(1, args.data))

data_collator = DataCollatorWithPadding(tokenizer=bert.tokenizer)
train_dataloader = DataLoader(train, shuffle=True, batch_size=16, collate_fn=data_collator)
eval_dataloader = DataLoader(valid, batch_size=16, collate_fn=data_collator)

trainer = pl.Trainer(limit_train_batches=100, max_epochs=2, logger=wandb_logger)
trainer.fit(model=bert, train_dataloaders=train_dataloader, val_dataloaders=eval_dataloader)
