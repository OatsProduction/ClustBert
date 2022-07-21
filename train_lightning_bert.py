import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from models.lightning.LightningBert import LightningBert
from training import DataSetUtils

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help='the number of GPUs to use. Default is 1')
parser.add_argument("--data", type=int, help="define the slice of training data to use. Default is whole dataset")
parser.add_argument("-s", "--save", help="saves the trained model", action="store_true")
args = parser.parse_args()

pl.seed_everything(1)
bert = LightningBert("bert-base-cased", 3)

wandb_logger = WandbLogger()
wandb_logger.watch(bert)

train, valid = DataSetUtils.get_snli_dataset()
if args.data is not None:
    train = train.select(range(1, args.data))
    valid = valid.select(range(1, args.data))

data_collator = DataCollatorWithPadding(tokenizer=bert.tokenizer)
train_dataloader = DataLoader(train, shuffle=True, batch_size=16, collate_fn=data_collator)
eval_dataloader = DataLoader(valid, batch_size=16, collate_fn=data_collator)

callbakcks = [
    ModelCheckpoint(dirpath = "checkpoints", every_n_train_steps=100)
]

trainer = pl.Trainer(limit_train_batches=100, max_epochs=2, logger=wandb_logger, accelerator="cpu", devices=8,
                     strategy="ddp")
trainer.fit(model=bert, train_dataloaders=train_dataloader, val_dataloaders=eval_dataloader)
