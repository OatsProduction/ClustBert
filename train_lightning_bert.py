import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from models.lightning.LightningBert import LightningBert
from training import DataSetUtils

parser = argparse.ArgumentParser()
parser.add_argument("--accelerator", default="gpu")
parser.add_argument("--devices", default=1)
parser.add_argument("--data", type=int, help="define the slice of training data to use. Default is whole dataset")
parser.add_argument("-s", "--save", help="saves the trained model", action="store_true")
args = parser.parse_args()

pl.seed_everything(1)

train, valid = DataSetUtils.get_snli_dataset()
if args.data is not None:
    train = train.select(range(1, args.data))
    valid = valid.select(range(1, args.data))

bert = LightningBert("bert-base-cased", 3)
train = DataSetUtils.preprocess_datasets(bert.tokenizer, train)
valid = DataSetUtils.preprocess_datasets(bert.tokenizer, valid)

data_collator = DataCollatorWithPadding(tokenizer=bert.tokenizer)
train_dataloader = DataLoader(train, num_workers=4, shuffle=True, batch_size=16, collate_fn=data_collator)
eval_dataloader = DataLoader(valid, num_workers=4, batch_size=16, collate_fn=data_collator)

callbacks = [
    ModelCheckpoint(dirpath="checkpoints", every_n_train_steps=100)
]

wandb_logger = WandbLogger()
wandb_logger.watch(bert)
trainer = pl.Trainer(limit_train_batches=100,
                     max_epochs=2,
                     logger=wandb_logger,
                     accelerator=args.accelerator,
                     devices=args.devices)
trainer.fit(model=bert, train_dataloaders=train_dataloader, val_dataloaders=eval_dataloader)
