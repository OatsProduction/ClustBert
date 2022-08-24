import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from transformers import Trainer

from models.pytorch.ClustBERT import ClustBERT
from training import DataSetUtils

if __name__ == '__main__':
    wandb.init()
    wandb_logger = WandbLogger()

    train = DataSetUtils.get_million_headlines()
    device = torch.device("cuda:0")
    model = ClustBERT(100, device)

    # setup Trainer
    trainer = Trainer(
        logger=wandb_logger,  # W&B integration
        gpus=-1,  # use all GPU's
        max_epochs=3  # number of epochs
    )

    # train
    trainer.fit(model, train)
