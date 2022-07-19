import pytorch_lightning as pl
from torch import nn, optim
from transformers import BertModel, BertTokenizer


class LightningBert(pl.LightningModule):

    def __init__(self, transformer: str, num_labels: int):
        super(LightningBert, self).__init__()
        self.model = BertModel.from_pretrained(transformer, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(transformer)
        self.labels = num_labels

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.labels)  # load and initialize weights
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
