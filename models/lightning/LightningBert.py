import pytorch_lightning as pl
import torch
from torch import nn, optim
from transformers import BertModel, BertTokenizer
from transformers.modeling_outputs import TokenClassifierOutput


class LightningBert(pl.LightningModule):

    def __init__(self, transformer: str, num_labels: int):
        super(LightningBert, self).__init__()
        self.model = BertModel.from_pretrained(transformer, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(transformer)
        self.labels = num_labels

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, self.labels)  # load and initialize weights
        self.save_hyperparameters()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state
        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

    def training_step(self, batch, batch_idx):
        batch = {k: v for k, v in batch.items()}
        outputs = self(**batch)
        loss = outputs.loss

        return loss

    def validation_step(self, batch, batch_idx):
        batch = {k: v for k, v in batch.items()}
        outputs = self(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.log("val_loss", logits)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
