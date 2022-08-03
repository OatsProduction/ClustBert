import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.functional import accuracy
from transformers import BertModel, BertTokenizer, AdamW
from transformers.modeling_outputs import TokenClassifierOutput


class LightningBert(pl.LightningModule):

    def __init__(self, transformer: str, num_labels: int):
        super(LightningBert, self).__init__()
        self.model = BertModel.from_pretrained(transformer, output_hidden_states=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = BertTokenizer.from_pretrained(transformer)
        self.labels = num_labels

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, self.labels)  # load and initialize weights
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state
        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        if labels is not None:
            loss = self.loss(logits.view(-1, self.labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

    def training_step(self, batch, batch_idx):
        self.train()
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        acc = accuracy(predictions, batch["labels"])
        self.log('valid_acc', acc)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-4)
        return optimizer
