import os
import pickle
from datetime import datetime

import torch.nn as nn
from transformers import BertTokenizer
from transformers.modeling_outputs import TokenClassifierOutput


class ClassifierTransformer(nn.Module):

    def __init__(self, transformer, device):
        super(ClassifierTransformer, self).__init__()
        self.model = transformer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.device = device

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 3)  # load and initialize weights
        self.to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state
        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

    def save(self):
        date = str(datetime.now())
        filename = 'plain_bert_' + date + ".model"
        if not os.path.exists("output"):
            os.mkdir("output")
        pickle.dump(self, open("output/" + filename, 'wb'))

    @staticmethod
    def load(file_name: str):
        return pickle.load(open(file_name, 'rb'))
