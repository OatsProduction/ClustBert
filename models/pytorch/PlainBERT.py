import os
import pickle
from datetime import datetime

import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
from transformers.modeling_outputs import TokenClassifierOutput


class ClassifierTransformer(nn.Module):

    def __init__(self, transformer: str, num_labels: int, device):
        super(ClassifierTransformer, self).__init__()
        self.model = BertModel.from_pretrained(transformer, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(transformer)
        self.device = device
        self.labels = num_labels

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.labels)  # load and initialize weights
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
            loss = loss_fct(logits.view(-1, self.labels), labels.view(-1))

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


def get_random_bert():
    config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True, gradient_checkpointing=False,
                                        pruned_heads=
                                        {
                                            0: list(range(12)),
                                            1: list(range(12)),
                                            2: list(range(12)),
                                            3: list(range(12)),
                                            4: list(range(12)),
                                            5: list(range(12)),
                                            6: list(range(12)),
                                        })
    return BertModel(config)
