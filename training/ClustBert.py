import string

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn


class ClustBERT(nn.Module):

    def __init__(self):
        super(ClustBERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, text, label):
        print(self.encoder(text))
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea

    def get_sentence_vector_from_cls(self, text):
        ids = self.tokenizer.encode(text)
        return self.tokenizer.convert_ids_to_tokens(ids)


def preload(file: string) -> []:
    returnList = []

    with open(file, 'r', encoding='utf-8') as file:
        data = file.read().rstrip()
        returnList.append(data)
    return returnList


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
result = preload('../dataset/News/train.csv')
model = ClustBERT().to(device)

print(model.get_sentence_vector_from_cls("This is cool"))

