import string

import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn


class ClustBERT(nn.Module):

    def __init__(self):
        super(ClustBERT, self).__init__()

        self.encoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_sentence_vector_from_cls(self, text):
        return torch.LongTensor(self.tokenizer.encode(text))
        # return self.tokenizer.convert_ids_to_tokens(ids)


def preload(file: string) -> []:
    returnList = []

    with open(file, 'r', encoding='utf-8') as file:
        data = file.read().rstrip()
        returnList.append(data)
    return returnList


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
result = preload('../dataset/News/train.csv')
model = ClustBERT().to(device)
model.encoder.eval()

tokens = model.get_sentence_vector_from_cls("This is cool")
tokens = tokens.to(device)
tokens = tokens.unsqueeze(0)

with torch.no_grad():
    out = model.encoder(input_ids=tokens)

# the output is a tuple
print(type(out))
# the tuple contains three elements as explained above)
print(out)
# we only want the hidden_states
hidden_states = out[2]
print(len(hidden_states))
