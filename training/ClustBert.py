import string

import torch
from datasets import Dataset
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel, TrainingArguments, Trainer
import torch.nn as nn


class ClustBERT(nn.Module):

    def __init__(self):
        super(ClustBERT, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_sentence_vectors_with_token_average(self, texts: list):
        return [self.get_sentence_vector_with_token_average(text) for text in texts]

    def get_sentence_vector_with_token_average(self, text):
        tokens = torch.LongTensor(self.tokenizer.encode(text))
        tokens = tokens.to(device)
        tokens = tokens.unsqueeze(0)

        with torch.no_grad():
            out = clust_bert.model(input_ids=tokens)

        # we only want the hidden_states
        hidden_states = out[2]
        return torch.mean(hidden_states[-1], dim=1).squeeze()


def preload(file: string) -> []:
    returnList = []

    with open(file, 'r', encoding='utf-8') as file:
        data = file.read().rstrip()
        returnList.append(data)
    return returnList


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
clust_bert = ClustBERT().to(device)
clust_bert.model.eval()

result = preload('../dataset/News/train.csv')
test_data = ["This is cool", "believe me.", "Let's try Bert."]
sentence_embedding = clust_bert.get_sentence_vectors_with_token_average(test_data)
X = [sentence.cpu().detach().numpy() for sentence in sentence_embedding]

k = 2
k_means = KMeans(k)
pseudo_labels = k_means.fit_predict(X)

train_encodings = clust_bert.tokenizer(test_data, truncation=True, padding=True)
my_dict = {"0": test_data, "1": ["Think about this."]}
dataset = Dataset.from_dict(my_dict)
print(dataset)

training_args = TrainingArguments(output_dir="test_trainer")
trainer = Trainer(model=clust_bert, args=training_args, train_dataset=dataset)

trainer.train()
