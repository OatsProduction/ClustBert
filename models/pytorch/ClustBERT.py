import os
import pickle
from datetime import datetime
from time import time
from typing import Tuple

import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torch import Tensor
from transformers import BertTokenizer, BertModel, BertConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.modeling_utils import no_init_weights


class ClustBERT(nn.Module):

    def __init__(self, k: int):
        super(ClustBERT, self).__init__()
        no_init_weights(True)
        config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
        self.model = BertModel(config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.loss = nn.CrossEntropyLoss()

        self.num_labels = k
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.num_labels)  # load and initialize weights
        self.kmeans_batch_size = 5 * 1024
        self.clustering = MiniBatchKMeans(
            self.num_labels,
            batch_size=self.kmeans_batch_size,
        )

    # def __init__(self, config):
    #     super(ClustBERT, self).__init__()
    #     self.model = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
    #     self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    #
    #     self.num_labels = config.k
    #     self.dropout = nn.Dropout(0.1)
    #     self.classifier = nn.Linear(768, self.num_labels)  # load and initialize weights
    #     self.clustering = KMeans(self.num_labels)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state
        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

    def preprocess_datasets(self, data_set: Dataset) -> Dataset:
        print("Preprocess the data")
        data_set = data_set.map(
            lambda examples: self.tokenizer(examples['new_sentence'], padding=True, truncation=True),
            batched=True)

        data_set.set_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        print("Finished the Preprocess the data")
        return data_set

    def cluster_and_generate(self, data: Dataset, device) -> Tuple[Dataset, float]:
        print("Start Step 1 --- Clustering")
        t0 = time()
        self.clustering = MiniBatchKMeans(
            self.num_labels,
            batch_size=self.kmeans_batch_size,
        )
        self.model.eval()

        sentence_embedding = self.get_sentence_vectors_with_cls_token(device, data)
        X = [sentence.cpu().detach().numpy() for sentence in sentence_embedding]
        pca = PCA(n_components=100)
        X = pca.fit_transform(X)

        pseudo_labels = self.clustering.fit_predict(X)

        silhouette = silhouette_score(X, pseudo_labels)
        data = data.map(lambda example, idx: {"labels": pseudo_labels[idx]}, with_indices=True)

        print("Finished Step 1 --- Clustering in %0.3fs" % (time() - t0))
        return data, silhouette

    def get_sentence_vectors_with_cls_token(self, device, texts: Dataset):
        lists = []
        for text in texts:
            embedding = self.get_sentence_vector_with_cls_token(device, text["input_ids"], text['token_type_ids'],
                                                                text['attention_mask'])
            lists.append(embedding.cpu().detach())
        return lists

    def get_sentence_vectors_with_token_average(self, device, texts: Dataset):
        return [self.get_sentence_vector_with_token_average(device, text["input_ids"], text['token_type_ids'],
                                                            text['attention_mask']) for text in texts]

    def get_sentence_vector_with_token_average(self, device, tokens: Tensor, token_type_ids=None, attention_mask=None):
        with torch.no_grad():
            out = self.model(input_ids=tokens.unsqueeze(0).to(device=device),
                             token_type_ids=token_type_ids.unsqueeze(0).to(device=device),
                             attention_mask=attention_mask.unsqueeze(0).to(device=device))

        # we only want the hidden_states
        hidden_states = out[2]
        return torch.mean(hidden_states[-1], dim=1).squeeze()

    def get_sentence_vector_with_cls_token(self, device, tokens: Tensor, token_type_ids=None, attention_mask=None):
        with torch.no_grad():
            out = self.model(input_ids=tokens.unsqueeze(0).to(device=device),
                             token_type_ids=token_type_ids.unsqueeze(0).to(device=device),
                             attention_mask=attention_mask.unsqueeze(0).to(device=device))

        y = out[0]  # outputs[0]=last hidden state
        y = y[:, 0, :]
        y = y.squeeze(0)

        return y

    def save(self):
        date = str(datetime.now())
        filename = 'clust_bert_' + date + "_k_" + str(self.num_labels) + ".model"
        if not os.path.exists("output"):
            os.mkdir("output")
        pickle.dump(self.model, open("output/" + filename, 'wb'))

    @staticmethod
    def load(file_name: str):
        return pickle.load(open(file_name, 'rb'))
