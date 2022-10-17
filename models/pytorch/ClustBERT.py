import logging
import os
import pickle
import random
from datetime import datetime
from time import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch import Tensor
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from training.PlainPytorchTraining import generate_clustering_statistic

logging.getLogger('matplotlib.pyplot').setLevel(logging.CRITICAL)
logging.getLogger('numba').setLevel(logging.CRITICAL)
import matplotlib.pyplot as plt
import umap
import umap.plot


class ClustBERT(nn.Module):

    def __init__(self, k: int, state="random", pooling="cls"):
        super(ClustBERT, self).__init__()
        if state == "base":
            self.model = BertModel.from_pretrained("bert-base-cased")
        else:
            config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
            self.model = BertModel(config)

        print("Starting ClustBERT with BERT: " + state + ", Pooling: " + pooling)
        self.pooling = pooling

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.loss = nn.CrossEntropyLoss()

        self.num_labels = k
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.num_labels)  # load and initialize weights
        self.kmeans_batch_size = 10 * 1024
        self.clustering = MiniBatchKMeans(
            self.num_labels,
            batch_size=self.kmeans_batch_size,
        )

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):

        if self.pooling == "average":
            output_vector = self.calculate_token_average(input_ids, token_type_ids, attention_mask)
        else:
            output_vector = self.calculate_token_cls(input_ids, token_type_ids, attention_mask)

        output_vectors = self.dropout(output_vector)
        logits = self.classifier(output_vectors)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

    def preprocess_datasets(self, data_set: Dataset) -> Dataset:
        print("Preprocess the data")
        data_set = data_set.map(
            lambda examples: self.tokenizer(examples['new_sentence'], padding=True, truncation=True),
            batched=True)

        data_set.set_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        print("Finished the Preprocess the data")
        return data_set

    def cluster_and_generate(self, data: Dataset, device, table, epoch: int, name="None") -> Tuple[Dataset, dict]:
        print("Start Step 1 --- Clustering")
        t0 = time()
        self.clustering = MiniBatchKMeans(
            self.num_labels,
            random_state=np.random.randint(1234),
            batch_size=self.kmeans_batch_size,
        )
        self.model.eval()

        print("Creating sentence embeddings")

        X_pre_pca = []
        for text in tqdm(data):
            embedding = self.get_sentence_embeddings(device, text)
            X_pre_pca.append(embedding[0].cpu().detach().numpy())

        pca = PCA(n_components=200)
        X_post_pca = pca.fit_transform(X_pre_pca)
        pseudo_labels = self.clustering.fit_predict(X_post_pca)

        data = data.map(lambda example, idx: {"labels": pseudo_labels[idx]}, with_indices=True)

        if table is not None:
            standard_embedding = umap.UMAP(n_components=2, metric='cosine').fit_transform(X_pre_pca)
            indicies = []

            for k in range(self.num_labels):
                results = [idx for idx, t in enumerate(pseudo_labels) if t == k]
                indicies.append(results)

            for idx, idx_list in enumerate(indicies):
                x = standard_embedding[:, 0][idx_list]
                y = standard_embedding[:, 1][idx_list]
                plt.plot(x, y, 'o', label=str(idx))

                results = data.select(idx_list)["text"]
                example_amount = 5
                if len(results) < 5:
                    example_amount = len(results)
                results = random.choices(results, k=example_amount)
                table.add_data(name, str(epoch), results, str(idx))

        dic = generate_clustering_statistic(data)
        dic["UMAP-Pseudo-Labels"] = plt
        if 'original_label' in data.column_names:
            dic["nmi"] = normalized_mutual_info_score(data["original_label"], pseudo_labels)
        dic["silhouette"] = silhouette_score(X_pre_pca, pseudo_labels)

        print("Finished Step 1 --- Clustering in %0.3fs" % (time() - t0))
        return data, dic

    def get_sentence_embeddings(self, device, text):
        if self.pooling == "cls":
            return self.get_sentence_vector_with_cls_token(device, text["input_ids"], text['token_type_ids'],
                                                           text['attention_mask'])
        else:
            return self.get_sentence_vector_with_token_average(device, text["input_ids"], text['token_type_ids'],
                                                               text['attention_mask'])

    def get_sentence_vector_with_token_average(self, device, tokens: Tensor, token_type_ids=None, attention_mask=None):
        with torch.no_grad():
            if len(tokens.shape) is 1:
                tokens = tokens.unsqueeze(0)
                token_type_ids = token_type_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            input_ids = tokens.to(device=device)
            token_type_ids = token_type_ids.to(device=device)
            attention_mask = attention_mask.to(device=device)

            return self.calculate_token_average(input_ids, token_type_ids, attention_mask)

    def calculate_token_average(self, input_ids=None, token_type_ids=None, attention_mask=None):
        out = self.model(input_ids=input_ids,
                         token_type_ids=token_type_ids,
                         attention_mask=attention_mask)

        token_embeddings = out.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        output_vectors = sum_embeddings / sum_mask
        return output_vectors

    def get_sentence_vector_with_cls_token(self, device, tokens: Tensor, token_type_ids=None, attention_mask=None):
        with torch.no_grad():
            if len(tokens.shape) is 1:
                tokens = tokens.unsqueeze(0)
                token_type_ids = token_type_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            input_ids = tokens.to(device=device)
            token_type_ids = token_type_ids.to(device=device)
            attention_mask = attention_mask.to(device=device)

            return self.calculate_token_cls(input_ids, token_type_ids, attention_mask)

    def calculate_token_cls(self, input_ids=None, token_type_ids=None, attention_mask=None):
        out = self.model(input_ids=input_ids,
                         token_type_ids=token_type_ids,
                         attention_mask=attention_mask)

        y = out.pooler_output
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
