import argparse

import torch
import wandb
from sklearn.cluster import MiniBatchKMeans

from models.pytorch.ClustBERT import ClustBERT
from training import DataSetUtils
from training.DataSetUtils import get_million_headlines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='the index of the cuda GPU. Default is 0')
    args = parser.parse_args()

    wandb.init(project="ClustBert")
    device = torch.device("cpu")
    print("Using device: " + str(device))
    clust_bert = ClustBERT(100)
    wandb.watch(clust_bert)

    dataset = get_million_headlines().shuffle(seed=525)
    dataset = dataset.select(range(1, 100000))
    dataset = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, dataset)

    sentence_embedding = clust_bert.get_sentence_embeddings(device, dataset)
    X = [sentence.cpu().detach().numpy() for sentence in sentence_embedding]

    for k in range(1, 10000):
        kmeans_batch_size = 10 * 1024
        kmean_model = MiniBatchKMeans(
            k, batch_size=kmeans_batch_size,
        )

        kmean_model.fit(X)
        wandb.log({"inertias": kmean_model.inertia_})
