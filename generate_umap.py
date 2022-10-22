import argparse
import logging

import torch
from tqdm import tqdm

from models.pytorch.ClustBERT import ClustBERT
from training import DataSetUtils

logging.getLogger('matplotlib.pyplot').setLevel(logging.CRITICAL)
logging.getLogger('numba').setLevel(logging.CRITICAL)
import matplotlib.pyplot as plt
import umap.plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='the index of the cuda GPU. Default is 0')
    args = parser.parse_args()

    device = torch.device("cuda:0")
    print("Using device: " + str(device))
    clust_bert = ClustBERT(10, state="seq", pooling="average")
    clust_bert.to(device)

    dataset = DataSetUtils.get_pedia_classes().shuffle(seed=525)
    dataset = dataset.select(range(1, 20000))
    dataset = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, dataset)

    X = []
    for text in tqdm(dataset):
        embedding = clust_bert.get_sentence_embeddings(device, text)
        X.append(embedding[0].cpu().detach().numpy())

    standard_embedding = umap.UMAP().fit_transform(X)
    x = standard_embedding[:, 0]
    y = standard_embedding[:, 1]
    plt.plot(x, y, 'o')
    plt.show()
