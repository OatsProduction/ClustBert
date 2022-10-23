import argparse
import logging

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from models.pytorch.ClustBERT import ClustBERT
from training import DataSetUtils

logging.getLogger('matplotlib.pyplot').setLevel(logging.CRITICAL)
logging.getLogger('numba').setLevel(logging.CRITICAL)
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='the device used by the program. Default is cuda:0')
    args = parser.parse_args()

    device = "cuda:0" if args is None or args.device is None else str(args.device)
    print("Using device: " + str(device))
    clust_bert = ClustBERT(10, state="seq", pooling="average")
    clust_bert.to(device)

    dataset = DataSetUtils.get_imdb().shuffle(seed=525)
    dataset = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, dataset)

    X = []
    for text in tqdm(dataset):
        embedding = clust_bert.get_sentence_embeddings(device, text)
        X.append(embedding[0].cpu().detach().numpy())

    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
