import argparse
import logging

import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepcluster.alexnet import init_alexnet
from models.pytorch.ClustBERT import ClustBERT
from training import DataSetUtils

logging.getLogger('matplotlib.pyplot').setLevel(logging.CRITICAL)
logging.getLogger('numba').setLevel(logging.CRITICAL)
import matplotlib.pyplot as plt
import umap.plot


def compute_features(dataloader, model, N):
    print('Compute features')
    model.eval()
    # discard the label information in the dataloader
    for i, batch in enumerate(dataloader):
        batch = {k: v for k, v in batch.items()}
        aux = model(batch).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

    return features


def generate_alexnet_embeddings(dataset: Dataset):
    alexnet = init_alexnet()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    dataset = dataset.map(pre_process, load_from_cache_file=False)
    dataset.set_format("torch", columns=['img'])

    dataloader = DataLoader(dataset, batch_size=2, pin_memory=True)

    return compute_features(dataloader, alexnet, len(dataset))


def pre_process(data_point):
    # data_point["img"] = transforms.Resize(256)(data_point['img'])
    data_point["img"] = transforms.ToTensor()(data_point['img'])

    return data_point


def generate_bert_embeddings(dataset: Dataset):
    device = torch.device("cuda:0")
    clust_bert = ClustBERT(10, state="seq", pooling="average")
    clust_bert.to(device)
    dataset = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, dataset)

    X = []
    for text in tqdm(dataset):
        embedding = clust_bert.get_sentence_embeddings(device, text)
        X.append(embedding[0].cpu().detach().numpy())

    return X


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='the index of the cuda GPU. Default is 0')
    args = parser.parse_args()

    # wandb.init(project="ClustBert")

    dataset = load_dataset("cifar10")
    dataset = dataset["train"]
    dataset = dataset.select(range(1, 2000))
    X = generate_alexnet_embeddings(dataset)

    standard_embedding = umap.UMAP().fit_transform(X)
    x = standard_embedding[:, 0]
    y = standard_embedding[:, 1]
    plt.plot(x, y, 'o')
    plt.show()
