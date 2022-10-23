import argparse
import logging

import numpy as np
import torch
import torchvision as torchvision
import torchvision.transforms as transforms
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepcluster.alexnet import init_alexnet
from models.pytorch.ClustBERT import ClustBERT
from training import DataSetUtils

logging.getLogger('matplotlib.pyplot').setLevel(logging.CRITICAL)
logging.getLogger('numba').setLevel(logging.CRITICAL)
import matplotlib.pyplot as plt
import umap.plot


def generate_alexnet_embeddings(dataset: Dataset):
    alexnet = init_alexnet(out=10)
    alexnet.cuda()

    batches = 20
    N = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batches, pin_memory=True)
    print('Start - Compute features')
    alexnet.eval()

    for i, (input_tensor, _) in enumerate(tqdm(dataloader)):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = alexnet(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * batches: (i + 1) * batches] = aux
        else:
            # special treatment for final batch
            features[i * batches:] = aux

    print('Done - Compute features')
    return features


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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]
    # dataset = torchvision.datasets.CIFAR10('/home/willem/.cache/masterarbeit/datasets/cifar10', download=True,
    dataset = torchvision.datasets.Flowers102('/home/willem/.cache/masterarbeit/datasets/flower', download=True,
                                              transform=transforms.Compose(tra))
    X = generate_alexnet_embeddings(dataset)

    embedding = umap.UMAP().fit_transform(X)
    plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title('UMAP projection of the Digits dataset', fontsize=24)
    plt.show()
