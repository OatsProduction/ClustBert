import argparse
import logging

import numpy as np
import torch
import torchvision.transforms as transforms
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader

from deepcluster.alexnet import alexnet

logging.getLogger('matplotlib.pyplot').setLevel(logging.CRITICAL)
logging.getLogger('numba').setLevel(logging.CRITICAL)
import matplotlib.pyplot as plt
import umap.plot


def compute_features(dataloader, model, N):
    if args.verbose:
        print('Compute features')
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='the index of the cuda GPU. Default is 0')
    args = parser.parse_args()

    wandb.init(project="ClustBert")
    device = torch.device("cuda:0")
    print("Using device: " + str(device))
    alexNet = alexnet()
    alexNet.to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    # load the data
    dataset = load_dataset("imagenet-1k", split="validation")
    dataset = dataset.select(range(1, 10000))
    dataloader = DataLoader(dataset, batch_size=8, pin_memory=True)

    embeddings = compute_features(dataloader, alexNet, len(dataset))

    standard_embedding = umap.UMAP().fit_transform(embeddings)
    x = standard_embedding[:, 0]
    y = standard_embedding[:, 1]
    plt.plot(x, y, 'o')
    wandb.log({"UMAP-Pseudo-Labels": plt})
