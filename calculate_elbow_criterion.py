import argparse

from pytorch_lightning.loggers import wandb
from sklearn.cluster import MiniBatchKMeans
from wandb.wandb_torch import torch

from models.pytorch.ClustBERT import ClustBERT
from training import DataSetUtils
from training.DataSetUtils import get_million_headlines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, help='the index of the cuda GPU. Default is 0')
    args = parser.parse_args()

    wandb.init(project="test-project", entity="clustbert")

    cuda_index = "0" if args.cuda is None else str(args.cuda)
    device = torch.device('cuda:' + cuda_index if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    clust_bert = ClustBERT(100)
    clust_bert.to(device)

    dataset = get_million_headlines()
    dataset = DataSetUtils.preprocess_datasets(clust_bert.tokenizer, dataset)

    sentence_embedding = clust_bert.get_sentence_vectors_with_cls_token(device, dataset)
    X = [sentence.cpu().detach().numpy() for sentence in sentence_embedding]

    for k in range(10000):
        kmean_model = MiniBatchKMeans(n_clusters=k)
        kmean_model.fit(X)
        wandb.log({"inertias": kmean_model.inertia_})
