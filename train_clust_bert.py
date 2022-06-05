import torch

from training import DataSetUtils, PlainPytorchTraining
from training.ClustBert import ClustBERT

cuda_index = str(input("Which Cuda to use ? "))
device = torch.device('cuda:' + cuda_index if torch.cuda.is_available() else 'cpu')

snli = DataSetUtils.get_snli_dataset()
clust_bert = ClustBERT(3, device)
dataset = clust_bert.preprocess_datasets(snli)

generated_dataset = clust_bert.cluster_and_generate(dataset)
PlainPytorchTraining.start_training(clust_bert, generated_dataset)

clust_bert.save()
PlainPytorchTraining.plot()
