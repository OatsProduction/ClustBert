import argparse
import logging

import senteval
import torch
import wandb as wandb
from transformers import BertTokenizer

from evaluation.print_evaluation import get_senteval_from_json, get_sts_from_json
from models.pytorch.ClustBERT import ClustBERT


def prepare(params, samples):
    return


def batcher(params, batch):
    sentences = [" ".join(s).lower() for s in batch]

    with torch.no_grad():
        y = params['tokenizer'](sentences, padding=True, truncation=True, return_tensors="pt")
        y = params['model'].get_sentence_embeddings(params["device"], y.data)
    return y


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
sts = ["STS12", "STS13", "STS14", "STS15", "STS16"]
senteval_tasks = ["MR", "CR", "SUBJ", "MPQA", "SST2", "SST5", "TREC", "MRPC"]


def evaluate_model(transformer, tasks, senteval_path):
    transformer.eval()
    for parameter in transformer.parameters():
        parameter.requires_grad = False

    dev = torch.device("cpu")
    transformer.to(dev)

    params = {
        'model': transformer,
        'tokenizer': BertTokenizer.from_pretrained("bert-base-cased"),
        'task_path': senteval_path,
        "device": dev,
        'usepytorch': True,
        'kfold': 10,
        'classifier': {
            'nhid': 0,
            'optim': 'adam',
            'batch_size': 64,
            'tenacity': 5,
            'epoch_size': 4
        }
    }
    se = senteval.engine.SE(params, batcher, prepare)

    return se.eval(tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="define the path to model to load")
    parser.add_argument("--sts", help="perform all STS", action="store_true")
    parser.add_argument("--senteval", help="perform all SentEval Tests", action="store_true")
    parser.add_argument("--all", help="perform all evaluation Tests", action="store_true")
    parser.add_argument("--senteval_path", type=str, help="define the path to model to load")
    parser.add_argument('--device', type=str, help='the device used by the program. Default is cuda:0')
    args = parser.parse_args()

    wandb.init(config=args)
    config = wandb.config
    bert_type = "random"
    wandb.run.name = "BERT_" + bert_type + "_average" + "_" + wandb.run.id

    device = "cuda:0" if args.device is None else str(args.device)
    print("Started the evaluation script with the device: " + str(device))

    clust_bert = ClustBERT(10, state=bert_type, pooling="average")
    clust_bert.to(device)

    result = evaluate_model(clust_bert, sts + senteval_tasks, config.senteval_path)

    sts_result = [wandb.run.name] + get_sts_from_json(result, sts)
    my_table = wandb.Table(columns=["Id"] + sts, data=[sts_result])
    wandb.log({"STS": my_table})

    senteval_result = [wandb.run.name] + get_senteval_from_json(result, senteval_tasks)
    my_table = wandb.Table(columns=["Id"] + senteval_tasks, data=[senteval_result])
    wandb.log({"SentEval": my_table})
