import argparse
import logging
import pickle

import senteval
import torch
from transformers import BertModel, BertTokenizer


def prepare(params, samples):
    return


def batcher(params, batch):
    sentences = [" ".join(s).lower() for s in batch]

    with torch.no_grad():
        y = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        y = y.to(device=device)
        y = transformer(y)[0]
        y = y[:, 0, :].view(-1, 768)

    return y


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="define the path to model to load")
    parser.add_argument("--sts", help="perform all STS", action="store_true")
    parser.add_argument("--senteval", help="perform all SentEval Tests", action="store_true")
    parser.add_argument("--senteval_path", type=str, help="define the path to model to load")
    parser.add_argument('--device', type=str, help='the device used by the program. Default is cuda:0')
    args = parser.parse_args()

    device = "cuda:0" if args.device is None else str(args.device)
    print("Started the evaluation script with the device: " + str(device))

    if args.model is not None:
        transformer = pickle.load(open(args.model, 'rb'))
        transformer.to(device=device)
    else:
        transformer = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        transformer.eval()
        transformer.to(device=device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    senteval_path = args.senteval_path if args.senteval_path is not None else '../../SentEval/data'
    params = {
        'task_path': senteval_path,
        'usepytorch': True,
        'kfold': 5
    }
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                            'tenacity': 3, 'epoch_size': 2}

    se = senteval.engine.SE(params, batcher, prepare)

    if args.sts:
        sts = "STS12,STS13,STS14,STS15,STS16"
        transfer_tasks = sts.split(",")
    if args.senteval:
        seneval_tasks = "MR,CR,SUBJ,MPQA,SST2,SST5,TREC,MRPC"
        transfer_tasks = seneval_tasks.split(",")
    else:
        transfer_tasks = ["SNLI"]

    print("Started the tasks")

    results = se.eval(transfer_tasks)
    print(results)
