import argparse
import json
import logging
import pickle

import senteval
import torch
from transformers import BertModel, BertTokenizer, BertConfig


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
sts = "STS12,STS13,STS14,STS15,STS16"
seneval_tasks = "MR,CR,SUBJ,MPQA,SST2,SST5,TREC,MRPC"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="define the path to model to load")
    parser.add_argument("--sts", help="perform all STS", action="store_true")
    parser.add_argument("--senteval", help="perform all SentEval Tests", action="store_true")
    parser.add_argument("--all", help="perform all evaluation Tests", action="store_true")
    parser.add_argument("--senteval_path", type=str, help="define the path to model to load")
    parser.add_argument('--device', type=str, help='the device used by the program. Default is cuda:0')
    args = parser.parse_args()

    device = "cuda:0" if args.device is None else str(args.device)
    print("Started the evaluation script with the device: " + str(device))

    if args.model == "random":
        model_name = args.model
        config = BertConfig.from_pretrained("bert-base-cased")
        transformer = BertModel(config)
    elif args.model is not None:
        model_name = args.model
        transformer = pickle.load(open("../output/" + args.model, 'rb'))
    else:
        model_name = "plain_bert"
        transformer = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)

    transformer.eval()
    transformer.to(device=device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    senteval_path = args.senteval_path if args.senteval_path is not None else '../../SentEval/data'
    params = {
        'task_path': senteval_path,
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

    if args.sts:
        transfer_tasks = sts.split(",")
    if args.senteval:
        transfer_tasks = seneval_tasks.split(",")
    if args.all:
        transfer_tasks = (sts + "," + seneval_tasks).split(",")
    else:
        transfer_tasks = ["SNLI"]

    results = se.eval(transfer_tasks)

    with open(model_name.removesuffix(".model") + '_evaluation_results.json', 'w') as outfile:
        json_object = json.dumps(results, indent=4)
        outfile.write(json_object)
