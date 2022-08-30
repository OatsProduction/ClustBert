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
        # y = torch.rand(len(sentences), 768)
        y = params['tokenizer'](sentences, max_length=512, padding=True, truncation=True, return_tensors="pt")[
            "input_ids"]
        y = params['model'](y)[0]
        y = y[:, 0, :].view(-1, 768)

    return y


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
sts = "STS12,STS13,STS14,STS15,STS16"
seneval_tasks = "MR,CR,SUBJ,MPQA,SST2,SST5,TREC,MRPC"


def evaluate_model(transformer, tasks, senteval_path):
    transformer.eval()

    params = {
        'model': transformer,
        'tokenizer': BertTokenizer.from_pretrained("bert-base-cased"),
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

    device = "cuda:0" if args.device is None else str(args.device)
    print("Started the evaluation script with the device: " + str(device))

    if args.model == "random":
        model_name = args.model
        config = BertConfig.from_pretrained("bert-base-cased", pruned_heads=
        {
            0: list(range(12)),
            1: list(range(12)),
            2: list(range(12)),
            3: list(range(12)),
            4: list(range(12)),
            5: list(range(12)),
            6: list(range(12)),
        })
        model = BertModel(config)
        # model.init_weights()
    elif args.model is not None:
        model_name = args.model
        model = pickle.load(open("../output/" + args.model, 'rb'))
    else:
        model_name = "plain_bert"
        model = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)

    if args.sts:
        transfer_tasks = sts.split(",")
    if args.senteval:
        transfer_tasks = seneval_tasks.split(",")
    if args.all:
        transfer_tasks = (sts + "," + seneval_tasks).split(",")
    else:
        transfer_tasks = ["STS13"]

    results = evaluate_model(model, transfer_tasks, args.senteval_path)

    with open(model_name.replace(".model", "") + '_evaluation_results.json', 'w') as outfile:
        json_object = json.dumps(results, indent=4)
        outfile.write(json_object)
