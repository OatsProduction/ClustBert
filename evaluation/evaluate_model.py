import argparse
import json
import logging
import pickle

import senteval
import torch
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.modeling_utils import no_init_weights


def prepare(params, samples):
    return


def batcher(params, batch):
    sentences = [" ".join(s).lower() for s in batch]

    with torch.no_grad():
        y = params['tokenizer'](sentences, padding=True, truncation=True, return_tensors="pt")
        y = params['model'].get_sentence_vectors_with_cls_token(params["device"], [y.data])
    return y[0]


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
sts = ["STS12", "STS13", "STS14", "STS15", "STS16"]
senteval_tasks = ["MR", "CR", "SUBJ", "MPQA", "SST2", "SST5", "TREC", "MRPC"]


def evaluate_model(transformer, tasks, senteval_path):
    transformer.eval()
    for parameter in transformer.parameters():
        parameter.requires_grad = False
    transformer.to(torch.device("cpu"))

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
    no_init_weights(True)

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
        config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True, gradient_checkpointing=False,
                                            pruned_heads=
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
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    elif args.model is not None:
        model_name = args.model
        model = pickle.load(open("../output/" + args.model, 'rb'))
    else:
        model_name = "plain_bert"
        model = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)

    if args.sts:
        transfer_tasks = sts
    if args.senteval:
        transfer_tasks = senteval_tasks
    if args.all:
        transfer_tasks = sts + senteval_tasks
    else:
        transfer_tasks = ["STS13"]

    results = evaluate_model(model, transfer_tasks, args.senteval_path)

    with open(model_name.replace(".model", "") + '_evaluation_results.json', 'w') as outfile:
        json_object = json.dumps(results, indent=4)
        outfile.write(json_object)
