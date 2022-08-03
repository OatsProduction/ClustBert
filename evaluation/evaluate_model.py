import argparse
import logging

import senteval
from transformers import BertModel, BertTokenizer


# SentEval prepare and batcher
# from models.pytorch.ClustBert import ClustBERT


def prepare(params, samples):
    return


def batcher(params, batch):
    sentences = [" ".join(s).lower() for s in batch]

    y = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    y = y.to(device=device)
    y = transformer(y)
    y = y[:, 0, :].view(-1, 768)

    return y


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="define the path to model to load")
    parser.add_argument('--device', type=str, help='the device used by the program. Default is cuda:0')
    args = parser.parse_args()

    device = "cuda:0" if args.device is None else str(args.device)
    print("Started the evaluation script with the device: " + str(device))

    if args.model is not None:
        transformer = ClustBERT.load(args.model)
        transformer.to(device=device)
    else:
        transformer = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        transformer.to(device=device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    params = {
        'task_path': '../../SentEval/data',
        'usepytorch': True,
        'kfold': 5
    }
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                            'tenacity': 3, 'epoch_size': 2}

    batch = "I love it"
    result = batcher(None, batch)
    se = senteval.engine.SE(params, batcher, prepare)

    transfer_tasks = ['MR']
    print("Started the tasks")

    results = se.eval(transfer_tasks)
    print(results)
