import argparse

import senteval
from transformers import BertModel, BertTokenizer

from models.pytorch.ClustBert import ClustBERT


# SentEval prepare and batcher
def prepare(params, samples):
    return


def batcher(params, batch):
    sentences = [" ".join(s).lower() for s in batch]

    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    tokens = tokens.to(device=device)
    result = transformer.model(tokens)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="define the path to model to load")
    parser.add_argument('--device', type=str, help='the device used by the program. Default is cuda:0')
    args = parser.parse_args()

    device = "cuda:0" if args.device is None else str(args.device)
    print("Using device: " + str(device))

    if args.model is not None:
        transformer = ClustBERT.load(args.model)
        transformer.to(device=device)
    else:
        transformer = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    params = {
        'task_path': '../SentEval/data',
        'usepytorch': True,
        "batch_size": 1
    }
    se = senteval.engine.SE(params, batcher, prepare)

    transfer_tasks = ['MR']
    results = se.eval(transfer_tasks)
