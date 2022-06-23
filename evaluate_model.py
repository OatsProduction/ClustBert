import argparse

import senteval

from training.ClustBert import ClustBERT

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="define the path to model to load")
parser.add_argument('--device', type=str, help='the device used by the program. Default is cuda:0')
args = parser.parse_args()

device = "cuda:0" if args.device is None else str(args.device)
print("Using device: " + str(device))

clustBert = ClustBERT.load(args.model)
clustBert.to(device=device)


# SentEval prepare and batcher
def prepare(params, samples):
    return


def batcher(params, batch):
    sentences = [" ".join(s).lower() for s in batch]

    tokens = clustBert.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    tokens = tokens.to(device=device)
    result = clustBert.model(tokens)
    return result


params = {'task_path': '/home/willem/Documents/Uni/SentEval/data', 'usepytorch': True, "batch_size": 1}
se = senteval.engine.SE(params, batcher, prepare)

transfer_tasks = ['MR']
results = se.eval(transfer_tasks)
