import argparse

from training.ClustBert import ClustBERT

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, help='the index of the cuda GPU. Default is 0')
parser.add_argument("--model", type=str, help="define the path to model to load")
args = parser.parse_args()

clustBert = ClustBERT.load(args.model)
