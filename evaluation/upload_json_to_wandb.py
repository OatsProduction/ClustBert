import json

import wandb

from evaluation.evaluate_model import sts, senteval_tasks
from evaluation.print_evaluation import get_sts_from_json, get_senteval_from_json

if __name__ == '__main__':
    wandb.init(project="ClustBert")

    f = open("bert/random_evaluation_results.json")
    result = json.load(f)

    sts_result = get_sts_from_json(result)
    my_table = wandb.Table(columns=sts, data=[sts_result])
    wandb.log({"STS": my_table})

    senteval_result = get_senteval_from_json(result)
    my_table = wandb.Table(columns=senteval_tasks, data=[senteval_result])
    wandb.log({"SentEval": my_table})
