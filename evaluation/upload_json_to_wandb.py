import json

import wandb

from evaluate_model import sts, senteval_tasks
from evaluation.print_evaluation import get_sts_from_json, get_senteval_from_json

if __name__ == '__main__':
    wandb.init(project="ClustBert")

    f = open("bert/random_evaluation_results.json")
    result = json.load(f)
    run_name = "l_4_test_me"

    sts_result = [run_name] + get_sts_from_json(result, sts)
    my_table = wandb.Table(columns=["Id"] + sts, data=[sts_result])
    wandb.log({"STS": my_table})

    senteval_result = [run_name] + get_senteval_from_json(result, senteval_tasks)
    my_table = wandb.Table(columns=["Id"] + senteval_tasks, data=[senteval_result])
    wandb.log({"SentEval": my_table})
