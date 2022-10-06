import argparse
import json
import os


def get_sts_from_json(json) -> list:
    result = []

    for test in sts:
        result.append(str(round(json[test]["all"]["pearson"]["mean"], 2)))

    return result


def get_senteval_from_json(json) -> list:
    result = []

    for sent in senteval_tasks:
        result.append(str(json[sent]["acc"]))

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='the folder, where the json are located')
    args = parser.parse_args()

    for filename in os.listdir(os.getcwd() + "/" + args.folder):
        with open(os.path.join(os.getcwd() + "/" + args.folder, filename), 'r') as f:
            print("Used file: " + filename)
            eval_json = json.loads(f.read())

            results = get_sts_from_json(eval_json)
            print(results)

            results = get_senteval_from_json(eval_json)
            print(results)
            print("----------------------------------------------")
