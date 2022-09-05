import argparse
import json
import os

from evaluation.evaluate_model import sts, seneval_tasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='the folder, where the json are located')
    args = parser.parse_args()

    for filename in os.listdir(os.getcwd() + "/" + args.folder):
        with open(os.path.join(os.getcwd() + "/" + args.folder, filename), 'r') as f:
            print("Used file: " + filename)
            stsSplit = sts.split(",")
            eval_json = json.loads(f.read())
            bigstring = ""

            for test in stsSplit:
                bigstring += test + ": " + str(eval_json[test]["all"]["pearson"]["mean"]) + " | "

            print(bigstring)

            bigstring = ""

            for sent in seneval_tasks.split(","):
                bigstring += sent + ": " + str(eval_json[sent]["acc"]) + " | "

            print(bigstring)
            print("----------------------------------------------")
