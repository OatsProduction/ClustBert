# ClustBert

This repository contains code to train a BERT model with the ClustBERT method.
The main script is the following.
Other scripts are helpful for other functions.

Use --help to get more information about the script and the parameters.

    train_hyper_parameter_clustbert.py

How to download the SentEval datasets.
Needs to be in the same folder as ClustBERT.

    git clone https://github.com/facebookresearch/SentEval.git
    cd $PWD/SentEval/data/downstream && bash get_transfer_data.bash > /dev/null 2>&1
    python setup.py install

-------

Useful commands:

- nvidia-smi
- source .env/bin/activate
- CUDA_VISIBLE_DEVICES=0 (When runnining multiple sweeps on a single machine, we can
  use the following command.)
