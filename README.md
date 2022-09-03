# ClustBert

Zum Wechsel zu Python Environment

    source .env/bin/activate

SentEval download

    git clone https://github.com/facebookresearch/SentEval.git
    cd $PWD/SentEval/data/downstream && bash get_transfer_data.bash > /dev/null 2>&1
    python setup.py install

Nützliche Befehle:

- nvidia-smi

Evaluation

    Xnohup python evaluate_model.py --model $model_name$ --device cpu --senteval_path ../../SentEval/data --all &

