import datasets


def get_snli_dataset():
    data = datasets.load_dataset('snli', split='train').select([0, 1, 2, 3, 4, 5, 6, 7, 10, 20, 30, 40, 50])
    data = data.map(lambda example: {'new_sentence': example['premise'] + example['hypothesis']})
    data = data.remove_columns(["premise", "hypothesis"])

    return data
