from typing import Union

from datasets import load_dataset


def get_snli_dataset() -> Union:
    print("Getting the SNLI datasets")
    train = load_dataset('snli', split='train')
    train = train.map(lambda example: {'new_sentence': example['premise'] + example['hypothesis']})
    train = train.remove_columns(["premise", "hypothesis"])
    train = train.rename_column("label", "labels")

    valid_data = load_dataset('snli', split='validation')
    valid_data = valid_data.map(lambda example: {'new_sentence': example['premise'] + example['hypothesis']})
    valid_data = valid_data.map(lambda example: {'label': max(0, int(example['label']))})
    valid_data = valid_data.remove_columns(["premise", "hypothesis"])
    valid_data = valid_data.rename_column("label", "labels")
    print("Finished getting the  SNLI datasets")

    return train, valid_data
