from typing import Union

from datasets import Dataset
from datasets import load_dataset


def get_snli_dataset() -> Union:
    print("Getting the SNLI datasets")
    train = load_dataset('snli', split='train')
    train = train.map(lambda example: {'new_sentence': example['premise'] + example['hypothesis']})
    train = train.remove_columns(["premise", "hypothesis"])
    train = train.map(lambda example: {'label': max(0, int(example['label']))})
    train = train.rename_column("label", "labels")

    valid_data = load_dataset('snli', split='validation')
    valid_data = valid_data.map(lambda example: {'new_sentence': example['premise'] + example['hypothesis']})
    valid_data = valid_data.map(lambda example: {'label': max(0, int(example['label']))})
    valid_data = valid_data.remove_columns(["premise", "hypothesis"])
    valid_data = valid_data.rename_column("label", "labels")
    print("Finished getting the  SNLI datasets")

    return train, valid_data


def preprocess_datasets(tokenizer, data_set: Dataset) -> Dataset:
    print("Preprocess the data")
    data_set = data_set.map(
        lambda examples: tokenizer(examples['new_sentence'], padding=True, truncation=True),
        batched=True)

    for data in data_set:
        if data["labels"] is -1:
            print("UHU")

    data_set.set_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    print("Finsihed the Preprocess the data")
    return data_set
