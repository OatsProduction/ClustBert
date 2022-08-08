from typing import Union

from datasets import Dataset
from datasets import load_dataset


def get_snli_dataset() -> Union:
    print("Getting the SNLI datasets")
    train = load_dataset('snli', split='train')
    train = train.map(lambda example: {'text': example['premise'] + example['hypothesis']})
    train = train.remove_columns(["premise", "hypothesis"])
    train = train.map(lambda example: {'label': max(0, int(example['label']))})
    train = train.rename_column("label", "labels")

    valid_data = load_dataset('snli', split='validation')
    valid_data = valid_data.map(lambda example: {'text': example['premise'] + example['hypothesis']})
    valid_data = valid_data.map(lambda example: {'label': max(0, int(example['label']))})
    valid_data = valid_data.remove_columns(["premise", "hypothesis"])
    valid_data = valid_data.rename_column("label", "labels")
    print("Finished getting the  SNLI datasets")

    return train, valid_data


def get_million_headlines() -> Dataset:
    dataset = load_dataset("DeveloperOats/Million_News_Headlines")
    dataset = dataset.rename_column("headline_text", "text")
    dataset = dataset.remove_columns("publish_date")
    dataset = dataset["train"]

    return dataset


def get_pedia_classes() -> Dataset:
    dataset = load_dataset("DeveloperOats/DBPedia_Classes")
    dataset = dataset.remove_columns(["l1", "l2"])
    dataset = dataset["train"]

    return dataset


def preprocess_datasets(tokenizer, data_set: Dataset) -> Dataset:
    print("Preprocess the data")
    data_set = data_set.map(
        lambda data_point: tokenizer(data_point['text'], padding=True, truncation=True),
        batched=True)

    # for data in data_set:
    #     if data["labels"] == -1:
    #         print("UHU")

    if 'labels' in data_set:
        data_set.set_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    else:
        data_set.set_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask'])

    print("Finsihed the Preprocess the data")
    return data_set
