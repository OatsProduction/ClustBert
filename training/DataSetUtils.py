from datasets import load_dataset


def get_snli_dataset():
    print("Getting the SNLI datasets")
    data = load_dataset('snli', split='train')
    data = data.map(lambda example: {'new_sentence': example['premise'] + example['hypothesis']})
    data = data.remove_columns(["premise", "hypothesis"])
    print("Finished getting the  SNLI datasets")

    return data
