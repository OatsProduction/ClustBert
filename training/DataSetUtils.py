import datasets


def get_snli_dataset():
    print("Getting the SNLI datasets")
    data = datasets.load_dataset('snli', split='train').select(range(1, 1000))
    data = data.map(lambda example: {'new_sentence': example['premise'] + example['hypothesis']})
    data = data.remove_columns(["premise", "hypothesis"])
    print("Finished getting the  SNLI datasets")

    return data
