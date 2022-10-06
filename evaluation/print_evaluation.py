def get_sts_from_json(json, sts) -> list:
    result = []

    for test in sts:
        result.append(str(round(json[test]["all"]["pearson"]["mean"], 2)))

    return result


def get_senteval_from_json(json, senteval_tasks) -> list:
    result = []

    for sent in senteval_tasks:
        result.append(str(json[sent]["acc"]))

    return result
