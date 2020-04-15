"""
This script reads SQUAD 1.1 and adds <ref> and </ref> tags to interrogation words
in questions
"""

import json

qwords = {"how", "what", "when", "where", "which", "who", "whom", "whose", "why"}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": "1.1", "data": data}, f)


def get_coverage(data):
    count = total = 0
    for dp in data:
        for para in dp["paragraphs"]:
            for qa in para["qas"]:
                question = set(qa["question"].lower().split())
                if question.intersection(qwords) != None:
                    count += 1
                total += 1
    return count / total


def annotate_qwords(data):
    for dp_idx, dp in enumerate(data):
        for para_idx, para in enumerate(dp["paragraphs"]):
            for qa_idx, qa in enumerate(para["qas"]):
                words = qa["question"].split()
                for word_idx, word in enumerate(words):
                    if word.lower() in qwords:
                        words[word_idx] = "<ref> " + word + " </ref>"
                        data[dp_idx]["paragraphs"][para_idx]["qas"][qa_idx][
                            "question"
                        ] = " ".join(words)
                        break
    return data


def main():
    train = read_json("./data/sluice/train.json")["data"]
    dev = read_json("./data/sluice/dev.json")["data"]
    test = read_json("./data/sluice/test.json")["data"]

    print("Coverage of training set: {}".format(get_coverage(train)))
    print("Coverage of development set: {}".format(get_coverage(dev)))
    print("Coverage of test set: {}".format(get_coverage(test)))

    write_json(annotate_qwords(train), "./data/sluice/train.json")
    write_json(annotate_qwords(dev), "./data/sluice/dev.json")
    write_json(annotate_qwords(test), "./data/sluice/test.json")


if __name__ == "__main__":
    main()
