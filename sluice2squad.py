import sys
import json

from collections import Counter


def to_squad_format(esc_data_path, is_train):
    data = []
    key_errors = 0
    value_errors = 0
    with open(esc_data_path, "r", encoding="utf-8") as f:
        esc_data = f.readlines()

    for example in esc_data:
        try:
            example = json.loads(example)
            before_sentences = " ".join(
                before_sent["string"] for before_sent in example["before"]
            )
            match_sentence = example["match"]["sentence"]["string"]
            after_sentences = " ".join(
                before_sent["string"] for before_sent in example["after"]
            )
            context = " ".join((before_sentences, match_sentence, after_sentences))

            sluice_id = "{0[file]}_{0[line]}_{0[treeNode]}".format(example["metadata"])
            title = "irrelevant"

            answers = []
            for annotation in example["annotations"]:
                ans = annotation["tags"]["Antecedent"][0]["text"]
                start_index = (
                    context.index(
                        annotation["tags"]["Antecedent"][0]["offsets"][0]["lineText"]
                    )
                    + annotation["tags"]["Antecedent"][0]["offsets"][0]["start"]
                )
                end_index = (
                    context.index(
                        annotation["tags"]["Antecedent"][0]["offsets"][0]["lineText"]
                    )
                    + annotation["tags"]["Antecedent"][0]["offsets"][0]["end"]
                )
                answers.append(
                    {"text": ans, "answer_start": start_index, "answer_end": end_index}
                )
                """
                if ans != context[start_index:end_index]:
                    print(match_sentence)
                    print(ans)
                    print(context[start_index:end_index])
                    print()
                """

            if is_train:
                answers = [answers[0]]

            dp = {
                "title": title,
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "question": match_sentence,
                                "answers": answers,
                                "id": sluice_id,
                            }
                        ],
                    }
                ],
            }
            data.append(dp)
        except KeyError:
            key_errors += 1
            continue
        except ValueError:
            value_errors += 1
            continue

    print(
        "There were {} key errors and {} value errors while parsing {} datapoints".format(
            key_errors, value_errors, len(esc_data)
        )
    )
    return {"version": "v1.1", "data": data}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Exactly one argument is required: path of file to convert!")
        sys.exit(1)

    with open(sys.argv[1], "w") as outfile:
        json.dump(to_squad_format(sys.argv[1], "train" in sys.argv[1]), outfile)
