import json
import glob
import spacy

from tqdm import tqdm
from random import Random
from uuid import UUID
from collections import defaultdict

raw_path = "./data/vpe/docs/wsj/"
ann_path = "./data/vpe/ann/wsj/"

nlp = spacy.load("en")
start_str = ".START "

# for reproducibility
rd = Random()
rd.seed(42)


def read_context_files():
    context = {}
    for filepath in glob.glob(raw_path + "**/wsj*", recursive=True):
        with open(filepath, "r", encoding="utf-8") as f:
            context[filepath.split("/")[-1]] = f.read()
    return context


def read_ann_files():
    ann = defaultdict(list)
    for filepath in glob.glob(ann_path + "**/*.ann", recursive=True):
        with open(filepath, "r", encoding="utf-8") as f:
            for row in f.readlines():
                row = row.split()
                ann[row[0]].append(
                    {
                        "vpe_start": int(row[1]),
                        "vpe_end": int(row[2]),
                        "ant_start": int(row[3]),
                        "ant_end": int(row[4]),
                    }
                )
    return ann


def index2phrase(context, ann):
    vpe_start = ann["vpe_start"]
    vpe_end = ann["vpe_end"]
    ant_start = ann["ant_start"]
    ant_end = ann["ant_end"]
    return context[vpe_start:vpe_end], context[ant_start:ant_end]


def get_sent_from_idx(context, start_idx, end_idx):
    # we need the entire sentence in which the ellipsis is present to form the question
    doc = nlp(context)
    for sent in doc.sents:
        if start_idx >= sent.start_char and end_idx <= sent.end_char:
            return sent.text
    # print('Could not find a sentence which encloses the ellipsis. Weird!')
    return None


def write_squad(filename, data):
    with open("./data/vpe/{}.json".format(filename), "w", encoding="utf-8") as f:
        json.dump({"version": "1.1", "data": data}, f)


print("Processing context files...")
context = read_context_files()

print("Processing annotation files...")
ann = read_ann_files()

squad_train = []
squad_val = []
squad_test = []
train_qa = 0
val_qa = 0
test_qa = 0
skip_num = 0

for idx in tqdm(ann.keys()):
    squad_obj = {"title": "irrelevant", "paragraphs": []}
    cut_start = False
    c = context[idx]

    if c.startswith(start_str):
        c = c[len(start_str) :]
        cut_start = True
    squad_obj["paragraphs"].append({"context": c})

    for a in ann[idx]:
        qas = []
        if cut_start:
            ant_start, ant_end = (
                a["ant_start"] - len(start_str),
                a["ant_end"] - len(start_str),
            )
            vpe_start, vpe_end = (
                a["vpe_start"] - len(start_str),
                a["vpe_end"] - len(start_str),
            )
        answer = c[ant_start:ant_end]
        question = get_sent_from_idx(c, vpe_start, vpe_end)
        question = question.replace(
            c[vpe_start:vpe_end], "<ref> " + c[vpe_start:vpe_end] + " </ref>"
        )
        if not question:
            skip_num += 1
            continue

        qas.append(
            {
                "question": question,
                "answers": [
                    {"text": answer, "answer_start": ant_start, "answer_end": ant_end}
                ],
                "id": UUID(int=rd.getrandbits(128)).hex,
            }
        )
    squad_obj["paragraphs"][0]["qas"] = qas

    # Sections 00-19 are used as train/dev data.
    # Sections 20-24 were used as test data
    # according to Robust VP Ellipsis Resolution in DR Theory
    section = int(idx.split("_")[1][:2])
    if section in range(18):
        squad_train.append(squad_obj)
        train_qa += 1
    elif section in [18, 19]:
        squad_val.append(squad_obj)
        val_qa += 1
    else:
        squad_test.append(squad_obj)
        test_qa += 1

print(
    "Writing {} train data points with {} QA pairs...".format(
        len(squad_train), train_qa
    )
)
print(
    "Writing {} validation data points with {} QA pairs...".format(
        len(squad_val), val_qa
    )
)
print(
    "Writing {} test data points with {} QA pairs...".format(len(squad_test), test_qa)
)
print(
    "Skipped {} QA pairs because ellipsis could not be identified...".format(skip_num)
)
write_squad("train", squad_train)
write_squad("val", squad_val)
write_squad("test", squad_test)
