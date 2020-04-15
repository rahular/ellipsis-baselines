import re
import os
import sys
import json
import tempfile
import subprocess
import operator
import collections

from tqdm import tqdm
from collections import Counter

squad_pred_file = "./DrQA/models/ontonotes.preds"
squad_test_file = "./data/ontonotes/test.json"
conll_test_file = "./data/ontonotes/test.conll"
conll_output_file = "./DrQA/models/ontonotes.conll"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_bound(mention, ant):
    return mention.find(" " + ant + " ") > -1 or ant.find(" " + mention + " ") > -1


BEGIN_DOCUMENT_REGEX_ONTONOTES = re.compile(r"#begin document \((.*)\); part (\d+)")
BEGIN_DOCUMENT_REGEX_WIKICOREF = re.compile(r"#begin document (.+)")
COREF_RESULTS_REGEX = re.compile(
    r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*",
    re.DOTALL,
)


def get_doc_key(doc_id, part):
    return "{}_{}".format(doc_id, int(part))


def output_conll(input_file, output_file, predictions):
    prediction_map = {}
    for doc_key, clusters in tqdm(predictions.items()):
        start_map = collections.defaultdict(list)
        end_map = collections.defaultdict(list)
        word_map = collections.defaultdict(list)
        for cluster_id, mentions in enumerate(clusters):
            for start, end in mentions:
                if start == end:
                    word_map[start].append(cluster_id)
                else:
                    start_map[start].append((cluster_id, end))
                    end_map[end].append((cluster_id, start))
        for k, v in start_map.items():
            start_map[k] = [
                cluster_id
                for cluster_id, end in sorted(
                    v, key=operator.itemgetter(1), reverse=True
                )
            ]
        for k, v in end_map.items():
            end_map[k] = [
                cluster_id
                for cluster_id, start in sorted(
                    v, key=operator.itemgetter(1), reverse=True
                )
            ]
        prediction_map[doc_key] = (start_map, end_map, word_map)

    word_index = 0
    for line in tqdm(input_file.readlines()):
        row = line.split()
        if len(row) == 0:
            output_file.write("\n")
        elif row[0].startswith("#"):
            begin_match_o = re.match(BEGIN_DOCUMENT_REGEX_ONTONOTES, line)
            begin_match_w = re.match(BEGIN_DOCUMENT_REGEX_WIKICOREF, line)
            if begin_match_o:
                doc_key = get_doc_key(begin_match_o.group(1), begin_match_o.group(2))
                start_map, end_map, word_map = prediction_map[doc_key]
                word_index = 0
            elif begin_match_w:
                doc_key = begin_match_w.group(1)
                start_map, end_map, word_map = prediction_map[doc_key]
                word_index = 0
            output_file.write(line)
            output_file.write("\n")
        else:
            assert row[0] == doc_key or get_doc_key(row[0], row[1]) == doc_key
            coref_list = []
            if word_index in end_map:
                for cluster_id in end_map[word_index]:
                    coref_list.append("{})".format(cluster_id))
            if word_index in word_map:
                for cluster_id in word_map[word_index]:
                    coref_list.append("({})".format(cluster_id))
            if word_index in start_map:
                for cluster_id in start_map[word_index]:
                    coref_list.append("({}".format(cluster_id))

            if len(coref_list) == 0:
                row[-1] = "-"
            else:
                row[-1] = "|".join(coref_list)

            output_file.write("   ".join(row))
            output_file.write("\n")
            word_index += 1

    input_file.close()
    output_file.close()


def official_conll_eval(gold_path, predicted_path, metric):
    cmd = [
        "/home/wjv316/e2e-coref/conll-2012/scorer/v8.01/scorer.pl",
        metric,
        gold_path,
        predicted_path,
        "none",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        print(stderr)

    coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    print("METRIC: {}\tp: {}\tr: {}\t f1: {}".format(metric, precision, recall, f1))


global_clusters = collections.defaultdict(list)


def append2global(doc_key):
    if not clusters:
        global_clusters[doc_key].append([])
        return

    # don't add a mention if it is already added in any cluster in the doc!
    # this may result in the loss of some mentions because we did a string match to get
    # mentions rather than use their gold indices (TODO' can we improve this?)
    flat_mentions = [mention for cluster in clusters.values() for mention in cluster]
    for key, value in clusters.items():
        if key != (-1, -1) and key not in flat_mentions:
            value.append(key)
        global_clusters[doc_key].append(value)


clusters = {}


def add2cluster(mention, ant):
    keys = clusters.keys()
    if len(keys) == 0:
        # no clusters formed. create one
        clusters[ant] = [mention]
    else:
        added = False
        for key in keys:
            if key == ant:
                # found a cluster to which the mention belongs
                clusters[key].append(mention)
                added = True
        if not added:
            # nothing fits, add a new cluster
            clusters[ant] = [mention]


def get_ant_idx(context, ant):
    ant = ant.split()
    for idx, word in enumerate(context):
        if word == ant[0] and ant == context[idx : idx + len(ant)]:
            return (idx, idx + len(ant) - 1)
    # could not find the antecedent!
    return None


test_data = read_json(squad_test_file)["data"]
pred_data = read_json(squad_pred_file)
# nbest = read_json(nbest_pred_file)
missed_counter, total_counter = 0, 0
for dp in test_data:
    doc_key = dp["title"]
    clusters = {}
    for para in dp["paragraphs"]:
        context = para["context"].split()
        for qa in para["qas"]:
            mention = qa["question"].split("<ref>")[1].split("</ref>")[0]
            pred_ant = pred_data[qa["id"]]

            mention_idx = tuple(qa["mention_span"])
            pred_idx = get_ant_idx(context, pred_ant)
            # pred_idx = (nbest[qa['id']][0]['start_index'],
            # 			nbest[qa['id']][0]['end_index'])
            if pred_idx:
                add2cluster(mention_idx, pred_idx)
            else:
                missed_counter += 1
            total_counter += 1
    # break
    append2global(doc_key)

print("Number of misses: {}/{}".format(missed_counter, total_counter))
ctest = open(conll_test_file, "r", encoding="utf-8")
cout = open(conll_output_file, "w", encoding="utf-8")
output_conll(ctest, cout, global_clusters)

[
    official_conll_eval(conll_test_file, conll_output_file, m)
    for m in ("muc", "bcub", "ceafe")
]
