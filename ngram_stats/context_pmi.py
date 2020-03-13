import pandas as pd
from tools.helper import *
import sys

import csv
from ast import literal_eval
import numpy as np


def get_pmi_avg(unigrams, context_count, corpus, corpus_count, stop):
    """
    Get the average pmi for each context
    :param unigrams:
    :param context_count:
    :param context_set:
    :param corpus:
    :return:
    """
    pmi_avgs = defaultdict(float)
    next_word_count = defaultdict(int)
    file_list = file_locations("trigram", corpus, stop)
    for file in file_list:
        with open(file, 'r') as file_in:
            if corpus == 'google':
                reader = csv.reader(file_in, dialect='excel-tab', quoting=csv.QUOTE_NONE)
            else:
                reader = csv.reader(file_in)

            for row in reader:
                temp_gram = row[0].lower().split(' ')
                temp_context = ' '.join(word for word in temp_gram[:-1])
                if len(temp_gram) < 2:
                    print(temp_gram, 'skipped')
                elif temp_context in context_count:
                    try:
                        prob = (literal_eval(row[1]) * corpus_count) / (context_count[temp_context] * unigrams[temp_gram[-1]])
                        pmi_avgs[temp_context] += np.log2(prob)
                        next_word_count[temp_context] += 1

                    except ZeroDivisionError:
                        print(temp_context)
                        sys.stdout.flush()
    pmi_avgs = normalize_pmi(pmi_avgs, next_word_count)
    return pmi_avgs


def normalize_pmi(pmi_dict, next_word_dict):
    for context in pmi_dict:
        pmi_dict[context] = pmi_dict[context] / next_word_dict[context]
    return pmi_dict


def main(argv):
    corpus = argv[1]
    if corpus=="google":
        filein = f"/hal9000/masih/surprisal/all_data/{corpus}_trigram.csv"
        stop = None
    else:
        filein = f"/hal9000/masih/surprisal/all_data/{corpus}_trigram_False.csv"
        stop = False
    df = pd.read_csv(filein, encoding='utf-8')
    counts_df = {"native": 2238501491, "wiki": 1486997861, "google": 1024908267229}


    context_count = pd.Series(df.context_count.values,index=df.context).to_dict()
    unigrams = get_gram_count("unigram", corpus, stop)
    df["pmi"] = df.apply(lambda row: np.log2(counts_df[corpus]/row["count"])- row["surprisal"], axis=1)
    pmi_dict = get_pmi_avg(unigrams, context_count, corpus, counts_df[corpus],stop)
    print("got pmi averages")
    df["pmi_avg"] = df["context"].apply(lambda x: pmi_dict[x])
    df["deviated_pmi"] = df.apply(lambda row: row["pmi"] - row["pmi_avg"], axis=1)
    df.to_csv(filein, index=None, encoding="utf-8")
    print("done", corpus)

if __name__=="__main__":
    main(sys.argv)