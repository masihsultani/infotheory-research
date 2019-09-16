import csv
import sys
from ast import literal_eval
from collections import defaultdict
from typing import Set

import numpy as np
import pandas as pd

from helper import get_context, file_locations, get_gram_count


def compute_entropy(infile, corpus, gram, stop_words=None):
    """

    :param infile: str
        location of the ngrams
    :param corpus: str
        the corpus we are calculating entropy for
    :param stop_words: str
        corpus with stop words removed or not

    :return:
    """

    df = pd.read_csv("all_forms.csv", encoding="utf-8")  # load csv with long and short form words
    short_forms = set(list(df.short.values))
    long_forms = set(list(df.long.values))
    prime_prob = defaultdict(lambda: defaultdict(float))
    short_set = set()
    long_set =set()

    context_files = file_locations(gram, infile, corpus=corpus,stop_words=stop_words)
    for file in context_files:
        long_contexts = get_context(long_forms, file, corpus)
        short_contexts = get_context(short_forms, file, corpus)

        long_set.update(long_contexts)
        short_set.update(short_contexts)
        print(len(long_set),len(short_set))
        sys.stdout.flush()

    context_set = short_set | long_set
    context_count = get_gram_count(gram_conv[gram],infile,corpus,stop_words)

    for file in context_files:
        with open(file, 'r', encoding="utf-8") as file_2:
            if corpus=="google":
                reader = csv.reader(file_2, dialect="excel-tab", quoting=csv.QUOTE_NONE)
            else:
                reader = csv.reader(file_2)
            for row in reader:
                temp_gram = row[0].split()
                temp_context = ' '.join(word for word in temp_gram[:-1])
                if temp_context in context_set:
                    try:
                        prime_prob[temp_context][temp_gram[-1]] = literal_eval(row[1]) / context_count[temp_context]
                    except ZeroDivisionError:
                        print(temp_context, " ZeroDivision Error")
                    sys.stdout.flush()
        file_2.close()

    entropy_dict = entropy_set_calc(context_set, prime_prob)
    return entropy_dict


def entropy_set_calc(context_set, probs):
    entropy_dict = defaultdict(float)

    for context in context_set:
        entropy =0
        for word in probs[context]:
            entropy += probs[context][word] * (np.log2(1 / (probs[context][word])))

        entropy_dict[context] = entropy

    return entropy_dict


# if __name__ == "__main__":
#     corpus = sys.argv[1]  # native or nonnative or google
#     stopword = ['True', 'False'] #sys.argv[2]
#
#     if corpus == 'google':
#         filein = f"/w/nobackup/131/web1t-5gram-v1"
#         compute_entropy(filein, corpus)
#     else:
#         for x in stopword:
#             filein = f"/ais/hal9000/masih/surprisal/{corpus}/"
#             compute_entropy(filein, corpus,x)
