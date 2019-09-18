from helper import *
import numpy as np
import pandas as pd


def compute_surprisal(infolder, corpus, ngrams, stop_word):
    """

    :param infolder:
    :param corpus:
    :param ngrams:
    :param stop_word:
    :return:
    """
    df = pd.read_csv("all_forms.csv", encoding="utf-8")  # load csv with long and short form words
    short_forms = set(list(df.short.values))
    long_forms = set(list(df.long.values))
    all_forms = short_forms | long_forms

    context_count = get_gram_count(gram_conv[ngrams], infolder, corpus, stop_word)
    probs = get_ngram_probs(ngrams, context_count, all_forms, infolder, corpus, stop_word)
    del context_count
    for word in probs:
        for context in probs[word]:
            p = probs[word][context]
            probs[word][context] = np.log2(1 / p)

    return probs
