from tools.helper import *
import pandas as pd


def compute_surprisal(corpus, ngrams, stop_word):
    """

    :param corpus:
    :param ngrams:
    :param stop_word:
    :return:
    """
    df = pd.read_csv("../data/all_forms.csv", encoding="utf-8")  # load csv with long and short form words
    short_forms = set(list(df.short.values))
    long_forms = set(list(df.long.values))
    all_forms = short_forms | long_forms

    context_count = get_gram_count(gram_conv[ngrams], corpus, stop_word)
    probs = get_ngram_probs(ngrams, context_count, all_forms, corpus, stop_word)
    del context_count
    return probs
