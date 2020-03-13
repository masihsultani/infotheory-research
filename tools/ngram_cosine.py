from collections import defaultdict

import pandas as pd

from tools.helper import get_context


def compute_cosine(corpus, gram, model, stop_words=None):
    """

    :param corpus:
    :param gram:
    :param model:
    :param stop_words:
    :return:
    """

    df = pd.read_csv("all_forms.csv", encoding="utf-8")  # load csv with long and short form words
    short_forms = set(list(df.short.values))
    long_forms = set(list(df.long.values))

    long_set = get_context(long_forms, gram, corpus, stop_words, True)
    short_set = get_context(short_forms, gram, corpus, stop_words, True)
    all_set = long_set | short_set
    all_cosine_dict = get_cosines(all_set, model)

    return all_cosine_dict


def get_cosines(phrase_set, model):
    """

    :param phrase_set:
    :param model:
    :return:
    """
    cosine_dict = defaultdict(lambda: defaultdict(float))
    for phrase in phrase_set:
        phrase_list = phrase.split()
        word = phrase_list[-1]
        context = " ".join(x for x in phrase_list[:-1])
        dist = cosine_sim(phrase, model)
        cosine_dict[context][word] = dist
    return cosine_dict


def cosine_sim(x, model):
    """

    :param x:
    :param model:
    :return:
    """
    words = x.split()
    mean_dist = 0
    if words[-1] in model.wv.vocab:
        for word in words[:-1]:
            if word in model.wv.vocab:
                mean_dist += model.similarity(word, words[-1])
    try:
        mean_dist = mean_dist / (len(words) - 1)

    except ZeroDivisionError:
        mean_dist = 0
    return mean_dist

# if __name__ == "__main__":
#     data = sys.argv[1]  # native or nonnative or google
#     stopword = ['True', 'False'] #sys.argv[2]
#
#     if data == 'google':
#         filein = f"/w/nobackup/131/web1t-5gram-v1"
#         main_prog(filein, data)
#     else:
#         for x in stopword:
#             filein = f"/ais/hal9000/masih/surprisal/{data}/"
#             main_prog(filein, data,x)
