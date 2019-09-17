import csv
import sys
from ast import literal_eval
from collections import defaultdict
from typing import Set

import numpy as np
import pandas as pd
from ngram_entropy import compute_entropy
from ngram_cosine import compute_cosine
from ngram_surprisal import compute_surprisal
from helper import get_gram_count
import gensim

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

from gensim.models import KeyedVectors



if __name__ == "__main__":
    corpus = sys.argv[1]  # native or nonnative or google
    stop_word = sys.argv[2]
    ngrams =  sys.argv[3]
    m = sys.argv[4]
    if m == "glove":
        glove_file = '/hal9000/masih/models/glove.6B.300d.txt'
        tmp_file = get_tmpfile("test_word2vec.txt")
        _ = glove2word2vec(glove_file, tmp_file)
        glove2word2vec(glove_file, tmp_file)
        model = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)
    else:
        model = gensim.models.Word2Vec.load_word2vec_format('/hal9000/masih/models/GoogleNews-vectors-negative300.bin',
                                                            binary=True)

    if corpus == 'google':
        filein = f"/w/nobackup/131/web1t-5gram-v1"
        stop_word = None
    else:
        filein = f"/ais/hal9000/masih/surprisal/{corpus}/"

    entropy_dict = compute_entropy(filein, corpus,ngrams,stop_word)
    cosine_dict = compute_cosine(filein, corpus, ngrams, model,stop_word)
    surprisal_list = compute_surpisal(filein, corpus, ngrams,stop_word)
    unigram_count = get_gram_count("unigram",filein,corpus,stop_word)
    sense_dict = {'porn': 1, 'photo': 2, 'phone': 3, 'bike': 4, 'tv': 5, 'carb': 6, 'math': 7, 'limo': 8,
                 'ref': 15, 'roach': 16, 'fridge': 17, 'exam': 18, 'chemo': 19, 'sax': 20, 'frat': 21, 'memo': 22,
                 'dorm': 9, 'kilo': 10, 'rhino': 11, 'undergrad': 12, 'hippo': 13, 'chimp': 14,
                 'telephone': 3, 'refrigerator': 17, 'undergraduate': 12, 'mathematics': 7,
                 'examination': 18, 'television': 5, 'photograph': 2, 'memorandum': 22, 'bicycle': 4,
                 'pornography': 1, 'fraternity': 21, 'limousine': 8, 'referee': 15, 'saxophone': 20,
                 'carbohydrate': 6, 'chemotherapy': 19, 'hippopotamus': 13, 'cockroach': 16,
                 'kilogram': 10, 'rhinoceros': 11, 'dormitory': 9, 'chimpanzee': 14}

    if corpus == "google":
        out_file = f"/hal9000/masih/surprisal/all_data/{corpus}_{ngrams}.csv"
    else:
        out_file = f"/hal9000/masih/surprisal/all_data/{corpus}_{ngrams}_{str(stop_words)}.csv"

    with open(out_file, 'w') as out_put:
        csvout = csv.writer(out_put)
        header =["context","word","form", "sense","count", "surprisal","entropy","cosine"]
        csvout.writerow(header)
        for row in surprisal_list:
            word = row[1]
            context = row[0]
            final_row = row[:-1] + [sense_dict[word]] +[unigram_count[word]] +[row[-1]] +[entropy_dict[context]] + [cosine_dict[context][word]]
            csvout.writerow(final_row)
    out_put.close()
