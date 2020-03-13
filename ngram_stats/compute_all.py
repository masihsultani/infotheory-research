import csv
import sys
from tools.ngram_entropy import compute_entropy
from tools.ngram_surprisal import compute_surprisal
from tools.helper import get_gram_count, WORD_SENSE_DICT
import pandas as pd
import numpy as np


def main(argv):
    corpus = argv[1]  # native or nonnative or google or wiki
    stop_word = argv[3]
    ngrams = argv[2]
    #m = argv[4]
    # if m == "glove":
    #     glove_file = '/hal9000/masih/models/glove.6B.300d.txt'
    #     tmp_file = get_tmpfile("test_word2vec.txt")
    #     _ = glove2word2vec(glove_file, tmp_file)
    #     glove2word2vec(glove_file, tmp_file)
    #     model = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)
    # else:
    #     model = gensim.models.KeyedVectors.load_word2vec_format(
    #         '/hal9000/masih/models/GoogleNews-vectors-negative300.bin',
    #         binary=True)

    df = pd.read_csv("all_forms2.csv", encoding="utf-8")  # load csv with long and short form words
    short_forms = set(list(df.short.values))

    entropy_dict = compute_entropy(corpus, ngrams, stop_words=stop_word)
    # cosine_dict = compute_cosine(filein, corpus, ngrams, model, stop_word)
    surprisal_dict = compute_surprisal(corpus, ngrams, stop_word)
    unigram_count = get_gram_count("unigram", corpus, stop_word)


    if corpus == "google":
        out_file = f"/hal9000/masih/surprisal/all_data/{corpus}_{ngrams}.csv"
    else:
        out_file = f"/hal9000/masih/surprisal/all_data/{corpus}_{ngrams}_{str(stop_word)}.csv"

    with open(out_file, 'a') as out_put:
        csvout = csv.writer(out_put)
        # header = ["context", "word", "form", "sense", "count", "surprisal", "entropy"]
        # csvout.writerow(header)

        for word in surprisal_dict:
            if word in short_forms:
                i = 0
            else:
                i = 1
            for context in surprisal_dict[word]:
                surprisal = np.log2(1 / surprisal_dict[word][context])

                final_row = [context, word, i, WORD_SENSE_DICT[word], unigram_count[word], surprisal, entropy_dict[context]]
                csvout.writerow(final_row)
    out_put.close()


if __name__ == "__main__":
    main(sys.argv)
