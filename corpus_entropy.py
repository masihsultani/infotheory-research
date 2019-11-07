import numpy as np
from helper import *


def compute_entropy(unigrams, token_count):
    entropy = 0
    for word in unigrams:
        prob = unigrams[word] / token_count
        entropy += prob * (np.log2(1 / prob))
    return entropy


def compute_cond_entropy(corpus, gram, stop_word, context_count, token_count):
    all_files = file_locations(gram, corpus, stop_word)
    entropy = 0
    for file in all_files:
        with open(file, 'r', encoding="utf-8") as file_2:
            if corpus == "google":
                reader = csv.reader(file_2, dialect="excel-tab", quoting=csv.QUOTE_NONE)
            else:
                reader = csv.reader(file_2)
            for row in reader:
                temp_gram = row[0].lower().split()
                temp_context = ' '.join(word for word in temp_gram[:-1])
                try:
                    cond_prop = int(row[1]) / context_count[temp_context]
                except ZeroDivisionError:
                    continue

                joint_prop = int(row[1]) / token_count
                entropy += joint_prop * np.log2(1 / cond_prop)
    return entropy


def main_func(argv):
    import pandas as pd
    counts = pd.read_csv("counts.csv", encoding="utf-8", index_col=0)
    counts.loc["google"] = [1024908267229]
    corpus = argv[0]
    gram = argv[1]
    tokens = counts.loc[corpus][f"unigram"]
    if corpus != "google":
        stop_word = argv[2]
    else:
        stop_word = None
    first_gram_count = get_gram_count(gram_conv[gram], corpus, stop_word)

    if gram == "bigram":
        unigrams = first_gram_count
    else:
        unigrams = get_gram_count("unigram", corpus, None)
    entropy = compute_entropy(unigrams, tokens)
    print("done entropy")
    sys.stdout.flush()
    conditional_entropy = compute_cond_entropy(corpus, gram, stop_word, first_gram_count, tokens)

    mutualInformation = entropy - conditional_entropy
    print(corpus, gram, stop_word, f"MI= {mutualInformation}")
    sys.stdout.flush()

if __name__=="__main__":
    corpus = sys.argv[1]
    grams = ["bigram", "trigram"]
    stop_words = ["True", "False"]

    for gram in grams:
        for t in stop_words:
            if (corpus == "google") and (t == "True"):
                continue
            args =[corpus,gram,t]
            main_func(args)



