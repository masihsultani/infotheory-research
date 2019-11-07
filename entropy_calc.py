from corpus_entropy import compute_entropy
from helper import*


if __name__ == '__main__':
    corpora =["google", "US","wiki","europe", "native","nonnative"]
    import pandas as pd

    counts = pd.read_csv("counts.csv", encoding="utf-8", index_col=0)
    counts.loc["google"] = [1024908267229]

    for corpus in corpora:
        unigrams = get_gram_count("unigram",corpus, None)
        tokens = counts.loc[corpus][f"unigram"]
        entropy = compute_entropy(unigrams,tokens)
        print(f"{corpus} {entropy}")