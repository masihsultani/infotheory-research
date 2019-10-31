import pandas as pd
import csv
from helper import *
from collections import defaultdict
import sys
def main():
    corpora = ["native", "nonnative", "google", "wiki"]
    grams = ["bigram", "trigram"]
    stop_words = ["True", "False"]
    gram_conv = {"trigram": "bigram", "bigram": "unigram"}
    for corpus in corpora:
        for gram in grams:
            for stop_word in stop_words:
                if (stop_word == "False") and (corpus == "google"):
                    continue
                if corpus == "google":
                    file_name = f"/ais/hal9000/masih/surprisal/all_data/{corpus}_{gram}.csv"
                else:
                    file_name = f"/ais/hal9000/masih/surprisal/all_data/{corpus}_{gram}_{stop_word}.csv"
                df = pd.read_csv(f"{corpus}_{gram}_{stop_word}.csv", encoding="utf-8")
                gram_count = get_gram_count(gram_conv[gram], corpus, stop_word)
                df["context_count"] = df["context"].apply(lambda x: gram_count[x])
                original_words = set(df["word"].unique())
                phrase_dict = defaultdict(lambda: defaultdict(int))
                all_files = file_locations(gram, corpus, stop_word)

                for file in all_files:
                    with open(file, 'r', encoding='utf-8') as (file_in):
                        if corpus == 'google':
                            reader = csv.reader(file_in, dialect='excel-tab', quoting=csv.QUOTE_NONE)
                        else:
                            reader = csv.reader(file_in)
                        for row in reader:
                            temp_gram = row[0].lower().split(' ')
                            if len(temp_gram) < 2:
                                print(temp_gram, 'skipped')
                            elif temp_gram[(-1)] in original_words:
                                temp_context = ' '.join(word for word in temp_gram[:-1])
                                phrase_dict[temp_gram[-1]][temp_context] = literal_eval(row[1])

                    file_in.close()
                df["phrase_count"] = df.apply(lambda x: phrase_dict[x.word][x.context])
                df.to_csv(file_name, index=None, encoding="utf-8")
                print(f"{corpus} {gram} done")
                sys.stdout.flush()

if __name__=="__main__":
    main()