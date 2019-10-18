import pandas as pd
from helper import *
from ast import literal_eval
import sys
def get_count(gram_size, corpus, stop_words):
    """

    :param gram_size: str
    :param corpus: str
    :param stop_words: str
    :return: dictionary
    """
    total_count =0
    all_gram_files = file_locations(gram_size, corpus=corpus, stop_words=stop_words)
    for file_ in all_gram_files:
        with open(file_, 'r', encoding='utf-8') as (temp_file):
            if corpus == 'google':
                csv_reader = csv.reader(temp_file, dialect='excel-tab', quoting=csv.QUOTE_NONE)
                for row in csv_reader:
                    total_count += literal_eval(row[1])

            else:
                csv_reader = csv.reader(temp_file)
                for row in csv_reader:
                    total_count += literal_eval(row[1])

        temp_file.close()

    return total_count

if __name__=="__main__":
    all_counts =[]
    corpora = ["native", "nonnative", "google", "wiki"]
    grams = ["bigram", "trigram"]
    stop_words = ["True", "False"]

    for corpus in corpora:
        corpus_list =[]
        for gram in grams:
            for stop_word in stop_words:
                if corpus=="google" and stop_word=="True":
                    count=0
                    corpus_list.append(0)
                else:
                    count = get_count(gram,corpus,stop_word)
                    corpus_list.append(count)
                sys.stdout.flush()

                print(f"{corpus} {gram} {count}")
                sys.stdout.flush()
        unigram_count = get_count("unigram",corpus,None)
        corpus_list.append(unigram_count)
        all_counts.append(corpus_list)
    header =["bigram True","bigram False","trigram True","trigram False","unigram"]
    count_df = pd.DataFrame(all_counts,index=corpora,columns =header)
    count_df.to_csv("word_counts.csv",encoding="utf-8")