import pandas as pd
from tools.helper import *


def get_count(gram_size, corpus, stop_words):
    """

    :param gram_size: str
    :param corpus: str
    :param stop_words: str
    :return: dictionary
    """

    files = file_locations(gram_size, corpus=corpus, stop_words=stop_words)

    df = pd.read_csv(files[0], header=None, encoding="utf-8", usecols=[1])
    total_count = df[1].sum()
    # all_gram_files = file_locations(gram_size, corpus=corpus, stop_words=stop_words)
    # for file_ in all_gram_files:
    #     with open(file_, 'r', encoding='utf-8') as (temp_file):
    #         if corpus == 'google':
    #             csv_reader = csv.reader(temp_file, dialect='excel-tab', quoting=csv.QUOTE_NONE)
    #             for row in csv_reader:
    #                 total_count += 1
    #
    #         else:
    #             csv_reader = csv.reader(temp_file)
    #             for row in csv_reader:
    #                 total_count += 1
    #
    #     temp_file.close()

    return total_count


all_counts =[]
corpora = ["native", "nonnative", "wiki", "UK","US","europe"]
grams = ["bigram", ]


for corpus in corpora:
    corpus_list =[]
    unigram_count = get_count("unigram",corpus,None)

    corpus_list.append(unigram_count)
    all_counts.append(corpus_list)
header =["unigram"]
count_df = pd.DataFrame(all_counts,index=corpora,columns =header)
count_df.to_csv("counts.csv",encoding="utf-8")