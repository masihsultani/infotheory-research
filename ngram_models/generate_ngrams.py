import json
import re
import sys
from collections import Counter
from string import punctuation

import pandas as pd

regex = re.compile('[%s]' % re.escape(punctuation))

"""
A script to generate ngrams from native or non native reddit corpus

"""

def main_prog(infile, corpus, gram_size, stop_word, csv_file=True):
    """

    :param infile: str
        The location of corpus
    :param corpus: str
        string native or nonnative for saving final file
    :param gram_size: str
        bigram or trigram to count bigram or trigram count for corpus
    :param csv_file: boolean
        If true, corpus is a csv file

    :param stop_word: boolean
        True to remove stop_words before computing bigrams and trigram counts
    :return: None
    """


    gram_counter = Counter()#bounter(size_mb=35840)
    for file_in in infile:
        with open(file_in, "r") as file:

            if csv_file:
                file = csv.reader(file, delimiter=',')
                for line in file:
                    clean_list = clean_string(line[-1], stop_word) #clean the string from punctuations and stop words
                    all_grams = gen_grams(clean_list, gram_size)
                    gram_counter.update(" ".join(tuple) for tuple in all_grams)
            elif ".out" in infile:
                for line in file:
                    s = line
                    raw_data = json.loads(s)
                    if "body" in raw_data.keys():
                        raw_text = raw_data["body"]

                    elif "selftext" in raw_data.keys():

                        raw_text = raw_data["selftext"]
                    else:
                        continue

                    clean_list = clean_string(raw_text, stop_word)  # clean the string from punctuations and stop words
                    all_grams = gen_grams(clean_list, gram_size)
                    gram_counter.update(" ".join(tuple) for tuple in all_grams)

            else:
                for line in file:
                    clean_list = clean_string(line, stop_word) #clean the string from punctuations and stop words
                    all_grams = gen_grams(clean_list, gram_size)
                    gram_counter.update(" ".join(tuple) for tuple in all_grams)

    if gram_size == "unigram":
        file_out = f'/ais/hal9000/masih/surprisal/{corpus}/{gram_size}_{corpus}.csv'
    else:
        file_out = f'/ais/hal9000/masih/surprisal/{corpus}/{gram_size}_{corpus}_{stop_word}.csv'

    with open(file_out, 'w') as gram_file:
        writer = csv.writer(gram_file)
        for key, value in gram_counter.items():
            writer.writerow([key, value])
    gram_file.close()
    print(f"{gram_size}_saved")

def gen_grams(clean_list, gram):

    """
    Helper function to generate ngrams
    :param clean_list: list of words
    :param gram: size of ngram to generate
    :return: zip object

    """
    if gram == "bigram":

        bigrams = zip(clean_list, clean_list[1:])  # compute bigrams
        return bigrams

    elif gram=="trigram":
        trigrams = zip(clean_list, clean_list[1:], clean_list[2:])  # compute trigrams
        return trigrams

    else:
        return zip(clean_list)

def clean_string(string, stop_words):
    """
    Helper function to clean a string and tokenize it
        :param string: str
            The str to clean by removing punctuation
        :param stop_words: boolean
            remove stop or not
        :return: list
            Cleaned list of tokens to be returned
    """

    global regex, FUNCTION_WORDS
    string = string.lower()

    string = regex.sub("", string)
    string = string.split()
    if stop_words:
        string = [x for x in string if x not in FUNCTION_WORDS]
    return string


if __name__ == "__main__":
    corpus = sys.argv[1]
    #stop = sys.argv[2]
    stops = ['yes', 'no']
    words = pd.read_csv("function_words_127.txt", delimiter=' ', header=None)
    words = [clean_string(w, False)[0] for w in words[0].values]
    grams=["trigram","bigram", "unigram"]
    FUNCTION_WORDS = set(words)
    for gram in grams:
        for stop in stops:
            #data = sys.argv[1] # native or non native corpus
            if (gram =="unigram")  and(stop=="yes"):
                continue

            dic2 = {'yes': True, 'no': False}

            if corpus== "native":
                filein = "/ais/hal9000/ella/reddit_2018/reddit.n.sent.all.shuf.csv"
                main_prog(filein, corpus, gram_size=gram, stop_word=dic2[stop])
            elif corpus== "nonnative":
                filein= "/ais/hal9000/ella/reddit_2018/reddit.nn.sent.all.shuf.csv"
                main_prog(filein, corpus, gram_size=sys.argv[2], stop_word=dic2[stop])
            elif corpus== "learners":
                filein = "/ais/hal9000/ella/learners_data/learners.txt"
                main_prog(filein, corpus, gram_size=sys.argv[2], stop_word=dic2[stop], csv_file=False)
            elif corpus== "wiki":
                filein = "/ais/hal9000/ella/wikipedia/eng/wikipedia.txt"
                main_prog(filein, corpus, gram_size=sys.argv[2], stop_word=dic2[stop], csv_file=False)
            elif corpus == "US":
                filein = f"/ais/hal9000/ella/reddit_2018/reddit_raw_data/reddit.{corpus}.raw.out"
                main_prog(filein, corpus, sys.argv[2], stop_word=dic2[stop], csv_file=False)
            elif corpus == "UK":
                filein = [f"/ais/hal9000/masih/surprisal/UK/unitedkingdom.comment.json.out", f"/ais/hal9000/masih/surprisal/UK/casualuk.comment.json.out"]
                main_prog(filein, corpus, sys.argv[2], stop_word=dic2[stop], csv_file=False)
            elif corpus == "europe":
                filein = [f"/ais/hal9000/masih/surprisal/UK/europe.comment.json.out", f"/ais/hal9000/masih/surprisal/UK/askeurope.comment.json.out"]
                main_prog(filein, corpus, gram, stop_word=dic2[stop], csv_file=False)
            else:
                sys.exit()


