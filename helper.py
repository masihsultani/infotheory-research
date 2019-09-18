# uncompyle6 version 3.4.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.7 (default, Oct 22 2018, 11:32:17) 
# [GCC 8.2.0]
# Embedded file name: /h/118/masih/surpisal/info_theory_helper.py
# Compiled at: 2019-09-13 12:12:10
# Size of source mod 2**32: 4894 bytes
import csv
from collections import defaultdict
from ast import literal_eval
import sys

gram_conv = {"trigram": "bigram", "bigram": "unigram"}


def get_context(words, gram, corpus, stop_words, keep_words=False):
    """
    Helper function to get contexts for a set of words
    :param keep_words:
    :param gram:
    :param stop_words:
    :param corpus: str
    :param words: set
    :return:
    """
    contexts = set()
    all_files = file_locations(gram, corpus, stop_words)
    for infile in all_files:
        with open(infile, mode='r', encoding='utf-8') as (inputfile):
            if corpus == 'google':
                csvfile = csv.reader(inputfile, dialect='excel-tab', quoting=csv.QUOTE_NONE)
            else:
                csvfile = csv.reader(inputfile)
            for row in csvfile:
                templst = row[0].lower().split(' ')
                if templst[(-1)] in words:
                    if keep_words:
                        contexts.add(row[0].lower())
                    else:
                        contexts.add(' '.join(word for word in templst[:-1]))

            inputfile.close()
    return contexts


def file_locations(gram, corpus, stop_words=None):
    """

    :param stop_words:
    :param corpus:
    :param gram: str
    bigram or trigram
    :return: list
    list of string for file locations
    """

    infolder = f"/ais/hal9000/masih/surprisal/{corpus}/"
    lst = []
    if corpus == 'google':
        infolder = "/w/nobackup/131/web1t-5gram-v1"
        if gram == 'bigram':
            infile_2 = f"{infolder}/dvd1/data/2gms/2gm.idx"
            with open(infile_2, encoding='utf-8') as (file_1):
                csv_file = csv.reader(file_1, delimiter='\t')
                for row in csv_file:
                    string = row[0].strip('\\.gz')
                    file_loc = f"{infolder}/dvd1/data/2gms/{string}"
                    lst.append(file_loc)

        elif gram == 'trigram':
            count = 0
            infile_1 = f"{infolder}/dvd1/data/3gms/3gm.idx"
            with open(infile_1, encoding='utf-8') as (file_1):
                csv_file = csv.reader(file_1, delimiter='\t')
                for row in csv_file:
                    if count < 46:
                        string = row[0].strip('\\.gz')
                        file_loc = f"{infolder}/dvd1/data/3gms/{string}"
                        lst.append(file_loc)
                        count += 1
                    else:
                        string = row[0].strip('\\.gz')
                        file_loc = f"{infolder}/dvd2/data/3gms/{string}"
                        lst.append(file_loc)

        elif gram == 'unigram':
            lst.append(f"{infolder}/dvd1/data/1gms/vocab")
    elif gram == 'bigram' or gram == 'trigram':
        file_in = f"{infolder}{gram}_{corpus}_{stop_words}.csv"
        lst.append(file_in)
    elif gram == 'unigram':
        file_in = f"{infolder}unigram_{corpus}.csv"
        lst.append(file_in)
    return lst


def get_gram_count(gram_size, corpus, stop_words):
    """

    :param gram_size: str
    :param corpus: str
    :param stop_words: str
    :return: dictionary
    """
    gram_count = defaultdict(int)
    all_gram_files = file_locations(gram_size, corpus=corpus, stop_words=stop_words)
    for file_ in all_gram_files:
        with open(file_, 'r', encoding='utf-8') as (temp_file):
            if corpus == 'google':
                csv_reader = csv.reader(temp_file, dialect='excel-tab', quoting=csv.QUOTE_NONE)
                for row in csv_reader:
                    gram_count[row[0].lower()] += literal_eval(row[1])

            else:
                csv_reader = csv.reader(temp_file)
                for row in csv_reader:
                    gram_count[row[0]] = literal_eval(row[1])

        temp_file.close()

    return gram_count


def get_ngram_probs(gram_size, context_count, words, corpus, stop_words):
    """

    :param gram_size:
    :param context_count:
    :param words:
    :param corpus:
    :param stop_words:
    :return:
    """
    prime_probs = defaultdict(lambda: defaultdict(float))
    all_files = file_locations(gram_size, corpus=corpus, stop_words=stop_words)
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
                elif temp_gram[(-1)] in words:
                    temp_context = ' '.join(word for word in temp_gram[:-1])
                    try:
                        prime_probs[temp_gram[(-1)]][temp_context] += literal_eval(row[1]) / context_count[temp_context]
                    except ZeroDivisionError:
                        print(temp_context)
                        sys.stdout.flush()

        file_in.close()

    return prime_probs
# okay decompiling info_theory_helper.cpython-36.pyc
