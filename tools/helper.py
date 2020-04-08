import csv
from collections import defaultdict
from ast import literal_eval
import sys

"""
Helper functions
"""

gram_conv = {"trigram": "bigram", "bigram": "unigram"}

WORD_SENSE_DICT = {'porn': 1, 'photo': 2, 'phone': 3, 'bike': 4, 'tv': 5, 'carb': 6, 'math': 7, 'limo': 8,
                   'ref': 15, 'roach': 16, 'fridge': 17, 'exam': 18, 'chemo': 19, 'sax': 20, 'frat': 21, 'memo': 22,
                   'dorm': 9, 'kilo': 10, 'rhino': 11, 'undergrad': 12, 'hippo': 13, 'chimp': 14,
                   'telephone': 3, 'refrigerator': 17, 'undergraduate': 12, 'mathematics': 7,
                   'examination': 18, 'television': 5, 'photograph': 2, 'memorandum': 22, 'bicycle': 4,
                   'pornography': 1, 'fraternity': 21, 'limousine': 8, 'referee': 15, 'saxophone': 20,
                   'carbohydrate': 6, 'chemotherapy': 19, 'hippopotamus': 13, 'cockroach': 16,
                   'kilogram': 10, 'rhinoceros': 11, 'dormitory': 9, 'chimpanzee': 14, 'lab': 23, 'laboratory': 23,
                   'info': 24, 'information': 24}


def get_context(words, gram, corpus,keep_word, stop_words):
    """
    Helper function to get contexts for a set of words
    :param keep_word: bool
        when returning contexts, keep original word?
    :param gram: str
        ngram length (bigram/trigram)
    :param stop_words: str
        remove stop_words or not
    :param corpus: str
        name of the corpus
    :param words: set
        set of words to find contexts for
    :return: set
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
                    if keep_word:
                        contexts.add(row[0].lower())
                    else:
                        contexts.add(' '.join(word for word in templst[:-1]))

            inputfile.close()
    return contexts


def file_locations(gram, corpus, stop_words=None):
    """
    Returns list of file location where n gram counts are stored
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
    Get a dictionary count for n gram phrases
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
            else:
                csv_reader = csv.reader(temp_file)
            for row in csv_reader:
                gram_count[row[0].lower()] += literal_eval(row[1])

        temp_file.close()

    return gram_count


def get_ngram_probs(gram_size, context_count, words, corpus, stop_words):
    """
    Memory intensive function that loads conditional probability of n gram
    :param gram_size: str
    :param context_count: dictionary
    :param words: list
    :param corpus: str
    :param stop_words: str
    :return: dictionary nested dict[word][context] = prob
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


def get_context_counts(context_set, gram_size, corpus, stop_words):
    """

    :param context_set:
    :param gram_size:
    :param corpus:
    :param stop_words:
    :return:
    """
    context_counts = defaultdict(float)
    all_gram_files = file_locations(gram_size, corpus=corpus, stop_words=stop_words)
    for file_ in all_gram_files:
        with open(file_, 'r', encoding='utf-8') as (temp_file):
            if corpus == 'google':
                csv_reader = csv.reader(temp_file, dialect='excel-tab', quoting=csv.QUOTE_NONE)
            else:
                csv_reader = csv.reader(temp_file)
            for row in csv_reader:
                if row[0].lower() in context_set:
                    context_counts[row[0].lower()] += literal_eval(row[1])
        temp_file.close()

    return context_counts
# okay decompiling info_theory_helper.cpython-36.pyc
