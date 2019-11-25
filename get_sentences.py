import csv
import pandas as pd
import sys
import re
from string import punctuation

regex = re.compile('[%s]' % re.escape(punctuation))


def main_prog(infile, out_file,csv_file, stop_words=False):
    """

    :param out_file:
    :param stop_words:
    :param infile: str
        The location of corpus
    :param csv_file: boolean
        If true, corpus is a csv file

    :return: None
    """

    df = pd.read_csv("all_forms.csv", encoding="utf-8")  # load csv with long and short form words
    short_forms = set(list(df.short.values))
    long_forms = set(list(df.long.values))

    with open(out_file, "w") as out_file:
        with open(infile, "r") as file:
            csvout = csv.writer(out_file)
            if csv_file:
                reader = csv.reader(file)
            else:
                reader = file
            csvout.writerow(["sentence", "form", "word"])
            for line in reader:
                if csv_file:
                    cleaned_string = clean_string(line[3])
                else:
                    cleaned_string = clean_string(line)
                cleaned_words = cleaned_string.split()
                if len(cleaned_words) > 14:
                    for word in cleaned_words:
                        if word in short_forms:
                            temp = [cleaned_string.replace(word,""), 0, word]
                            csvout.writerow(temp)
                        elif word in long_forms:
                            temp = [cleaned_string.replace(word,""), 1, word]
                            csvout.writerow(temp)
                        else:
                            continue


def clean_string(string):
    global regex
    string = string.lower()

    string = regex.sub("", string)
    return string


if __name__ == "__main__":
    corpus = sys.argv[1]
    # stop = sys.argv[2]
    # stops = sys.argv[2]
    # words = pd.read_csv("function_words_127.txt", delimiter=' ', header=None)
    # words = [clean_string(w, False)[0] for w in words[0].values]

    # FUNCTION_WORDS = set(words)
    dic2 = {'yes': True, 'no': False}

    if corpus == "native":
        filein = "/ais/hal9000/ella/reddit_2018/reddit.n.sent.all.shuf.csv"
        csv_file = True
    elif corpus == "nonnative":
        filein = "/ais/hal9000/ella/reddit_2018/reddit.nn.sent.all.shuf.csv"
        csv_file = True
    elif corpus == "wiki":
        filein = "/ais/hal9000/ella/wikipedia/eng/wikipedia.txt"
        csv_file = False
    else:
        sys.exit()
    file_out = f"/ais/hal9000/masih/sentences/{corpus}_sentences_{False}.csv"
    main_prog(filein, file_out, csv_file)
