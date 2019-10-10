import csv
import pandas as pd


def main_prog(infile, csv_file=True):
    """

    :param infile: str
        The location of corpus
    :param csv_file: boolean
        If true, corpus is a csv file

    :return: None
    """

    df = pd.read_csv("all_forms.csv", encoding="utf-8")  # load csv with long and short form words
    all_forms = set(list(df.short.values) + list(df.long.values))
    fileout = "/ais/hal9000/masih/native_sentences.txt"
    with open(fileout, "w") as file_out:
        with open(infile, "r") as file:

            if csv_file:
                file = csv.reader(file, delimiter=',')
                for line in file:
                    sentences = line[-1].lower().split('.')
                    for x in sentences:
                        for y in x.split():
                            if y in all_forms:
                                file_out.write(x)

                file.close()
                file_out.close()


if __name__ == "__main__":
    filein = "/ais/hal9000/ella/reddit_2018/reddit.n.sent.all.shuf.csv"
    main_prog(filein)
