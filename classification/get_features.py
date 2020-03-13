import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import pandas as pd

sense_dict = {'porn': 1, 'photo': 2, 'phone': 3, 'bike': 4, 'tv': 5, 'carb': 6, 'math': 7, 'limo': 8,
              'ref': 15, 'roach': 16, 'fridge': 17, 'exam': 18, 'chemo': 19, 'sax': 20, 'frat': 21, 'memo': 22,
              'dorm': 9, 'kilo': 10, 'rhino': 11, 'undergrad': 12, 'hippo': 13, 'chimp': 14}

sense_dict2 = {v: k for k, v in sense_dict.items()}

def train_all(df, model):
    short = df[(df.word1.isnull() == False) & (df.word2.isnull())]
    long = df[(df.word1.isnull()) & (df.word2.isnull() == False)]
    short["form"] = 0
    long["form"] = 1
    short["sentence"] = short["sentence"].apply(lambda x: x.replace("<s>", ""))
    long["sentence"] = long["sentence"].apply(lambda x: x.replace("<l>", ""))

    short = short.groupby('word1', group_keys=False).apply(pd.DataFrame.sample, n=100)

    long = long.groupby('word2', group_keys=False).apply(pd.DataFrame.sample, n=100)

    df = pd.concat([short, long], ignore_index=True)
    all_data = df[["sentence", "form"]].values
    headlines, Y = all_data[:, 0], np.array(all_data[:, 1], dtype='int64')
    vectorizor = CountVectorizer(min_df=0.01, ngram_range=(1, 2))
    X_fitted = vectorizor.fit_transform(headlines)
    X = X_fitted.toarray()
    model.fit(X, Y)

    coefs = model.coef_[0]
    features_names = vectorizor.get_feature_names()
    sorted_features = [x for _, x in sorted(zip(coefs, features_names), reverse=True)]
    longs = sorted_features[:20]
    shorts = list(reversed(sorted_features[-20:]))

    return longs, shorts


def train_model(df, model):
    """
    A function that converts text data into M x N matrix where M
    is number of headlines, and N is feature space dimension(vocabulary)

    Returns 2d list containing training(70%), validation(15%) and test(15&) data
    load_data[0] is location of X values (train,val,test)
    load_data[1] is location of Y values (train,val,test)
    real_file: str
        location of real headline file
    fake_file: str
        location of fake headline file
    """
    short = df[(df.word1.isnull() == False) & (df.word2.isnull())]
    long = df[(df.word1.isnull()) & (df.word2.isnull() == False)]
    try:
        count = min(len(short), len(long))
    except ValueError:
        return [],[]
    if count < 1000:
        return [],[]
    short["form"] = 0
    long["form"] = 1
    short["sentence"] = short["sentence"].apply(lambda x: x.replace("<s>", ""))
    long["sentence"] = long["sentence"].apply(lambda x: x.replace("<l>", ""))
    short = short.sample(count)
    long = long.sample(count)
    df = pd.concat([short, long], ignore_index=True)

    all_data = df[["sentence", "form"]].values
    headlines, Y = all_data[:, 0], np.array(all_data[:, 1], dtype='int64')
    vectorizor = CountVectorizer(min_df=0.01, ngram_range=(1, 2))
    X_fitted = vectorizor.fit_transform(headlines)
    X = X_fitted.toarray()
    model.fit(X, Y)

    coefs = model.coef_[0]
    features_names = vectorizor.get_feature_names()
    sorted_features = [x for _, x in sorted(zip(coefs, features_names), reverse=True)]
    longs = sorted_features[:20]
    shorts = list(reversed(sorted_features[-20:]))

    return longs, shorts



if __name__ == '__main__':
    corpora = ["native", "nonnative", "wiki"]
    for corpus in corpora:
        file_out = f"{corpus}_top20_features.txt"
        df = pd.read_csv(f"/ais/hal9000/masih/sentences/{corpus}_sentences_False.csv", encoding='utf-8')
        with open(file_out, "w") as infile:

            for i in range(0, 23):
                if i != 0:

                    df1 = df[df["sense"] == i]
                    string = f"Sense = {sense_dict2[i]} \n"
                    infile.writelines(string)
                    lr = LogisticRegression(solver='lbfgs', max_iter=200)
                    lr_score = train_model(df1, lr)
                else:
                    df1 = df
                    string = "Comparing all words: \n"
                    infile.writelines(string)
                    lr = LogisticRegression(solver='lbfgs', max_iter=200)
                    lr_score = train_all(df1, lr)

                # rf = RandomForestClassifier(n_estimators=25, criterion="entropy")

                long_string = f"Long form top 20: {str(lr_score[0])} \n"
                short_string = f"Short from top 20: {str(lr_score[1])} \n"
                infile.writelines(long_string)
                infile.writelines(short_string)
                infile.writelines("\n")
                print(i)
        infile.close()
