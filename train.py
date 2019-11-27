import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_model(df, model):
    """
    A function that converts text data into N x M matrix where N
    is number of data points/sentences, and N is feature space dimension(vocabulary)
    then trains a classifier to predict long or short form of word used
    :param df: Dataframe
    :param model: Classifier
    :return:

    """
    try:
        count = min(df.form.value_counts())
    except ValueError:
        count = 0
    if count < 500:
        return np.zeros(10), np.zeros(10)
    short = df[df["form"] == 0].sample(count)
    long = df[df["form"] == 1].sample(count)
    df = pd.concat([short, long], ignore_index=True)
    all_data = df[["sentence", "form"]].values
    headlines, Y = all_data[:, 0], np.array(all_data[:, 1], dtype='int64')
    vectorizor = CountVectorizer(min_df=0.001)
    X_fitted = vectorizor.fit_transform(headlines)
    X = X_fitted.toarray()

    lr_scores = cross_val_score(model, X, Y, cv=10)
    return lr_scores.mean()


def compute_scores(df, model):
    """

    :param df: Dataframe
    :param model: Classifier
    :return:
    """
    lr_scores = []

    for i in range(23):
        if i != 0:
            df1 = df[df["sense"] == i]
        else:
            df1 = df
        if model == "LR":
            m = LogisticRegression()
        else:
            m = RandomForestClassifier(n_estimators=20)

        all_score = train_model(df1, m)
        lr_scores.append(all_score)
    return lr_scores


if __name__ == "__main__":
    model = sys.argv[1]
    corpora = ["native", "nonnative", "wiki"]
    all_rf_scores = []
    all_lr_scores = {}
    for corpus in corpora:
        file_in = f"/ais/hal9000/masih/sentences/{corpus}_sentences_False.csv"
        df = pd.read_csv(file_in, encoding="utf-8")
        s = compute_scores(df, model)

        all_lr_scores[corpus] = s

    lr_results = pd.DataFrame(data=all_lr_scores)

    lr_results.to_csv("logistic_reg_results.csv", encoding="utf-8", index=None)
