import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_model(df):
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
    try:
        count = min(df.form.value_counts())
    except ValueError:
        count =0
    if count<500:
        return np.zeros(10), np.zeros(10)
    short = df[df["form"] == 0].sample(count)
    long = df[df["form"] == 1].sample(count)
    df = pd.concat([short, long], ignore_index=True)
    all_data = df[["sentence", "form"]].values
    headlines, Y = all_data[:, 0], np.array(all_data[:, 1], dtype='int64')
    vectorizor = CountVectorizer(min_df=0.01)
    X_fitted = vectorizor.fit_transform(headlines)
    X = X_fitted.toarray()
    rf = RandomForestClassifier(n_estimators=25, criterion="entropy")
    lr = LogisticRegression()
    rf_scores = cross_val_score(rf, X, Y, cv=10)
    lr_scores = cross_val_score(lr, X, Y, cv=10)
    return lr_scores,rf_scores


def compute_scores(df):
    lr_scores = []
    rf_scores = []
    for i in range(23):
        if i != 0:
            df = df[df["sense"] == i]

        all_score = train_model(df)
        lr_scores.append(all_score[0].mean())
        rf_scores.append(all_score[1].mean())
    return lr_scores, rf_scores


if __name__ == "__main__":
    corpora = ["native", "nonnative", "wiki"]
    all_rf_scores = []
    all_lr_scores = []
    for corpus in corpora:
        file_in = f"/ais/hal9000/masih/sentences/{corpus}_sentences_False.csv"
        df = pd.read_csv(file_in, encoding="utf-8")
        s = compute_scores(df)
        all_rf_scores.append(s[1])
        all_lr_scores.append(s[0])
    rf_results = pd.DataFrame(data=all_rf_scores, columns=corpora)
    lr_results = pd.DataFrame(data=all_lr_scores, columns=corpora)
    rf_results.to_csv("random_forest_results.csv", encoding="utf-8", index=None)
    lr_results.to_csv("logistic_reg_results.csv", encoding="utf-8", index=None)