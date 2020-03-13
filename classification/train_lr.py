import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


def get_sentence_vectors(df):
    """
    A method to get convert sentences into bag of words vectors
    :param df: Dataframe
        sentences, sense, and labels
    :return: Array

    """
    df["sentence"] = df.apply(lambda row: remove_mask(row["sentence"], row["form"]), axis=1)
    df = df[df["sentence"]!=0]
    all_data = df[["sentence", "form"]].values
    sentences, Y = all_data[:, 0], np.array(all_data[:, 1], dtype='int64')
    vectorizor = CountVectorizer(min_df=0.01, ngram_range=(1, 2))
    X_fitted = vectorizor.fit_transform(sentences)
    X = X_fitted.toarray()

    return X, Y


def get_single_feature_vectors(df, feature):
    df["context"] = df.apply(lambda row: find_context(row['sentence'], row['form']), axis=1)
    df[feature] = df.apply(lambda row: feature_calc(row["context"], row["word"], feature), axis=1)
    df = df[df[feature] != 0]

    all_data = df[[feature, "form"]].values
    X, Y = all_data[:, 0], np.array(all_data[:, 1], dtype='int64')
    X = X.reshape(-1, 1)

    return X, Y


def find_context(sentence, label):
    sentence_list = sentence.split()
    if label == 0:
        index = sentence_list.index("<s>")
    else:
        index = sentence_list.index("<l>")
    if index > 1:
        return " ".join(x for x in sentence_list[index - 2:index])
    else:
        return "no context"


def feature_calc(context, word, feature):
    global entropy_dict, surprisal_dict
    if feature == "entropy":
        try:
            entropy = entropy_dict[context]
            return entropy
        except KeyError:
            return 0
    else:
        try:
            surprisal = surprisal_dict[context][word]
            return surprisal
        except KeyError:
            return 0


def remove_mask(sentence, form):
    word_list = sentence.split()
    if form == 0:
        index = word_list.index("<s>")
    else:
        index = word_list.index("<l>")
    if index>0:
        new_sentence = " ".join(word for  word in word_list[:index])
    else:
        new_sentence = 0
    return new_sentence


def sample_sentences(df, sense, count):
    df = df[df["sense"] == sense]
    short = df[(df.word1.isnull() == False) & (df.word2.isnull())]
    long = df[(df.word1.isnull()) & (df.word2.isnull() == False)]
    short["word"] = short["word1"]
    long["word"] = long["word2"]
    try:
        min_count = min(len(short), len(long))
        print(min_count)
    except ValueError:
        return None
    if min_count < count:
        return None
    if min_count > 1000:
        count = 1000
    short["form"] = 0
    long["form"] = 1
    short = short.sample(count)
    long = long.sample(count)
    df = pd.concat([short, long], ignore_index=True)
    return df


def get_stack_feature_vectors(df, feature):
    df["context"] = df.apply(lambda row: find_context(row['sentence'], row['form']), axis=1)
    df[feature] = df.apply(lambda row: feature_calc(row["context"], row["word"], feature), axis=1)
    df = df[df[feature] != 0]

    all_data = df[["sentence", feature, "form"]].values
    sentences, X2, Y = all_data[:, 0], all_data[:, 1], np.array(all_data[:, 2], dtype='int64')
    vectorizor = CountVectorizer(min_df=0.01, ngram_range=(1, 2))
    X_fitted = vectorizor.fit_transform(sentences)
    X1 = X_fitted.toarray()
    X2 = X2.reshape(-1, 1)
    return X1, X2, Y


def stack_train_valid(X1, X2, Y):
    X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(X1, X2, Y, test_size=0.2)
    model_1 = LogisticRegression().fit(X1_train, Y_train)
    X1_new = model_1.predict_proba(X1_train)
    X1_test_new = model_1.predict_proba(X1_test)
    model_2 = LogisticRegression().fit(X2_train, Y_train)
    X2_new = model_2.predict_proba(X2_train)
    X2_test_new = model_2.predict_proba(X2_test)
    X_new = np.concatenate([X1_new, X2_new], axis=1)
    X_test_new = np.concatenate([X1_test_new, X2_test_new], axis=1)
    final_model = LogisticRegression().fit(X_new, Y_train)
    score = final_model.score(X_test_new, Y_test)

    return score


def get_feature(df, model):
    surprisal = defaultdict(lambda: defaultdict(float))
    for index, row in df.iterrows():
        if model in {"pmi", "stack"}:
            surprisal[row["context"]][row["word"]] = row["surprisal"] - np.log2(row["count"])
        else:
            surprisal[row["context"]][row["word"]] = row["surprisal"]

    return surprisal


def reg_train_valid(X, Y):
    model = LogisticRegression()
    scores = cross_val_score(model, X, Y, cv=5)
    return scores.mean()


def compute_scores(df, model):
    """

    :param df: Dataframe
    :param model: Classifier
    :return:
    """
    count = 800
    lr_scores = {}

    for sense in sense_dict:

        df1 = sample_sentences(df, sense, count)


        if df1 is None:
            score = 0
        else:
            if model in {"entropy", "surprisal", "pmi"}:
                print(model, sense)
                X, Y = get_single_feature_vectors(df1, model)
                score = reg_train_valid(X, Y)
            elif model == "stack":
                X1, X2, Y = get_stack_feature_vectors(df1, model)
                score = stack_train_valid(X1, X2, Y)
            else:
                X, Y = get_sentence_vectors(df1)
                score = reg_train_valid(X, Y)

        lr_scores[sense] = score
    return lr_scores


if __name__ == "__main__":
    sense_dict = {'porn': 1, 'photo': 2, 'phone': 3, 'bike': 4, 'tv': 5, 'carb': 6, 'math': 7, 'limo': 8,
                  'ref': 15, 'roach': 16, 'fridge': 17, 'exam': 18, 'chemo': 19, 'sax': 20, 'frat': 21, 'memo': 22,
                  'dorm': 9, 'kilo': 10, 'rhino': 11, 'undergrad': 12, 'hippo': 13, 'chimp': 14, 'lab': 23, 'info': 24}
    inv_map = {v: k for k, v in sense_dict.items()}

    model = sys.argv[1]
    corpora = ["native", "wiki"]
    results_file = "bert_lr_results.csv"
    classifier_df = pd.read_csv(results_file)
    for corpus in corpora:
        file_in = f"/ais/hal9000/masih/sentences/{corpus}_sentences_False.csv"
        df = pd.read_csv(file_in, encoding="utf-8", low_memory=False)

        if model in {"entropy", "surprisal", "stack", "pmi"}:
            df1 = pd.read_csv(f"/hal9000/masih/surprisal/all_data/{corpus}_trigram_False.csv", encoding='utf-8',
                              low_memory=False)
            surprisal_dict = get_feature(df1, model)
            entropy_dict = pd.Series(df1.entropy.values, index=df1.context).to_dict()
        score_dict = compute_scores(df, model)

        classifier_df[f"{corpus}_{model}_prefix"] = classifier_df["sense"].map(score_dict)

    classifier_df.to_csv(results_file, index=None, encoding="utf-8")
