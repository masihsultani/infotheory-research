import tensorflow as tf
import torch

import sys
from collections import defaultdict

import pandas as pd

import numpy as np

from transformers import BertForMaskedLM, BertTokenizer
import warnings
warnings.filterwarnings('ignore')

"""
A very simple script that evaluates the bert classification model to predict whether a short form or long form
of a word would be used. Script is messy, lots of needs to be done.

"""


# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

df1 = pd.read_csv("all_forms.csv")
a = df1.set_index('long')["short"].to_dict()
b = df1.set_index("short")["long"].to_dict()
OTHER_WORD = {**a, **b}

sense_dict = {'porn': 1, 'photo': 2, 'phone': 3, 'bike': 4, 'tv': 5, 'carb': 6, 'math': 7, 'limo': 8,
              'ref': 15, 'roach': 16, 'fridge': 17, 'exam': 18, 'chemo': 19, 'sax': 20, 'frat': 21, 'memo': 22,
              'dorm': 9, 'kilo': 10, 'rhino': 11, 'undergrad': 12, 'hippo': 13, 'chimp': 14,
              'telephone': 3, 'refrigerator': 17, 'undergraduate': 12, 'mathematics': 7,
              'examination': 18, 'television': 5, 'photograph': 2, 'memorandum': 22, 'bicycle': 4,
              'pornography': 1, 'fraternity': 21, 'limousine': 8, 'referee': 15, 'saxophone': 20,
              'carbohydrate': 6, 'chemotherapy': 19, 'hippopotamus': 13, 'cockroach': 16,
              'kilogram': 10, 'rhinoceros': 11, 'dormitory': 9, 'chimpanzee': 14,'lab':23,'laboratory':23,'info':24,
              'information':24}
sense_dict2 = {'all': 0, 'porn': 1, 'photo': 2, 'phone': 3, 'bike': 4, 'tv': 5, 'carb': 6, 'math': 7, 'limo': 8,
               'ref': 15, 'roach': 16, 'fridge': 17, 'exam': 18, 'chemo': 19, 'sax': 20, 'frat': 21, 'memo': 22,
               'dorm': 9, 'kilo': 10, 'rhino': 11, 'undergrad': 12, 'hippo': 13, 'chimp': 14, 'lab':23, 'info':24}
inv_map = {v: k for k, v in sense_dict2.items()}

def load_data(df):

    df["length"] = df["sentence"].apply(lambda x: len(x.split()))
    df = df[df["length"]<500]
    
    short = df[(df.word1.isnull()==False) & (df.word2.isnull())]
    long = df[(df.word1.isnull()) & (df.word2.isnull()==False)]
    try:
        count = min(len(short),len(long))
    except ValueError:
        return [], []
    if count > 1000:
        count = 1000
    long = long.sample(count)
    short = short.sample(count)
    short["sentence"] = short["sentence"].apply(lambda x: x.replace("<s>", "[MASK]"))
    long["sentence"] = long["sentence"].apply(lambda x: x.replace("<l>", "[MASK]"))
    short["label"] = short["word1"]
    long["label"] = long["word2"]
    df = pd.concat([short, long], ignore_index=True)

    sentences = df["sentence"].values
    labels = df["label"].values
    return sentences, labels

# Load pre-trained model tokenizer (vocabulary)


def build_tokens(tokenizer, sentences):
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        tokens = tokenizer.tokenize(sent)
        if len(tokens) == 0:
            continue
        if tokens[0] != CLS:
            tokens = [CLS] + tokens
        if tokens[-1] != SEP:
            tokens.append(SEP)
        if len(tokens)>510:
            continue

        # Add the encoded sentence to the list.
        input_ids.append(tokens)
    return input_ids




PAD, MASK, CLS, SEP = '[PAD]', '[MASK]', '[CLS]', '[SEP]'


def to_bert_input(tokens, bert_tokenizer):
    token_idx = torch.tensor(bert_tokenizer.convert_tokens_to_ids(tokens))
    sep_idx = tokens.index('[SEP]')
    segment_idx = token_idx * 0
    segment_idx[(sep_idx + 1):] = 1
    mask = (token_idx != 0)
    return token_idx.unsqueeze(0), segment_idx.unsqueeze(0), mask.unsqueeze(0)


def bert_predict(bert_model, tokenizer, tokens, label):
    global OTHER_WORD, sense_dict, inv_map

    token_idx, segment_idx, mask = to_bert_input(tokens, tokenizer)
    token_idx = token_idx.to(device)
    segment_idx = segment_idx.to(device)
    with torch.no_grad():
        logits = bert_model(token_idx, segment_idx, masked_lm_labels=None)

    logits = logits[0].squeeze().detach().cpu()

    probs = torch.softmax(logits, dim=-1)
    #     predicted_index = torch.argmax(predictions[0, masked_index]).item()
    #     predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    idx = tokens.index('[MASK]')

    #             topk_prob, topk_indices = torch.topk(probs[idx, :], 5)
    label_index = tokenizer.convert_tokens_to_ids(label)
    other_label = tokenizer.convert_tokens_to_ids(OTHER_WORD[label])
    prob1 = probs[idx, label_index].item()
    prob2 = probs[idx, other_label].item()
    return np.array([prob1, prob2])


def compute_acc(token_list, labels, model, tokenizer):
    word_scores =0
    for i in range(len(token_list)):
        probs = bert_predict(model, tokenizer, token_list[i], labels[i])
        if np.argmax(probs) == 0:
            word_scores += 1
    return word_scores/len(labels)

def compute_scores(df, model, tokenizer):
    """

    :param df: Dataframe
    :param model: Classifier
    :return:
    """
    bert_scores = []

    for i in range(1,25):

        df1 = df[df["sense"] == i]
        sentences, labels = load_data(df1)
        token_list = build_tokens(tokenizer, sentences)
        scores = compute_acc(token_list, labels, model, tokenizer)


        bert_scores.append(scores)
        print(i)
        sys.stdout.flush()
    return bert_scores


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    bert_model.cuda()
    bert_model.eval()
    corpora = ["native","wiki"]
    all_lr_scores = {}
    for corpus in corpora:
        print(corpus)
        sys.stdout.flush()
        file_in = f"/hal9000/masih/sentences/{corpus}_sentences_False.csv"
        df = pd.read_csv(file_in)
        s = compute_scores(df, bert_model, tokenizer)

        all_lr_scores[f"{corpus}_bert"] = s

    bert_results = pd.DataFrame(data=all_lr_scores)
    bert_results["sense"] = bert_results.index.to_series().apply(lambda x: inv_map[x+1])
    bert_results.to_csv("bert_results.csv", index=None)

