from tqdm import tqdm
import logging
from config import SearchConfig
from collections import Counter
import pandas as pd
import re
import nltk
import json
import numpy as np
from w2v import Word2VecModel
from vocabulary import Vocabulary
import os
# import jieba
# from transformers import BertTokenizer
from nltk import word_tokenize


def convert_to_features(config, data, word_vocab, char_vocab, use_char=True):

    sent_limit = config.limit
    char_limit = config.char_limit

    data_idx = word_vocab.sent2seq(data, max_length=sent_limit)
    data_idx = np.array(data_idx)

    data_char_idx = np.zeros([sent_limit, char_limit], dtype=np.int32)
    for i, word in enumerate(data):
        if i >= sent_limit:
            continue
        chars = char_vocab.sent2seq(list(word), max_length=char_limit)
        for j, char in enumerate(chars):
            data_char_idx[i][j] = char

    return data_idx, data_char_idx


def build_features(config, examples, data_type, out_file, word_vocab, char_vocab, is_test=False):
    print("Processing {} examples...".format(data_type))
    total = 0
    total_ = 0
    meta = {}
    N = len(examples[0])
    data_idxs = []
    data_char_idxs = []
    labels = []
    ids = []
    for n, example in enumerate(examples):
        total += 1
        total_ += 1
        review = example['review']
        if len(review) == 1:
            data_idx, data_char_idx = convert_to_features(config, review[0], word_vocab, char_vocab)
            data_idxs.append([data_idx])
            data_char_idxs.append([data_char_idx])
            labels.append(example['label'])
            ids.append(n)
        else:
            data_idx_0, data_char_idx_0 = convert_to_features(config, review[0], word_vocab, char_vocab)
            data_idx_1, data_char_idx_1 = convert_to_features(config, review[1], word_vocab, char_vocab)
            data_idxs.append([data_idx_0, data_idx_1])
            data_char_idxs.append([data_char_idx_0, data_char_idx_1])
            labels.append(example['label'])
            ids.append(n)

    print("data_idxs", np.array(data_idxs).shape)
    print("data_char_idxs", np.array(data_char_idxs).shape)
    np.savez(out_file, data_idxs=np.array(data_idxs), data_char_idxs=np.array(data_char_idxs),
             labels=np.array(labels), ids=np.array(ids))
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    return meta


class JiebaTokenizer():
    def __init__(self):
        jieba.lcut("我爱北京天安门")

    def tokenize(self, text):
        return jieba.lcut(text)


def sst_reader(scr_data_path, word_counter, char_counter, mode="train"):
    st = scr_data_path+'/train.tsv'
    if mode == "test":
        st = scr_data_path+'/dev.tsv'
    if mode == "dev":
        st = scr_data_path+'/dev.tsv'

    # tokenizer = BertTokenizer(vocab_file = 'bert-base-uncased-vocab.txt')

    def clean_text(text):
        text = text.lower()
        text = text.replace("-", " - ").replace("\/", " / ").strip()
        text = word_tokenize(text)
        text = [i for i in text if i != "" and i != " "]
        for token in text:
            word_counter[token] += 1
            for char in list(token):
                char_counter[char] += 1
        return text

    data = []
    with open(st, "r") as file:
        for index, line in enumerate(file):
            if index == 0:
                continue
            sts = line.strip().split("\t")
            label = int(sts[-1])
            review = clean_text(" ".join(sts[:-1]))
            data.append({"review": [review], "label": label})

    return data


def amazon_reader(scr_data_path, word_counter, char_counter, mode="train"):
    st = scr_data_path+'/train.txt'
    if mode == "test":
        st = scr_data_path+'/test.txt'
    if mode == "dev":
        st = scr_data_path+'/dev.txt'

    # tokenizer = BertTokenizer(vocab_file = 'bert-base-uncased-vocab.txt')

    def clean_text(text):
        text = text.lower()
        text = text.replace("-", " - ").replace("\/", " / ").strip()
        text = word_tokenize(text)
        text = [i for i in text if i != "" and i != " "]
        for token in text:
            word_counter[token] += 1
            for char in list(token):
                char_counter[char] += 1
        return text

    data = []
    with open(st, "r", errors='ignore') as file:
        for line in file:
            sts = line.strip().split('\t')
            try:
                label = int(sts[0])
                review = clean_text(sts[1])
                data.append({"review": [review], "label": label})
            except:
                continue
    return data

def mrpc_reader(scr_data_path, word_counter, char_counter, mode="train"):
    st = scr_data_path+'/train.tsv'
    if mode == "test":
        st = scr_data_path+'/test.tsv'
    if mode == "dev":
        st = scr_data_path+'/dev.tsv'

    # tokenizer = BertTokenizer(vocab_file = 'bert-base-uncased-vocab.txt')

    def clean_text(text):
        text = text.lower()
        text = text.replace("-", " - ").replace("\/", " / ").strip()
        text = word_tokenize(text)
        text = [i for i in text if i != "" and i != " "]
        for token in text:
            word_counter[token] += 1
            for char in list(token):
                char_counter[char] += 1
        return text

    data = []
    with open(st, "r", errors='ignore') as file:
        for line in file:
            sts = line.strip().split('\t')
            try:
                label = int(sts[0])
                s1 = clean_text(sts[3])
                s2 = clean_text(sts[4])
                data.append({"review": [s1, s2], "label": label})
            except:
                continue
    return data


def rte_reader(scr_data_path, word_counter, char_counter, mode="train"):
    st = scr_data_path+'/train.tsv'
    if mode == "test":
        st = scr_data_path+'/test.tsv'
    if mode == "dev":
        st = scr_data_path+'/dev.tsv'
    l = { "entailment": 0, "not_entailment": 1}
    # tokenizer = BertTokenizer(vocab_file = 'bert-base-uncased-vocab.txt')

    def clean_text(text):
        text = text.lower()
        text = text.replace("-", " - ").replace("\/", " / ").strip()
        text = word_tokenize(text)
        text = [i for i in text if i != "" and i != " "]
        for token in text:
            word_counter[token] += 1
            for char in list(token):
                char_counter[char] += 1
        return text

    data = []
    with open(st, "r", errors='ignore') as file:
        for line in file:
            sts = line.strip().split('\t')
            try:
                label = l[sts[-1]]
                s1 = clean_text(sts[1])
                s2 = clean_text(sts[2])
                data.append({"review": [s1, s2], "label": label})
            except:
                continue
    return data


def db_reader(scr_data_path, word_counter, char_counter, mode="train"):
    st = scr_data_path+'/train.csv'
    if mode == "test":
        st = scr_data_path+'/test.csv'
    if mode == "dev":
        return None

    # tokenizer = BertTokenizer(vocab_file = 'bert-base-uncased-vocab.txt')

    def clean_text(text):
        text = text.lower()
        text = text.replace("-", " - ").replace("\\/", " / ").replace("\n", " ").replace("\\", " ").strip()
        text = word_tokenize(text)
        text = [i for i in text if i != "" and i != " "]
        for token in text:
            word_counter[token] += 1
            for char in list(token):
                char_counter[char] += 1
        return text

    data = []

    df = pd.read_csv(st, header=None)
    df = df.fillna('')
    df.columns = ["label", "title", "review"]
    for k, row in df.iterrows():
        label = int(row["label"])-1
        # print(row["title"])
        # print(row["review"])
        review = clean_text(row["title"]+" "+row["review"])
        data.append({"review": review, "label": label})

    return data

def yelp_reader(scr_data_path, word_counter, char_counter, mode="train"):
    st = scr_data_path+'/train.csv'
    if mode == "test":
        st = scr_data_path+'/test.csv'
    if mode == "dev":
        return None

    # tokenizer = BertTokenizer(vocab_file = 'bert-base-uncased-vocab.txt')

    def clean_text(text):
        text = text.lower()
        text = text.replace("-", " - ").replace("\\/", " / ").replace("\n", " ").replace("\\", " ").strip()
        text = word_tokenize(text)
        text = [i for i in text if i != "" and i != " "]
        for token in text:
            word_counter[token] += 1
            for char in list(token):
                char_counter[char] += 1
        return text

    data = []

    df = pd.read_csv(st, header=None)
    df = df.fillna('')
    df.columns = ["label", "review"]
    for k, row in df.iterrows():
        label = int(row["label"])-1
        # print(row["title"])
        # print(row["review"])
        review = clean_text(row["review"])
        data.append({"review": review, "label": label})

    return data


def yahoo_reader(scr_data_path, word_counter, char_counter, mode="train"):
    st = scr_data_path+'/train.csv'
    if mode == "test":
        st = scr_data_path+'/test.csv'
    if mode == "dev":
        return None

    # tokenizer = BertTokenizer(vocab_file = 'bert-base-uncased-vocab.txt')

    def clean_text(text):
        text = text.lower()
        text = text.replace("-", " - ").replace("\\/", " / ").replace("\n", " ").replace("\\", " ").strip()
        text = word_tokenize(text)
        text = [i for i in text if i != "" and i != " "]
        for token in text:
            word_counter[token] += 1
            for char in list(token):
                char_counter[char] += 1
        return text

    data = []

    df = pd.read_csv(st, header=None)
    df = df.fillna('')
    df.columns = ["label", "title", "review", "answer"]
    for k, row in df.iterrows():
        label = int(row["label"])-1
        # print(row["title"])
        # print(row["review"])
        review = clean_text(row["title"]+" "+row["review"])
        data.append({"review": review, "label": label})

    return data


def imdb_reader(scr_data_path, word_counter, char_counter, mode="train"):
    st = scr_data_path+'/imdb_master.csv'

    # tokenizer = BertTokenizer(vocab_file = 'bert-base-uncased-vocab.txt')

    def clean_text(text):
        text = text.lower()
        text = text.replace("-", " - ").replace("<br />", "").strip()
        text = word_tokenize(text)
        text = [i for i in text if i != "" and i != " "]
        for token in text:
            word_counter[token] += 1
            for char in list(token):
                char_counter[char] += 1
        return text
    df = pd.read_csv(st, encoding="latin-1")
    data = []
    for i, row in df.iterrows():
        if row['type'] == mode and row['label'] != "unsup":
            review = clean_text(row["review"])
            label = 1 if row['label'] == "pos" else 0
            data.append({"review": review, "label": label})
    if len(data) == 0:
        return None
    return data


def load(filename, message=None):
    if message is not None:
        print("Loading {}...".format(message))

    with open(filename, "r", encoding="utf-8") as fh:
        examples = []
        for line in fh:
            example = json.loads(line.strip())
            examples.append(example)
        return examples


def save(filename, examples, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    with open(filename, "w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False)+"\n")


dataset2processer = {
    # "sst": sst_reader,
    # "imdb": imdb_reader,
    "dbpedia": db_reader,
    "amazon_f": db_reader,
    "amazon": db_reader,
    "yelp_f": yelp_reader,
    "yelp": yelp_reader,
    "yahoo": yahoo_reader,
    "ag_news": db_reader,
    "book": amazon_reader,
    "baby": amazon_reader,
    "MRPC": mrpc_reader,
    "SST-2": sst_reader,
    "RTE": rte_reader,
}

if __name__ == "__main__":
    config = SearchConfig()
    word_counter = Counter()
    char_counter = Counter()
    datasets = [config.datasets]
    to_dir = config.saved_dataset
    # mt-datasets
    for index, dataset in enumerate(datasets):
        scr_data_path = os.path.join(config.data_src_path, dataset)
        data_path = os.path.join(config.data_path, to_dir, dataset)
        os.makedirs(data_path, exist_ok=True)
        reader = dataset2processer.get(dataset, amazon_reader)

        train_data = reader(scr_data_path, word_counter, char_counter, mode="train")
        valid_data = reader(scr_data_path, word_counter, char_counter, mode="dev")
        test_data = reader(scr_data_path, word_counter, char_counter, mode="test")

        if not os.path.exists(data_path):
            os.mkdir(data_path)

        save(data_path+"/train.json", train_data, message="train data")
        save(data_path+"/test.json", test_data, message="test data")
        if valid_data is not None:
            save(data_path+"/dev.json", valid_data, message="dev data")
        if index == len(datasets) - 1:
            word_vocab = Vocabulary(word_counter, max_vocab_size=30000)
            embedding_path = os.path.join(config.data_path, to_dir, 'embedding')
            os.makedirs(embedding_path, exist_ok=True)
            word_vocab.save(embedding_path + '/' +config.word_vocab_file)
            char_vocab = Vocabulary(char_counter)
            char_vocab.save(embedding_path + '/' +config.char_vocab_file)

            # glove
            w2v_model = Word2VecModel(model_folder='glove', w2v_name="glove.840B.300d.txt", vocab_name="vocab.txt")
            if w2v_model._model_folder == "glove":
                w2v_model.load_glove_vector()
                w2v_model.change_vocab(word_vocab.vocabulary)
            else:
                w2v_model.load_model()
            w2v_model.save_model(model_folder=embedding_path, embeddings_name="embeddings.txt")
    for dataset in datasets:
        scr_data_path = config.data_src_path + dataset
        data_path = os.path.join(config.data_path, to_dir, dataset)
        train_data = load(data_path+"/train.json")
        test_data = load(data_path + "/test.json")
        if valid_data is not None:
            valid_data = load(data_path+"/dev.json")
        build_features(config, train_data, "Train", out_file=data_path+"/train.npz", word_vocab=word_vocab, char_vocab=char_vocab, is_test=False)
        # build_features(config, test_data, "Test", out_file=data_path+"/test.npz", word_vocab=word_vocab, char_vocab=char_vocab, is_test=False)
        if valid_data is not None:
            build_features(config, valid_data, "Valid", out_file=data_path+"/dev.npz", word_vocab=word_vocab, char_vocab=char_vocab, is_test=False)
    