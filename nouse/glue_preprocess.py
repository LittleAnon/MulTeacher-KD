'''Preprocessing functions and pipeline'''
import os
import logging as log
from collections import defaultdict,Counter
import ipdb as pdb # pylint disable=unused-import
import _pickle as pkl
import numpy as np
import torch
from vocabulary import Vocabulary
from w2v import Word2VecModel
from tasks import CoLATask, MRPCTask, MultiNLITask, QQPTask, RTETask, \
                  QNLITask, QNLIv2Task, SNLITask, SSTTask, STSBTask, WNLITask
import time
from config import SearchConfig

config = SearchConfig()

GLUE_PATH_PREFIX = config.data_src_path + 'glue_dataset/'
EXPORT_PATH = config.data_path + config.saved_dataset + '/'
EMBEDDING_PATH = EXPORT_PATH + 'embedding/'

ALL_TASKS = ['mnli', 'mrpc', 'qqp', 'rte', 'qnli', 'snli', 'sst', 'sts-b', 'wnli', 'cola']
NAME2INFO = {'sst': (SSTTask, 'SST-2/'),
            'cola': (CoLATask, 'CoLA/'),
            'mrpc': (MRPCTask, 'MRPC/'),
            'qqp': (QQPTask, 'QQP'),
            'sts-b': (STSBTask, 'STS-b/'),
            'mnli': (MultiNLITask, 'MNLI/'),
            'qnli': (QNLITask, 'QNLI/'),
            'qnliv2': (QNLIv2Task, 'QNLIv2/'),
            'rte': (RTETask, 'RTE/'),
            'snli': (SNLITask, 'SNLI/'),
            'wnli': (WNLITask, 'WNLI/')
            }
for k, v in NAME2INFO.items():
    NAME2INFO[k] = (v[0], GLUE_PATH_PREFIX + v[1])


def build_tasks(config):
    '''Prepare tasks'''
    def parse_tasks(task_list):
        '''parse string of tasks'''
        if task_list == ['all']:
            tasks = ALL_TASKS
        else:
            tasks = task_list
        return tasks
    train_task_names = parse_tasks(config.datasets)
    all_task_names = list(set(train_task_names))
    tasks = get_tasks(train_task_names, config.limit, config.load_tasks)
    os.makedirs(EMBEDDING_PATH, exist_ok=True)
    word2freq, char2freq = get_words(tasks)
    word_vocab = Vocabulary(Counter(word2freq), max_vocab_size=config.max_word_v_size)
    word_vocab.save(EMBEDDING_PATH + config.word_vocab_file)
    char_vocab = Vocabulary(Counter(char2freq), max_vocab_size=config.max_char_v_size)
    char_vocab.save(EMBEDDING_PATH + config.char_vocab_file)

    w2v_model = Word2VecModel(model_folder='glove', w2v_name="glove.840B.300d.txt", vocab_name='vocab.txt')
    if w2v_model._model_folder == "glove":
        w2v_model.load_glove_vector()
        w2v_model.change_vocab(word_vocab.vocabulary)
    else:
        w2v_model.load_model()
    w2v_model.save_model(model_folder=EMBEDDING_PATH, embeddings_name=config.word_emb_file)

    for task in tasks:
        process_task(task, word_vocab, char_vocab, config.limit, config.char_limit, EXPORT_PATH + task.name)
        log.info("\tFinished indexing tasks{}".format(task.name))

def get_tasks(task_names, max_seq_len, load):
    '''
    Load tasks
    '''
    tasks = []
    print(task_names)
    for name in task_names:
        assert name in NAME2INFO, 'Task not found!'
        pkl_path = NAME2INFO[name][1] + "%s_task.pkl" % name
        if os.path.isfile(pkl_path) and load:
            task = pkl.load(open(pkl_path, 'rb'))
            log.info('\tLoaded existing task %s', name)
        else:
            task = NAME2INFO[name][0](NAME2INFO[name][1], max_seq_len, name)
            pkl.dump(task, open(pkl_path, 'wb'))
        tasks.append(task)
    log.info("\tFinished loading tasks: %s.", ' '.join([task.name for task in tasks]))
    return tasks

def get_words(tasks):
    '''
    Get all words for all tasks for all splits for all sentences
    Return dictionary mapping words to frequencies.
    '''
    word2freq = defaultdict(int)
    char2freq = defaultdict(int)

    def count_sentence(sentence):
        '''Update counts for words in the sentence'''
        for word in sentence:
            word2freq[word] += 1
            for c in word:
                char2freq[c] += 1
        return

    for task in tasks:
        splits = [task.train_data_text, task.val_data_text, task.test_data_text]
        for split in [split for split in splits if split is not None]:
            for sentence in split[0]:
                count_sentence(sentence)
            if task.pair_input:
                for sentence in split[1]:
                    count_sentence(sentence)
    log.info("\tFinished counting words")
    return word2freq, char2freq


def process_task(task, vocab, char_vocab, word_limit, char_limit, export_dir):
    '''
    Convert a task's splits into AllenNLP fields then
    Index the splits using the given vocab (experiment dependent)
    '''
    os.makedirs(export_dir, exist_ok=True)
    if hasattr(task, 'train_data_text') and task.train_data_text is not None:
        train = process_split(task.train_data_text, vocab, char_vocab, task.pair_input, task.categorical, word_limit, char_limit)
        data_idxs, data_char_idxs, labels, ids = train
        np.savez(export_dir + '/train.npz', data_idxs=np.array(data_idxs), data_char_idxs=np.array(data_char_idxs),
             labels=np.array(labels), ids=np.array(ids))
    else:
        train = None
    if hasattr(task, 'val_data_text') and task.val_data_text is not None:
        val = process_split(task.val_data_text, vocab, char_vocab, task.pair_input, task.categorical, word_limit, char_limit)
        data_idxs, data_char_idxs, labels, ids = val
        np.savez(export_dir + '/val.npz', data_idxs=np.array(data_idxs), data_char_idxs=np.array(data_char_idxs),
             labels=np.array(labels), ids=np.array(ids))
    else:
        val = None
    if hasattr(task, 'test_data_text') and task.test_data_text is not None:
        test = process_split(task.test_data_text, vocab, char_vocab, task.pair_input, task.categorical, word_limit, char_limit)
        data_idxs, data_char_idxs, labels, ids = test
        np.savez(export_dir + '/test.npz', data_idxs=np.array(data_idxs), data_char_idxs=np.array(data_char_idxs),
             labels=np.array(labels), ids=np.array(ids))
    else:
        test = None
    # for instance in train + val + test:
    #     instance.index_fields(vocab)
    # return train, val, test

def convert_to_features(limit, char_limit, data, word_vocab, char_vocab, use_char=True):

    sent_limit = limit
    char_limit = char_limit

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

def process_split(split, vocab, char_vocab, pair_input, categorical, max_length=100, max_char_length=12):
    # data_idxs=np.array(data_idxs), data_char_idxs=np.array(data_char_idxs), labels=np.array(labels), ids=np.array(ids)
    data_idxs = []
    data_char_idxs = []
    labels = []
    ids = []
    if pair_input:
        for sent0, sent1 in zip(split[0], split[1]):
            data_idx_0, data_char_idx_0 = convert_to_features(max_length, max_char_length, sent0, vocab, char_vocab)
            data_idx_1, data_char_idx_1 = convert_to_features(max_length, max_char_length, sent1, vocab, char_vocab)
            data_idxs.append([data_idx_0, data_idx_1])
            data_char_idxs.append([data_char_idx_0, data_char_idx_1])
        labels = split[2]
        
        if len(split) == 4: # numbered test examples
            idxs = split
        else:
            idxs = list(range(len(labels)))

    else:
        for sent in zip(split[0]):
            data_idx, data_char_idx = convert_to_features(max_length, max_char_length, sent[0], vocab, char_vocab)
            data_idxs.append([data_idx])
            data_char_idxs.append([data_char_idx])
        labels = split[2]
        if len(split) == 4: # numbered test examples
            idxs = split
        else:
            idxs = list(range(len(labels)))
    print("data_idxs", np.array(data_idxs).shape)
    print("data_char_idxs", np.array(data_char_idxs).shape)
    return data_idxs, data_char_idxs, labels, ids

build_tasks(config)
