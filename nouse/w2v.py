# encoding=utf-8
# This Program is written by Victor Zhang at 2016-08-01 23:04:21
# Modified at 2019-11-06 16:30:21
# version 1.7
#

import logging
import os
from pathlib import Path
import numpy as np
from gensim.models.word2vec import Word2Vec, LineSentence
from gensim.models.keyedvectors import KeyedVectors
from collections import Counter
import multiprocessing
from vocabulary import Vocabulary

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Word2VecModel():
    def __init__(self,
                 size=300,
                 window=5,
                 min_count=5,
                 workers=-1,
                 isSkipGram=False,
                 model_folder="w2v_model",
                 vocab_name="vocab.txt",
                 embeddings_name="embeddings.txt",
                 w2v_name="word2vec.model",
                 load_model=False,
                 max_vocab_size=-1,
                 pad_token='<pad>',
                 unk_token='<unk>',
                 bos_token='<bos>',
                 eos_token='<eos>',
                 sep_token='<sep>',
                 mask_token='<msk>',
                 other_tokens=None,
                 **kwargs):
        """initialize the model

        Keyword Arguments:
            size {int} -- [the dimension of embeddings, usually 50,100,200,300] (default: {200})
            window {int} -- [the window size when training the embeddings] (default: {5})
            min_count {int} -- [the minimum count that the word is considered when training the model] (default: {5})
            workers {number} -- [number of workers to train the embeddings] (default: {5})
            isSkipGram {bool} -- [use skip-gram of CBOW] (default: {False})           
            model_folder {str} -- [the folder that store the trained model] (default: {"./models/"})
            vocab_name
            embeddings_name
            w2v_name
            load_model {bool} -- [whether load the model when initializing] (default: {False})
            max_vocab_size {int} -- [the max size of vocabulary] (default: {-1})
            special_token {bool} -- [whether to save or load special token from the vocabulary] (default: {True})
            pad_token
            unk_token
            bos_token
            eos_token
            sep_token
            mask_token
            other_tokens
            kwargs

        """

        # For masking purpose, 0 for padding
        self._model = None
        self._embeddings = None
        self._size = size  # size of embedding
        self._vocab_size = 0
        self._max_vocab_size = max_vocab_size
        self._window = window
        self._min_count = min_count
        self._model_folder = model_folder
        self._vocab_name = vocab_name
        self._embeddings_name = embeddings_name
        self._w2v_name = w2v_name
        self._workers = workers
        if self._workers == -1:
            self._workers = multiprocessing.cpu_count()
        self._sg = 1 if isSkipGram else 0

        self.kwargs = kwargs

        self._vocab = Vocabulary(pad_token=pad_token,
                                 unk_token=unk_token,
                                 bos_token=bos_token,
                                 eos_token=eos_token,
                                 sep_token=sep_token,
                                 mask_token=mask_token,
                                 other_tokens=other_tokens)

        if load_model:
            self.load_model()

    @property
    def model(self):
        return self._model

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def vocabulary(self):
        return self._vocab

    @property
    def vocab_file(self):
        return Path(self._model_folder) / self._vocab_name

    @property
    def embeddings_file(self):
        return Path(self._model_folder) / self._embeddings_name

    @property
    def w2v_file(self):
        return Path(self._model_folder) / self._w2v_name

    @property
    def model_folder(self):
        return Path(self._model_folder)

    def _read_file(self, filename, tokenizer=None):
        """
        read sentences from a file
        the file should list one sentence each line, and separate words with blank space.

        Parameters:
        ------------
            filename: string
                name of file where saved the sentences waiting for processing
            tokenizer: string
                name of function processing sentences

        Returns:
        ---------
            sentences: a list of string list
                a list of words list while each words list coming from a split sentence
        """
        if tokenizer is None:
            raise Exception("Please identify the tokenizer")
        ifile = open(filename, 'r', encoding='utf-8')
        sentences = []
        for line in ifile:
            words = tokenizer(line.strip())
            sentences.append(words)
        return sentences

    def train_model_from_file(self, filename, is_cut=False, tokenizer=None):
        """
        train model from a text file
        the file should have one sentence each line, and separate words with blank space.

        Parameters:
        ------------
            filename: string
                name of file where saved the sentences waiting for processing  
            is_cut: bool
                whether to use the word segmentation tool
            tokenizer: string
                name of function processing sentences
        """
        sentences = []
        if is_cut:
            sentences = self._read_file(filename)
        else:
            sentences = LineSentence(filename)
        self.train_model(sentences)

    def get_word2vec_model(self, st):
        """
        get the word2vec model
        return the trained model, if the model is not exist, load the model or train the model.

        Parameters:
        -----------
            st: a list of string list
                a list of words list while each words list coming from a split sentence

        Returns:
        ---------
            self._model: word2vec model
                a trained word2vec model
        """
        if self._model is None:
            if self.vocab_file.exists() and self.embeddings_file.exists():
                self._model = self.load_model()
            elif self.w2v_file.exists():
                self._model = self.load_model(load_embeddings=False, load_w2v_model=True)
            else:
                self._model = self.train_model(st)
        return self._model

    def train_model(self, sentences, add_special_token=True):
        """
        train the word2vec model
        a satisfactory 'sentences' is a list of split sentences, 
        for example
            sentences = [["i","like","apples"],
                        ["tom","likes","bananas"],
                        ["jane","hate","durian"]] 
        is a suitable list, it is highly recommended that the word should be in lower case.

        Parameters:
        ------------
            sentences: a list of string list
                a list of words list while each words list coming from a split sentence
            use_special: bool
                whether to add special tokens and their embeddings, default: true

        Returns:
        ---------
            self._model: word2vec model
        """
        logging.info('Trainning Word2Vec model')
        self._model = Word2Vec(sentences, size=self._size, window=self._window,
                               min_count=self._min_count, workers=self._workers,
                               sg=self._sg, **self.kwargs)

        self._vocab.build(counter=self._build_vocab_counter_from_wv(), add_special_token=add_special_token)
        #  create embedding
        self.set_embeddings(self._model.wv.vectors, add_special_token=add_special_token)
        self.save_model()

        return self._model

    def _build_vocab_counter_from_wv(self):
        """
        create the counter for building vocabulary from word2vec

        Returns:
        ---------
            counter: Counter
            the counter of trained word2vec model

        """
        counter = Counter()
        vocab = self._model.wv.vocab
        for word in vocab:
            counter[word] = vocab[word].count
        return counter

    def load_embeddings(self, embeddings_name=None):
        """
        only load the embeddings
        each line is the embedding of each word according to the dictionary.
        """
        if embeddings_name:
            self._embeddings_name = embeddings_name
        self._embeddings = np.loadtxt(self.embeddings_file)
        self._vocab_size, self._size = self._embeddings.shape
        print("embedding_size: ", self._vocab_size, "  X  ", self._size)
        return self._embeddings

    def set_embeddings(self, embeddings, add_special_token=True):
        """
        set embedding element for the class 
        by combining the common word embeddings with the special token embeddings.

        Parameters:
        ------------
            embeddings: a list of digits
                a list of common words vectors
            use_special: bool
                whether to add the embedding of special tokens, if true, then add them. default true.
        """
        self._embeddings = embeddings
        self._vocab_size, self._size = self._embeddings.shape

        speStrs_size = len(self._vocab.special_vocab)
        if add_special_token and speStrs_size != 0:
            zeros = np.zeros((speStrs_size, self._size))
            self._embeddings = np.r_[zeros, self._embeddings]
            self._vocab_size += speStrs_size

    def save_embeddings(self, embeddings_name=None):
        """
        save embedding to filename 
        """
        if embeddings_name:
            self._embeddings_name = embeddings_name
        np.savetxt(self.embeddings_file, self._embeddings)
        self._vocab.save(self.vocab_file)

    def sent2seq(self, sentence, max_length=-1, bos=False, eos=False):
        """
        convert a sentence to a list of index

        Parameters:
        -------------
            sentence: a list of string
                a list of words waiting for converting to target index
            max_length: int
                the max length of the return list
            bos: bool
                whether add the bos token, if True, then add it
            eos: bool
                whether add the eos token, if True, then add it 

        Returns:
        -------------
            self._vocab.sent2seq(): a list of index
        """
        return self._vocab.sent2seq(sentence, max_length, bos, eos)

    def seq2sent(self, seq, neglect_zero=True):
        """
        from sequence of index to list of words
        e.g. seq = [25,35,45]  --> ["I","like","apple"]

        Parameters:
        -------------
            seq: a list of int
                a list of index 
            neglect_zero: bool
                whether to convert 0 to Nil, if true, then do not convert 0, default True

        Returns:
        ----------
            []: list of string
                list of words
        """
        if neglect_zero:
            return [self._vocab.index2word(index) for index in seq if index != 0]
        else:
            return [self._vocab.index2word(index) for index in seq]

    def get_vector(self, word, case_sensitive=False):
        """
        get the word embedding of a word

        Parameters:
        -----------
            word: string
                the source word should be converted to vector
            case_sensitive: bool
                whether be sensitive to letter case, 
                if true, different case of letter is matched with different vector. 
                default false

        Returns:
        ----------
            []: vector
                the embedding vector of the word
        """
        if not case_sensitive:
            word = word.lower()
        if word in self._vocab.vocabulary:
            return self._embeddings[self._vocab.word2index(word)]
        else:
            # 修改了一下，否则的话得到的不是零向量是[[0.,0.,……0.]]
            return np.zeros((1, self._size))[0]

    def get_avg_vector(self, sent):
        """
        get the average embedding of a sentence

        Parameters:
        ------------
            sent: string
            the sentence is separated by blank space

        Returns:
        ----------
            []: vecotr
            the average embedding of the sentence
        """
        isum = np.zeros((self._size))
        cnt = 0
        for word in sent.split(' '):
            vec = self.get_vector(word)
            if vec is not None:
                isum += vec
                cnt += 1
        if cnt != 0:
            isum /= cnt
        return isum

    def save_model(self, binary=False, model_folder=None, save_w2v_model=False, embeddings_name=None):
        """
        save the model

        Parameters:
        -------------
            binary: bool
                whether to save model as binary file

        """
        logging.info('Saving Word2Vec model')
        if model_folder is not None:
            self._model_folder = model_folder
        if not self.model_folder.exists():
            self.model_folder.mkdir()
        if save_w2v_model:
            self._model.wv.save_word2vec_format(self.w2v_file, binary=binary)
        self.save_embeddings(embeddings_name)

    def load_model(self, load_embeddings=True, load_w2v_model=False, binary=False, add_special_token=True, embeddings_name=None):
        """
        load the model

        Parameters:
        -------------
            load_emneddings: bool
                whether to load embeddings from file, if true, then load embeddings and vocabulary from file, 
                default: true
            load_w2v_model: bool
                whether to load w2v model from file, if true, then load model from file,
                else set embeddings according to current variables. 
                default: false
            binary: bool
                whether to load w2v model as binary, if true, load binary. 
                default: false
            add_special_token: bool
                whether to use special tokens when setting embeddings, if true, use special tokens. 
                default: true

        Returns:
        ----------
            self._model: word2vec model
        """
        logging.info('Loading Word2Vec model')
        self._model = None
        if load_w2v_model:
            self._model = KeyedVectors.load_word2vec_format(self.w2v_file, binary=binary, encoding="utf-8")
            if not load_embeddings:
                self.set_embeddings(self._model.wv.vectors, add_special_token=add_special_token)
        if load_embeddings:
            self._vocab.load(self.vocab_file)
            self.load_embeddings(embeddings_name)
        return self._model

    def load_glove_vector(self, add_special_token=True):
        """
        load glove file, and create the vocab.txt and embedding.txt for glove

        Parameters:
        -----------
            add_special_token: bool
                whether to add special tokens while setting embeddings. if true, then add special tokens.
                default: true
        """
        counter = []
        embeddings = []
        with open(self.w2v_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.rstrip().split(' ')
                counter.append(line[0])
                embeddings.append(list(map(float, line[1:])))

        embeddings = np.array(embeddings)
        self.set_embeddings(embeddings, add_special_token=add_special_token)
        self._vocab.build(counter, add_special_token=add_special_token)

    def change_vocab(self, counter, add_special_token=False):
        """
        given a counter, eliminate the embedding of words does not in counter list.
        Then build new vocabulary.

        Parameters:
        -----------------
            counter: Counter
                a Counter parameter to filter useful embeddings in old vocabulary.
            add_special_token: bool
                whether to add special tokens in new vocabulary, if true, then add in. default false
        """
        counter_list = list(counter)
        index_list = [self._vocab.word2index(word) for word in counter_list]
        embeddings = self._embeddings[index_list]
        self.set_embeddings(embeddings, add_special_token=add_special_token)
        self._vocab.build(counter, add_special_token=add_special_token)
