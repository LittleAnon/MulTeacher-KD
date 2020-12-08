# encoding=utf-8
# This Program is written by Victor Zhang at 2016-08-01 23:04:21
# Modified at 2019-11-08 13:57:21
# version 1.8
#

import numpy as np
from pathlib import Path

class Vocabulary:
    def __init__(self,
                 counter_or_filename=None,
                 pad_token='<pad>',
                 unk_token='<unk>',
                 bos_token='<bos>',
                 eos_token='<eos>',
                 sep_token='<sep>',
                 mask_token='<msk>',
                 other_tokens=None,
                 **kwargs):
        """
        initialize class and build initial vocabulary if counter is not None.

        Parameters:
        ----------
            counter: map 
                each element of which contains a word and the count of the word
                for example:
                    sentence = 'this is a big big world.'
                    data_counter = {'big':2, 'this':1, 'is':1, 'a':1, 'world':1} 
            unk_token: str
                a special token representing the unknow word
            pad_token: str
                a special token representing the padding token 
            bos_token: str
                a special token representing the begin token of a sentence
            eos_token: str
                a special token representing the end token of a sentence 
            sep_token: str
                a special token representing the seperate token 
            other_tokens: a list of strings
                a list of special tokens defined by users

        """
        # initialize parameters
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.other_tokens = other_tokens
        if self.other_tokens is None:
            self.other_tokens = []

        self._vocab = []
        self._word2index_dict = {}
        self._index2word_dict = {}
        self._frequence_map = {}

        # build initial vocabulary
        special_vocab = [self.pad_token, self.unk_token,
                         self.bos_token, self.eos_token, self.sep_token, self.mask_token] + self.other_tokens
        self._special_vocab = list(filter(None, special_vocab))
        
        if counter_or_filename:
            if isinstance(counter_or_filename, str) or isinstance(counter_or_filename, Path):
                self.load(counter_or_filename, **kwargs)
            else:
                self.build(counter_or_filename, **kwargs)

    def save(self, filename, sep=" ", frequence=False, save_special_token=True):
        """
        save vocabulary to filename

        Parameters:
        -----------
        filename: str
            the name of file to save the vacabulary. 
        sep: string
            a string to seperate two strings.
        frequence: bool
            if true, save the count of each word with word and the number, else just save the word and the number of the word for each element.
        special_token: bool
            if true, save the special tokens before saving common words, else just save the common words.

        """
        with open(filename, 'w', encoding='utf-8') as file:
            for word in self._vocab:
                if save_special_token or (word not in self._special_vocab):
                    if frequence:
                        file.write(word + sep + str(self._frequence_map[word]) + '\n')
                    else:
                        file.write(word + '\n')

    def load(self, filename, sep=' ', frequence=False, add_special_token=True):
        """
        load vocabulary from filename

        Parameters:
        -----------
        filename: str
            the name of file to load the vocabulary.
        sep: string
            a string to seperate two strings
        frequence: bool
            if true, load the frequence of each word, else do not load frequence
        special_token: bool
            if true, load the special tokens, else not

        """
        self._vocab = []
        with open(filename, 'r', encoding='utf-8') as file:
            for k, line in enumerate(file):
                line = line.strip().split(sep)
                if add_special_token or (line[0] not in self._special_vocab):
                    self._vocab.append(line[0])
                    self._word2index_dict[line[0]] = k
                    self._index2word_dict[k] = line[0]
                    if frequence:
                        self._frequence_map[line[0]] = int(line[1])
                    else:
                        self._frequence_map[line[0]] = 0

    @property
    def vocabulary(self):
        """
        just return the vocabulary
        """
        return self._vocab

    @property
    def special_vocab(self):
        """
        just return the vocabulary
        """
        return self._special_vocab

    @property
    def frequence_map(self):
        """
        just return the frequnce
        """
        return self._frequence_map

    def __len__(self):
        return len(self._vocab)

    def build(self, counter=None, min_count=0, max_vocab_size=-1, add_special_token=True):
        """
        Build a vocabulary file (if it does not exist yet) from data file.

        Parameters:
        -----------
            counter: list of strings
                a list of word
            max_vocab_size: int
                limit on the size of the created vocabulary.

        """

        if isinstance(counter, list):
            self._vocab = counter
        else:
            self._vocab = [k for k, v in counter.most_common() if k in self._special_vocab or v >= min_count]

        if add_special_token:
            self._vocab = self._special_vocab + self._vocab

        if max_vocab_size != -1:
            self._vocab = self._vocab[:max_vocab_size]

        self._word2index_dict = {}
        self._index2word_dict = {}
        self._frequence_map = {}

        cnt = 0
        for word in self._vocab:
            self._word2index_dict[word] = cnt
            self._index2word_dict[cnt] = word
            cnt += 1
            if isinstance(counter, list) or word in self._special_vocab:
                self._frequence_map[word] = 0
            else:
                self._frequence_map[word] = counter[word]

    def word2index(self, word):
        """
        return the integer id of a word string

        Parameters:
        -----------
            word: str
                a word of an id in vocabulary.
        """

        if word in self._word2index_dict:
            return self._word2index_dict[word]
        else:
            return self._word2index_dict[self.unk_token]

    def _sent_add_special(self, sentence, max_length=-1, bos=False, eos=False, pad_last=True, trunc_last=True):
        new_setence = []
        if max_length == -1:
            length = len(sentence) + int(bos) + int(eos)
        else:
            length = max_length

        actual_length = length - int(bos) - int(eos)
        if trunc_last:
            trunc_sent = sentence[:actual_length]
        else:
            trunc_sent = sentence[-actual_length:]
        actual_length = len(trunc_sent)
        new_setence = [self.pad_token] * length
        if bos:
            new_setence[0] = self.bos_token
        start = int(bos)
        if pad_last:
            new_setence[start:start+actual_length] = trunc_sent
        else:
            new_setence[-actual_length-int(eos):] = trunc_sent
        if eos:
            new_setence[-1] = self.eos_token
        return new_setence



    def sent2seq(self, sentence, max_length=-1, bos=False, eos=False, pad_last=True, trunc_last=True, return_list=False):
        """
        convert a sentence to a list of index

        Parameters:
        ------------
            sentence: a list of string
                a list of words waiting for converting to target index
            max_length: int
                the max length of the return list
            bos: bool
                whether add the bos token, if True, then add it
            eos: bool
                whether add the eos token, if True, then add it 

        Returns:
        ------------
            seq: a list of index
        """
        new_setence = self._sent_add_special(sentence, max_length, bos, eos, pad_last, trunc_last)
        seq = [self.word2index(word) for word in new_setence]
        
        if return_list:
            return seq
        return np.array(seq)

    def sent2matrix(self, sentence, max_length=-1, char_vocab=None, max_char_length=10, bos=False, eos=False, pad_last=True, trunc_last=True):
        """
        convert a sentence to a matrix of index (char)

        Parameters:
        ------------
            sentence: a list of string
                a list of words waiting for converting to target index
            max_length: int
                the max length of the return list
            bos: bool
                whether add the bos token, if True, then add it
            eos: bool
                whether add the eos token, if True, then add it 

        Returns:
        ------------
            seq: a list of index
        """
        new_setence = self._sent_add_special(sentence, max_length, bos, eos, pad_last, trunc_last)
        
        seq = np.array([self.word2index(word) for word in new_setence])
        matrix = None
        
        if char_vocab is not None:
            length = len(new_setence)
            matrix = np.zeros([length, max_char_length], dtype=np.int32)

            for i, word in enumerate(new_setence):
                if word in self.special_vocab:
                    continue
                char_seq = char_vocab.sent2seq(list(word), max_length=max_char_length)
                matrix[i] = char_seq

        return seq, matrix

    def index2word(self, index):
        """
        return the word string of an integer id

        Parameters:
        ------------
            id: int
                an id of a word in vocabulary.
        """
        return self._index2word_dict[index]

    def __getitem__(self, key):
        """
        if the key is a string, then return the id of the key,
        else if the key is an integer, then return the word of the key.

        Parameters:
        -----------
            key: str or int
                It could be either a word or an id in the vocabulary.
        """
        if isinstance(key, str):
            return self.word2index(key)
        elif isinstance(key, int):
            return self.index2word(key)
        elif isinstance(key, slice):
            return self._vocab[key]
        else:
            raise Exception('Wrong type of key word %s: %s' % (key, type(key)))

    def __iadd__(self, other):
        """
        add other vocabulary to this vocabulary.
        This method is build for add two vocabularies.

        Parameters:
        -----------
            other:  Vocabulary or list
                It could be either a word or an id in the vocabulary.
        """
        special_vocab = []
        if isinstance(other, Vocabulary):
            vocab = other.vocabulary
            special_vocab = other.special_vocab
            pass
        else:
            vocab = list(other)
        cnt = len(self._word2index_dict)
        for word in vocab:
            if word not in special_vocab and word not in self._word2index_dict:
                self._vocab.append(word)
                self._word2index_dict[word] = cnt
                self._index2word_dict[cnt] = word
                cnt += 1
                if isinstance(other, Vocabulary):
                    self._frequence_map[word] = other.frequence_map[word]
                else:
                    self._frequence_map[word] = 0
        return self
    
    def difference(self, other):
        """
        add other vocabulary to this vocabulary.
        This method is build for add two vocabularies.

        Parameters:
        -----------
            other:  Vocabulary or list
                It could be either a word or an id in the vocabulary.
        """
        difference_list = []
        special_vocab = []
        if isinstance(other, Vocabulary):
            vocab = other.vocabulary
            special_vocab = other.special_vocab
            pass
        else:
            vocab = list(other)
        for word in vocab:
            if word not in special_vocab and word not in self._word2index_dict:
                difference_list.append(word)
        return difference_list
