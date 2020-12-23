""" Utilities """
import logging
import os
import random
import shutil

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import RobertaConfig, AutoTokenizer, \
    GPT2Config

from bert_fineturn.data_processor.glue import glue_compute_metrics as compute_metrics
from dataset import OneInputDataset, get_tensor_data
from vocabulary import Vocabulary

DATASET_TYPE = {
    'mrpc': 2,
    'mnli': 2,
    'qnli': 2,
    'qqp': 2,
    'rte': 2,
    'snli': 2,
    'sts-b': 2,
    'wnli': 2
}
LOSS_TYPE = {'sts-b': 2}
NUM_LABLE = {'sts-b': 1, 'SST-2':2}


def random_search(n_nodes, n_opts, remove_none=True):
    connections = []
    options = []
    if remove_none:
        n_opts = n_opts - 1
    for i in range(n_nodes):
        t = random.randint(0, i)
        connections.append(t)
        c = random.randint(0, n_opts - 1)
        options.append(c)
    return (connections, options)


def choice2alpha(choice, n_nodes, n_ops):
    connections, options = choice
    assert len(connections) == n_nodes and len(options) == n_nodes
    alphas = []
    for i in range(n_nodes):
        alpha = np.zeros((i + 1, n_ops))
        alpha[connections[i]][options[i]] = 1.0
        alphas.append(alpha)
    # print(alphas)
    alphas = [torch.tensor(np.array(x)) for x in alphas]
    return alphas


def get_data(path, datasets):
    nums = DATASET_TYPE.get(datasets, 1)
    train_dataset = OneInputDataset(path + '/' + datasets + "/train.npz", 0, nums,)
    valid_dataset = OneInputDataset(path + '/' + datasets + "/dev.npz", 0, nums,)
    test_dataset = None
    if os.path.exists(path + '/' + datasets + "/test.npz"):
        test_dataset = OneInputDataset(path + '/' + datasets + "/test.npz", 0, nums, )
    return train_dataset, valid_dataset, test_dataset


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size())
        for k, v in model.named_parameters()
        if v.requires_grad and not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,), isTrain=False, output_modes="classification"):
    """ Computes the precision@k for the specified values of k """
    if output_modes == "classification":
        maxk = max(topk)
        batch_size = target.size(0)
        _, out_classes = output.max(dim=1)
        correct = (out_classes == target).sum()
        correct = correct.float() / batch_size
        return correct
    else:
        correct1 = pearsonr(
            output.reshape(-1).detach().cpu().numpy(),
            target.detach().cpu().numpy())[0]
        correct2 = spearmanr(
            output.reshape(-1).detach().cpu().numpy(),
            target.detach().cpu().numpy())[0]
        return (correct1 + correct2) / 2
    return res


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id



def convert_examples_to_features_v2(examples, label_list, max_seq_length, tokenizer, output_mode, is_master=True, gpt2=False,tok_type = None):
    label_map = {label: i for i, label in enumerate(label_list)}

    if tok_type == 'bert':
        cls_ = "[CLS]"
        sep_ = "[SEP]"
    elif tok_type == 'gpt2':
        cls_ = tokenizer.bos_token
        sep_ = tokenizer.eos_token
    elif tok_type == 'roberta':
        cls_ = tokenizer.cls_token
        sep_ = tokenizer.sep_token
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and is_master:
            print("Writing example %d of %d" % (ex_index, len(examples)))
        args = (
            (example.text_a,) if example.text_b is None else (examples.text_a + examples.text_b,)
        )
        result = tokenizer(*args, padding='max_length', max_length=max_seq_length, truncation=True)

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index == 0 and is_master:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: {}".format(example.label))
            print("label_id: {}".format(label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                seq_length=seq_length))
    return features


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizers, output_mode, is_master=True, tok_type = None):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = [[],[]]
    if tok_type is 'rAg':
        tok_type_list = ['roberta', 'gpt2']
    else:
        tok_type_list = [tok_type]
        tokenizers = [tokenizers]
    for i in range(len(tok_type_list)):
        tokenizer = tokenizers[i]
        tok_type = tok_type_list[i]

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0 and is_master:
                print("Writing example %d of %d" % (ex_index, len(examples)))

            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                if tok_type == 'gpt2':
                    if len(tokens_a) > max_seq_length - 1:
                        tokens_a = tokens_a[:(max_seq_length - 1)]
                else:
                    if len(tokens_a) > max_seq_length - 2:
                        tokens_a = tokens_a[:(max_seq_length - 2)]
            ## 和finetune的数据格式保持一致，bert和gpt2和roberta的特殊字符添加格式还不太一样
            if tok_type == 'bert':
                cls_ = tokenizer.cls_token
                sep_ = tokenizer.sep_token
                pad_ = tokenizer.pad_token_id
                tokens = [cls_] + tokens_a + [sep_]
            elif tok_type == 'gpt2':
                cls_ = tokenizer.bos_token
                sep_ = tokenizer.eos_token
                pad_ = tokenizer.pad_token_id
                tokens = tokens_a
            elif tok_type == 'roberta':
                cls_ = tokenizer.cls_token
                sep_ = tokenizer.sep_token
                pad_ = tokenizer.pad_token_id
                tokens = [cls_] + tokens_a + [sep_]
                if tokens_b:
                    tokens = tokens + [sep_]

            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            seq_length = len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            id_padding = [pad_] * (max_seq_length - len(input_ids))
            input_ids += id_padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)

            if ex_index == 0 and is_master:
                print("*** Example for %s ***" % (tok_type))
                print("guid: %s" % (example.guid))
                print("tokens: %s" % " ".join([str(x) for x in tokens]))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                print("label: {}".format(example.label))
                print("label_id: {}".format(label_id))

            features[i].append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    seq_length=seq_length))
    if tok_type is 'rAg':
        return features
    elif len(features) > 0:
        return features[0]


def convert_examples_to_features_new(examples, label_list, max_seq_length, tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))
        if not example.text_b:
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
    
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            seq_length = len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

        else:
            tokens_a = tokenizer.tokenize(example.text_a)            
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_a_cp = tokens_a.copy()
            tokens_b_cp = tokens_a.copy()
            _truncate_seq_pair(tokens_a_cp, tokens_b_cp, max_seq_length - 3)

            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
            if len(tokens_b) > max_seq_length - 2:
                tokens_b = tokens_b[:(max_seq_length - 2)]

            tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
            tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
            tokens_all = ["[CLS]"] + tokens_a_cp + ["[SEP]"] + tokens_b_cp + ["[SEP]"]
            segment_ids_a = [0] * len(tokens_a)
            segment_ids_b = [0] * len(tokens_b)
            segment_ids = [0] * len(tokens_a) + [1] * (len(tokens_b_cp) + 1)

            input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
            input_ids_b = tokenizer.convert_tokens_to_ids(tokens_a)
            input_ids_all = tokenizer.convert_tokens_to_ids(tokens_all)

            input_mask = [1] * len(input_ids_a)
            input_mask = [1] * len(input_ids_b)
            input_mask = [1] * len(input_ids_all)

            seq_length_a = len(input_ids_a)
            seq_length_b = len(input_ids_b)
            seq_length_all = len(input_ids_all)

            padding_a = [0] * (max_seq_length - len(input_ids_a))
            input_ids_a += padding_a
            input_mask_a += padding_a
            segment_ids_a += padding_a

            padding_b = [0] * (max_seq_length - len(input_ids_b))
            input_ids_b += padding_b
            input_mask_b += padding_b
            segment_ids_b += padding_b

            padding_all = [0] * (max_seq_length - len(input_ids_all))
            input_ids_all += padding_all
            input_mask_all += padding_all
            segment_ids_all += padding_all

            input_ids = [input_ids_a, input_ids_b, input_ids_all]
            input_mask = [input_mask_a, input_mask_b, input_mask_all]
            segment_ids = [segment_ids_a, segment_ids_b, segment_ids_all]
            seq_length = [seq_length_a, seq_length_b, seq_length_all]
            tokens = [tokens_a, tokens_b, tokens_all]

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index == 0:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: {}".format(example.label))
            print("label_id: {}".format(label_id))

        features.append(
            InputFeatures(
                input_ids_a=input_ids_a,
                input_mask_a=input_mask_a,
                segment_ids_a=segment_ids_a,
                label_id=label_id,
                seq_length_a=seq_length_a,
                input_ids_b=input_ids_b,
                input_mask_b=input_mask_b,
                segment_ids_b=segment_ids_b,
                seq_length_b=seq_length_b,
                input_ids_all=input_ids_all,
                input_mask_all=input_mask_all,
                segment_ids_all=segment_ids_all,
                seq_length_all=seq_length_all))
        
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def load_gpt2_embedding_weight(model, path, train=False):
    pretrain_dict = torch.load(path + "/pytorch_model.bin")
    new_dict = {}

    new_dict['stem.word_embeddings.weight'] = pretrain_dict['transformer.wte.weight']#torch.Size([50257, 768])
    new_dict['stem.position_embeddings.weight'] = pretrain_dict['transformer.wpe.weight']#torch.Size([1024, 768])

    new_dict['stem.LayerNorm.weight'] = pretrain_dict['transformer.h.0.ln_1.weight']#torch.Size([768])
    new_dict['stem.LayerNorm.bias'] = pretrain_dict['transformer.h.0.ln_1.bias']#torch.Size([768])
    model.load_state_dict(new_dict, strict=False)


def load_roberta_embedding_weight(model, path, train=False):
    pretrain_dict = torch.load(path + "/pytorch_model.bin")
    new_dict = {}

    new_dict['stem.word_embeddings.weight'] = pretrain_dict['roberta.embeddings.word_embeddings.weight']
    new_dict['stem.position_embeddings.weight'] = pretrain_dict['roberta.embeddings.position_embeddings.weight']
    new_dict['stem.LayerNorm.weight'] = pretrain_dict['roberta.embeddings.LayerNorm.weight']
    new_dict['stem.LayerNorm.bias'] = pretrain_dict['roberta.embeddings.LayerNorm.bias']
    model.load_state_dict(new_dict, strict=False)


def load_bert_embedding_weight(model, path, train=False):
    pretrain_dict = torch.load(path + "/pytorch_model.bin")
    new_dict = {}
    for key in pretrain_dict.keys():
        if 'embeddings' in key:
            new_k = key
            if 'LayerNorm' in key:
                new_k = new_k.replace('gamma', 'weight')
                new_k = new_k.replace('beta', 'bias')
            if train:
                new_dict[new_k.replace('bert.embeddings', 'net.stem')] = pretrain_dict[key]
            else:
                new_dict[key.replace('bert.embeddings', 'stem')] = pretrain_dict[key]
    print("="*10 + " RESTORE KEYS" + "="*10)
    for k, v in model.named_parameters():
        if k in new_dict:
            print(k)

    model.load_state_dict(new_dict, strict=False)

def load_data(config, logger):
    from bert_fineturn.data_processor.glue import glue_processors as processors
    from bert_fineturn.data_processor.glue import glue_output_modes as output_modes
    from vocabulary import Vocabulary

    task_name = config.datasets
    processor = processors[task_name.lower()]()
    output_mode = output_modes[task_name.lower()]
    label_list = processor.get_labels()
    n_classes = len(label_list)

    data_path = os.path.join(config.data_path, config.saved_dataset)
    embedding_path = os.path.join(data_path, 'embedding')
    word_emb_file = os.path.join(embedding_path, config.word_emb_file)
    if config.is_master:
        logger.info("load word embeddings = {}".format(word_emb_file))
    with open(word_emb_file, "rb") as fh:
        word_mat = np.loadtxt(word_emb_file)
    char_vocab_file = os.path.join(embedding_path, config.char_vocab_file)
    char_emb_file = os.path.join(embedding_path, config.char_emb_file)
    if os.path.exists(char_emb_file):
        char_mat = np.loadtxt(char_emb_file)
    else:
        from sklearn.preprocessing import normalize
        char_vocab = Vocabulary()
        char_vocab.load(char_vocab_file)
        c_vocab_size = len(char_vocab)
        char_mat = np.random.rand(c_vocab_size, config.d_cvec)
        new_mat = []
        for ch in char_mat:
            new_mat.append(normalize([ch])[0])
        char_mat = np.array(new_mat)
    # get data with meta
    if config.is_master:
        logger.info("loading dataset {}".format(config.datasets))
    batch_method = config.batch_method
    train_data, valid_data, test_data = get_data(data_path, config.datasets)
    train_eval_sampler = valid_data
    if config.is_master:
        logger.info("number of class for each dataset %s " % n_classes)
    
    if not config.multi_gpu:
        train_sampler = RandomSampler(train_data)
        train_eval_sampler = RandomSampler(valid_data)
    else:
        train_sampler = DistributedSampler(train_data)
        train_eval_sampler = DistributedSampler(valid_data)
    eval_sampler = SequentialSampler(valid_data)

    eval_dataloader = DataLoader(valid_data, sampler=eval_sampler, batch_size=config.batch_size)
    train_eval_dataloader = DataLoader(valid_data, sampler=train_eval_sampler, batch_size=config.batch_size)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size)
    return train_dataloader, train_eval_dataloader, eval_dataloader, (word_mat, char_mat), output_mode, n_classes



def load_glue_dataset(config):
    from bert_fineturn.data_processor.glue import glue_processors as processors
    from bert_fineturn.data_processor.glue import glue_output_modes as output_modes
    from modeling import BertConfig

    task_name = config.datasets
    processor = processors[task_name.lower()]()
    output_mode = output_modes[task_name.lower()]
    label_list = processor.get_labels()
    n_classes = len(label_list)


    if config.teacher_type == 'rAg':
        tokenizer_r = AutoTokenizer.from_pretrained(
            config.tokenizer_name if config.tokenizer_name else config.teacher_model[0],

            use_fast=config.use_fast_tokenizer
        )
        tokenizer_g = AutoTokenizer.from_pretrained(
            config.tokenizer_name if config.tokenizer_name else config.teacher_model[1],

            use_fast=config.use_fast_tokenizer
        )
        tokenizer = [tokenizer_r ,tokenizer_g]
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name if config.tokenizer_name else config.teacher_model,

            use_fast=config.use_fast_tokenizer
        )

    train_examples = processor.get_train_examples('data/' + config.source + '/' + task_name + '/')
    train_features = convert_examples_to_features(train_examples, label_list,
                                                        config.max_seq_length, tokenizer,
                                                        output_mode, config.is_master,tok_type=config.teacher_type)

    eval_examples = processor.get_dev_examples('data/' + config.source + '/' + task_name +
                                               '/')
    eval_features = convert_examples_to_features(eval_examples, label_list,
                                                 config.max_seq_length, tokenizer,
                                                 output_mode, config.is_master,
                                                 tok_type=config.teacher_type)


    if config.teacher_type == 'rAg':
        train_dataloader = []
        eval_dataloader = []
        train_eval_dataloader = []
        for i in range(2):
            train_data, _ = get_tensor_data(output_mode, train_features[i])
            eval_data, eval_labels = get_tensor_data(output_mode, eval_features[i])
            train_eval_data, _ = get_tensor_data(output_mode, eval_features[i])
            if not config.multi_gpu:
                train_sampler = RandomSampler(train_data)
                train_eval_sampler = RandomSampler(train_eval_data)
            else:
                train_sampler = DistributedSampler(train_data)
                train_eval_sampler = DistributedSampler(train_eval_data)
            eval_sampler = SequentialSampler(eval_data)

            train_dataloader.append(DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size))
            eval_dataloader.append(DataLoader(eval_data, sampler=eval_sampler, batch_size=config.batch_size))
            train_eval_dataloader.append(DataLoader(train_eval_data, sampler=train_eval_sampler,
                                               batch_size=config.batch_size))
    else:
        train_data, _ = get_tensor_data(output_mode, train_features)
        eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
        train_eval_data, _ = get_tensor_data(output_mode, eval_features)
        if not config.multi_gpu:
            train_sampler = RandomSampler(train_data)
            train_eval_sampler = RandomSampler(train_eval_data)
        else:
            train_sampler = DistributedSampler(train_data)
            train_eval_sampler = DistributedSampler(train_eval_data)
        eval_sampler = SequentialSampler(eval_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=config.batch_size)
        train_eval_dataloader = DataLoader(train_eval_data, sampler=train_eval_sampler, batch_size=config.batch_size)




    if config.teacher_type == 'bert':
        config.bert_config = BertConfig.from_json_file(config.teacher_model + "/config.json")
        config.hidden_size = config.bert_config.hidden_size
    elif config.teacher_type == 'gpt2':
        config.gpt_config = GPT2Config.from_json_file(
            config.teacher_model + "/config.json")
        config.hidden_size = config.gpt_config.n_embd
    elif config.teacher_type == 'roberta':
        config.roberta_config = RobertaConfig.from_json_file(
            config.teacher_model + "/config.json")
        config.hidden_size = config.roberta_config.hidden_size
    elif config.teacher_type == 'rAg':
        config.roberta_config = RobertaConfig.from_json_file(
            config.teacher_model[0] + "/config.json")
        config.gpt_config = GPT2Config.from_json_file(
            config.teacher_model[1] + "/config.json")
        ##因为是一样的，所以用哪个去填hidden_size无所谓
        config.hidden_size = config.roberta_config.hidden_size

    return train_dataloader, train_eval_dataloader, eval_dataloader, output_mode, n_classes, config


from torch.utils.data.sampler import Sampler
class OrderdedSampler(Sampler):
    def __init__(self, dataset, order):
        self._dataset = dataset
        self._train_data_list = order
        self._train_data_list
    def __len__(self):
        return len(self._dataset)
    def __iter__(self):
        random.shuffle(self._train_data_list)
        for index in self._train_data_list:
            yield self._dataset[index]

def load_embedding(config, logger):
    data_path = os.path.join(config.data_path, config.saved_dataset)
    embedding_path = os.path.join(data_path, 'embedding')
    word_emb_file = os.path.join(embedding_path, config.word_emb_file)
    if config.is_master:
        logger.info("load word embeddings = {}".format(word_emb_file))
    with open(word_emb_file, "rb") as fh:
        word_mat = np.loadtxt(word_emb_file)
    char_vocab_file = os.path.join(embedding_path, config.char_vocab_file)
    char_emb_file = os.path.join(embedding_path, config.char_emb_file)
    if os.path.exists(char_emb_file):
        char_mat = np.loadtxt(char_emb_file)
    else:
        from sklearn.preprocessing import normalize
        char_vocab = Vocabulary()
        char_vocab.load(char_vocab_file)
        c_vocab_size = len(char_vocab)
        char_mat = np.random.rand(c_vocab_size, config.d_cvec)
        new_mat = []
        for ch in char_mat:
            new_mat.append(normalize([ch])[0])
        char_mat = np.array(new_mat)
    return word_mat, char_mat

def check_data_vaild(data1, data2):
    # data1, data2 = next(iter(data1)), next(iter(data2))
    def pad_replace(x):
        x = np.array(x)
        pad_mask = np.array([not(i == '[PAD]' or i == "<pad>") for i in x])
        new_x = x[pad_mask].tolist() + [f'[PAD] * { - sum(pad_mask - 1)}']
        return new_x
    def mask_replace(x):
        t = sum(x)
        new_x = f"1 * {t}, 0 * {len(x) - t}"
        return new_x
    with open('/data/lxk/NLP/github/darts-KD/data/MRPC-nas/embedding/vocab.txt') as f:
        vocab1 = {i:x.strip() for i, x in enumerate(f.readlines())}
    with open('/data/lxk/NLP/github/darts-KD/teacher_utils/teacher_model/MRPC/vocab.txt') as f:
        vocab2 = {i:x.strip() for i, x in enumerate(f.readlines())}

    sent_words = torch.split(data1[0], 1, dim=1)
    sent_words = [torch.squeeze(x, dim=1) for x in sent_words]

    mask = [x.ne(0) for x in sent_words]
    if len(mask) > 1:
        mask = torch.logical_or(mask[0], mask[1])
    else:
        mask = mask[0]

    print("SENT1:", pad_replace([vocab1[x.item()] for x in data1[0][0][0]]))
    if data1[0].shape[1] == 2:
        print("SENT2:", pad_replace([vocab1[x.item()] for x in data1[0][0][1]]))

    print("MASK:", mask_replace(mask[0]))

    print("LABEL:", data1[2][0].item())

    input_ids, input_mask, segment_ids, label_ids, seq_lengths = data2


    print("TEACHER SENT:", pad_replace([vocab2[x.item()] for x in input_ids[0]]))
    print("TEACHER MASK", mask_replace(input_mask[0]))
    print("TEACHER LABEL", label_ids[0].item())


def load_both(config, logger):
    from tokenization import BertTokenizer

    from bert_fineturn.data_processor.glue import glue_processors as processors
    from bert_fineturn.data_processor.glue import glue_output_modes as output_modes
    task_name = config.datasets
    processor = processors[task_name.lower()]()
    output_mode = output_modes[task_name.lower()]
    label_list = processor.get_labels()
    n_classes = len(label_list)
    ## BERT
    tokenizer = BertTokenizer.from_pretrained(config.teacher_model, do_lower_case=True)

    train_examples = processor.get_train_examples(config.data_path + config.source + "/" + task_name + '/')
    train_features = convert_examples_to_features(train_examples, label_list,
                                                        config.max_seq_length, tokenizer,
                                                        output_mode, config.is_master)
    train_data_bert, _ = get_tensor_data(output_mode, train_features)
    eval_examples = processor.get_dev_examples(config.data_path + config.source + "/" + task_name + '/')
    eval_features = convert_examples_to_features(eval_examples, label_list,
                                                    config.max_seq_length, tokenizer,
                                                    output_mode, config.is_master)
    eval_data_bert, eval_labels_bert = get_tensor_data(output_mode, eval_features)
    train_eval_data_bert, _ = get_tensor_data(output_mode, eval_features)

    train_sampler_bert = SequentialSampler(train_data_bert)
    train_eval_sampler_bert = SequentialSampler(train_eval_data_bert)
    eval_sampler_bert = SequentialSampler(eval_data_bert)

    train_dataloader_bert = DataLoader(train_data_bert, sampler=train_sampler_bert, batch_size=config.batch_size)
    eval_dataloader_bert = DataLoader(eval_data_bert, sampler=eval_sampler_bert, batch_size=config.batch_size)
    train_eval_dataloader_bert = DataLoader(train_eval_data_bert, sampler=train_eval_sampler_bert, batch_size=config.batch_size)

    #### GLOVE
    word_mat, char_mat = load_embedding(config, logger)
    # get data with meta
    logger.info("loading dataset {}".format(config.datasets))
    data_path = os.path.join(config.data_path, config.saved_dataset)
    train_data_glove, valid_data_glove, test_data_glove = get_data(data_path, config.datasets)
    logger.info("number of class for each dataset %s " % n_classes)

    train_sampler_glove = SequentialSampler(train_data_glove)
    train_eval_sampler_glove = SequentialSampler(valid_data_glove)
    eval_sampler_glove = SequentialSampler(valid_data_glove)

    train_dataloader_glove = DataLoader(train_data_glove, sampler=train_sampler_glove, batch_size=config.batch_size)
    eval_dataloader_glove = DataLoader(valid_data_glove, sampler=eval_sampler_glove, batch_size=config.batch_size)
    train_eval_dataloader_glove = DataLoader(valid_data_glove, sampler=train_eval_sampler_glove, batch_size=config.batch_size)
    # print("############## TRAIN DATA CHECK ##############")
    # check_data_vaild(train_dataloader_glove, train_dataloader_bert)
    # print("############## VAILD DATA CHECK ##############")
    # check_data_vaild(train_eval_dataloader_glove, train_eval_dataloader_bert)
    # exit(0)
    return train_dataloader_glove, eval_dataloader_glove, train_eval_dataloader_glove, train_dataloader_bert, eval_dataloader_bert, train_eval_dataloader_bert, (word_mat, char_mat), output_mode, n_classes


class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.33, last_epoch=-1):
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.total_epochs = 150
        self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        if self.curr_temp < self.temp_min:
            self.curr_temp = self.temp_min
        return self.curr_temp

class RandomSamplerByOrder(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples
    
def bert_batch_split(data, rank):
    data = [x.to(f"cuda:{rank}", non_blocking=True) for x in data]
    input_ids, input_mask, segment_ids, label_ids, seq_lengths = data
    X = [input_ids, input_mask, segment_ids, seq_lengths]
    Y = label_ids
    return X, Y

def get_acc_from_pred(output_mode, task_name, preds, eval_labels):
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name.lower(), preds, eval_labels)

    if task_name.lower() == "cola":
        acc = result['mcc']
    elif task_name.lower() in ["sst-2", "mnli", "mnli-mm", "qnli", "rte", "books"]:
        acc = result['acc']
    elif task_name.lower() in ["mrpc", "qqp"]:
        acc = result['f1']
    elif task_name.lower() == "sts-b":
        acc = result['corr']
    return result, acc

if __name__ == "__main__":
    top1 = AverageMeter()
    top1.update(0.5, 10)
    print(top1.avg)