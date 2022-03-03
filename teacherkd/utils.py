# coding=utf-8
""" Utilities """
import os
import shutil

import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import RobertaConfig, AutoTokenizer, \
    GPT2Config, BertConfig
import os
import socket

logger = logging.getLogger('imagenet_training')

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "books": "classification",
}

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
NUM_LABLE = {'sts-b': 1, 'SST-2': 2}





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




def convert_examples_to_features(examples, label_list, max_seq_length, tokenizers, output_mode, is_master=True, tok_type=None):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    features = [[], []]
    if tok_type == 'rAg':
        tok_type_list = ['roberta', 'gpt2']
    else:
        tok_type_list = [tok_type]
        tokenizers = [tokenizers]
    for i in range(len(tok_type_list)):
        tokenizer = tokenizers[i]
        tok_type_temp = tok_type_list[i]

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0 and is_master:
                print("Writing example %d of %d" % (ex_index, len(examples)))
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                if tok_type_temp == 'gpt2':
                    if len(tokens_a) > max_seq_length - 1:
                        tokens_a = tokens_a[:(max_seq_length - 1)]
                else:
                    if len(tokens_a) > max_seq_length - 2:
                        tokens_a = tokens_a[:(max_seq_length - 2)]
            # 和finetune的数据格式保持一致，bert和gpt2和roberta的特殊字符添加格式还不太一样
            if tok_type_temp == 'bert':
                cls_ = tokenizer.cls_token
                sep_ = tokenizer.sep_token
                pad_ = tokenizer.pad_token_id
                tokens = [cls_] + tokens_a + [sep_]
            elif tok_type_temp == 'gpt2':
                cls_ = tokenizer.bos_token
                sep_ = tokenizer.eos_token
                pad_ = tokenizer.pad_token_id
                tokens = tokens_a
            elif tok_type_temp == 'roberta':
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
                print("*** Example for %s ***" % (tok_type_temp))
                print("guid: %s" % (example.guid))
                print("tokens: %s" % " ".join([str(x)
                                               for x in tokens]).encode('utf-8'))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("input_mask: %s" % " ".join(
                    [str(x) for x in input_mask]))
                print("segment_ids: %s" % " ".join(
                    [str(x) for x in segment_ids]))
                print("label: {}".format(example.label))
                print("label_id: {}".format(label_id))

            features[i].append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    seq_length=seq_length))
    if tok_type == 'rAg':
        return features
    elif len(features) > 0:
        return features[0]



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

    # torch.Size([50257, 768])
    new_dict['stem.word_embeddings.weight'] = pretrain_dict['transformer.wte.weight']
    # torch.Size([1024, 768])
    new_dict['stem.position_embeddings.weight'] = pretrain_dict['transformer.wpe.weight']

    # torch.Size([768])
    new_dict['stem.LayerNorm.weight'] = pretrain_dict['transformer.h.0.ln_1.weight']
    # torch.Size([768])
    new_dict['stem.LayerNorm.bias'] = pretrain_dict['transformer.h.0.ln_1.bias']
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
                new_dict[new_k.replace(
                    'bert.embeddings', 'net.stem')] = pretrain_dict[key]
            else:
                new_dict[key.replace('bert.embeddings', 'stem')
                         ] = pretrain_dict[key]
    print("="*10 + " RESTORE KEYS" + "="*10)
    for k, v in model.named_parameters():
        if k in new_dict:
            print(k)

    model.load_state_dict(new_dict, strict=False)



def load_glue_dataset(config):
    from transformers.data.processors.glue import glue_processors as processors

    task_name = config.datasets
    processor = processors[task_name.lower()]()
    output_mode = glue_output_modes[task_name.lower()]
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
        tokenizer = [tokenizer_r, tokenizer_g]
        print("len tokenizer:",len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name if config.tokenizer_name else config.teacher_model,

            use_fast=config.use_fast_tokenizer
        )

    train_examples = processor.get_train_examples(
        config.ini_config['train']['data_base_dir'] + config.source + '/' + task_name + '/')
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  config.max_seq_length, tokenizer,
                                                  output_mode, config.is_master, tok_type=config.teacher_type)
    print("len featuers:",len(train_features))
    eval_examples = processor.get_dev_examples(config.ini_config['train']['data_base_dir'] + config.source + '/' + task_name +
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
            eval_data, eval_labels = get_tensor_data(
                output_mode, eval_features[i])
            train_eval_data, _ = get_tensor_data(output_mode, eval_features[i])
            if not config.multi_gpu:
                train_sampler = RandomSampler(train_data)
                train_eval_sampler = RandomSampler(train_eval_data)
            else:
                train_sampler = DistributedSampler(train_data)
                train_eval_sampler = DistributedSampler(train_eval_data)
            eval_sampler = SequentialSampler(eval_data)

            train_dataloader.append(DataLoader(
                train_data, sampler=train_sampler, batch_size=config.batch_size))
            eval_dataloader.append(DataLoader(
                eval_data, sampler=eval_sampler, batch_size=config.batch_size))
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

        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=config.batch_size)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=config.batch_size)
        train_eval_dataloader = DataLoader(
            train_eval_data, sampler=train_eval_sampler, batch_size=config.batch_size)

    if config.teacher_type == 'bert':
        config.bert_config = BertConfig.from_json_file(
            config.teacher_model + "/config.json")
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
        # 因为是一样的，所以用哪个去填hidden_size无所谓
        config.hidden_size = config.roberta_config.hidden_size

    return train_dataloader, train_eval_dataloader, eval_dataloader, output_mode, n_classes, config




def get_tensor_data(output_mode, features,mul_teacher=False):

    # if not mul_teacher:
    #     features = features[0]

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                all_seq_lengths)
    return tensor_data, all_label_ids



def init_gpu_params(params):
    """
    Handle single and multi-GPU / multi-node.
    """
    if params.n_gpu <= 0:
        params.local_rank = 0
        params.master_port = -1
        params.is_master = True
        params.multi_gpu = False
        return

    assert torch.cuda.is_available() == True

    logger.info("Initializing GPUs")
    if params.n_gpu > 1:
        assert params.local_rank != -1

        params.world_size = int(os.environ["WORLD_SIZE"])
        params.n_gpu_per_node = int(os.environ["N_GPU_NODE"])
        params.global_rank = int(os.environ["RANK"])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.multi_gpu = True

        assert params.n_nodes == int(os.environ["N_NODES"])
        assert params.node_id == int(os.environ["NODE_RANK"])

    # local job (single GPU)
    else:
        assert params.local_rank == -1

        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1
        params.multi_gpu = False

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in multi-node distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1

    # summary
    PREFIX = f"--- Global rank: {params.global_rank} - "
    logger.info(PREFIX + "Number of nodes: %i" % params.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % params.node_id)
    logger.info(PREFIX + "Local rank     : %i" % params.local_rank)
    logger.info(PREFIX + "World size     : %i" % params.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    logger.info(PREFIX + "Master         : %s" % str(params.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if params.multi_gpu:
        logger.info("Initializing PyTorch distributed")
        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
        )


