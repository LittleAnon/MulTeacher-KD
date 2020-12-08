""" Config class for search/augment """
import argparse
import os
import genotypes as gt
from functools import partial
import torch
import socket

def get_host_ip():
   try:
      s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      s.connect(('8.8.8.8', 80))
      ip = s.getsockname()[0]
   finally:
      s.close()
   return ip

def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser

teacher_model_dict = {
            'roberta':'/disk1/wuxiangbo/pretrainModel/roberta/robert_mrpc',
            'gpt2':'/disk1/wuxiangbo/darts-KD/teacher_utils/teacher_model/gpt2_mrpc',
            'bert':'/disk1/wuxiangbo/pretrainModel/bert/bert_mrpc'
        }
def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]

def gen_geno_random():
    import math
    import random
    PRIMITIVES = [
    'stdconv_3',
    'stdconv_5',
    'stdconv_7',
    'skip_connect',
    'dilconv_3',
    'dilconv_5',
    'dilconv_7',
    'avg_pool_3x3',
    'max_pool_3x3',]
    s = [random.choice(PRIMITIVES) for _ in range(6)]
    c = []
    for i in range(6):
        c.append(random.randint(0, math.ceil(i / 2)))
    geno = "Genotype(normal=[[('{}', {}), ('{}', {})], [('{}', {}), ('{}', {})], [('{}', {}), ('{}', {})]], normal_concat=range(2, 5), reduce=[], reduce_concat=range(2, 5))".format(s[0], c[0], s[1], c[1], s[2], c[2], s[3], c[3], s[4], c[4], s[5], c[5])
    return geno

class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class SearchConfig(BaseConfig):

    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--name', default='SST-2_CNN')
        parser.add_argument('--datasets', type=str, default='MRPC', help='input dataset')
        parser.add_argument('--max_seq_length', type=int, default=128, help='# of layers')
        parser.add_argument('--saved_dataset', default='MRPC-nas', help='input dataset')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        parser.add_argument('--w_lr', type=float, default=0.02, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=5e-4, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4, help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5., help='gradient clipping for weights')
        parser.add_argument('--init_channels', type=int, default=128)
        parser.add_argument('--layers', type=int, default=4, help='# of layers')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3, help='weight decay for alpha')
        parser.add_argument('--alpha_lr', type=float, default=1e-3, help='lr for alpha')
        parser.add_argument('--nodes', type=int, default=3, help='# of nodes')
        parser.add_argument('--epochs', type=int, default=250, help='# of training epochs')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')

        parser.add_argument('--limit', type=int, default=128, help='max length')
        parser.add_argument('--char_limit', type=int, default=12, help='max length')
        parser.add_argument('--stem_multiplier', type=int, default=1, help='# of stem_multiplier')
        parser.add_argument('--d_vec', type=int, default=300, help='# of nodes')
        parser.add_argument('--d_cvec', type=int, default=50, help='# of nodes')
        parser.add_argument('--n_classes', type=int, default=2, help='# of classes')
        parser.add_argument('--workers', type=int, default=0, help='# of workers')
        parser.add_argument('--max_word_v_size', type=int, default=100000, help='max word num')
        parser.add_argument('--max_char_v_size', type=int, default=200, help='max char num')
        parser.add_argument('--load_tasks', type=int, default=1, help='load pickle glue data')

        parser.add_argument('--one_step', type=str2bool, default=False, help='# of layers')
        parser.add_argument('--random_sample_epoch', type=int, default=0, help='# of layers')
        parser.add_argument('--freeze_alpha', type=int, default=0, help='# of layers')

        parser.add_argument('--distributed', type=bool, default=False, help='# of layers')
        parser.add_argument('--n_gpu', type=int, default=1, help='# of layers')
        parser.add_argument('--local_rank', type=int, default=-1, help='# of training epochs')

        parser.add_argument('--teacher_model', type=str, default='teacher_utils/teacher_model/RTE', help='# of layers')
        parser.add_argument('--use_kd', type=str2bool, default=True, help='# of layers')
        parser.add_argument('--use_emd', type=str2bool, default=True, help='# of layers')
        parser.add_argument('--kd_alpha', type=float, default=0.8, help='# of layers')
        parser.add_argument('--emd_rate', type=float, default=0.0, help='# of layers')
        parser.add_argument('--update_emd', type=str2bool, default=False, help='# of layers')

        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--log_name', type=str, default="", help='load pickle glue data')
        parser.add_argument('--emd_only', type=str2bool, default=False, help='# of layers')
        parser.add_argument('--save_supernet', type=str, default="", help='# of layers')
        parser.add_argument('--restore', type=str, default="", help='# of layers')
        parser.add_argument('--alpha_only', type=str2bool, default=False, help='# of layers')
        parser.add_argument('--alpha_optim', type=str, default='adam', help='# of layers')
        parser.add_argument('--eval_during_train', type=str2bool, default=False, help='# of layers')
        parser.add_argument('--l1', type=float, default=0.001, help='# of layers')
        parser.add_argument('--_temp', type=float, default=1.0, help='# of layers')
        parser.add_argument('--tb_dir', type=str, default="", help='# of layers')
        parser.add_argument('--t_att_type', type=str, default="", help='# of layers')
        parser.add_argument('--s_att_type', type=str, default="", help='# of layers')
        parser.add_argument('--hidn2attn', type=str2bool, default=False, help='# of layers')
        parser.add_argument('--add_op', type=str, default='cnn', help='# of layers')
        parser.add_argument('--rnn_op', type=bool, default=False, help='# of layers')
        parser.add_argument('--mul_att_out', type=bool, default=False, help='# of layers')
        parser.add_argument('--sep_alpha', type=bool, default=False, help='# of layers')
        parser.add_argument('--alpha_ac_rate', type=float, default=0.0, help='# of layers')
        parser.add_argument('--alpha_ep', type=int, default=-1, help='# of layers')
        parser.add_argument('--weight_rate', type=float, default=1.0, help='# of layers')
        parser.add_argument('--add_softmax', type=str2bool, default=True, help='# of layers')

        
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
        if 'att' in self.add_op:
            gt.PRIMITIVES.insert(-1, 'self_attn')
        elif 'rnn' in self.add_op:
            gt.PRIMITIVES.insert(-1, 'rnn')
        elif 'cnn' in self.add_op:
            gt.PRIMITIVES.insert(-1, 'dilconv_3')
            gt.PRIMITIVES.insert(-1, 'dilconv_5')
            gt.PRIMITIVES.insert(-1, 'dilconv_7')
        if self.datasets.lower() in ["sst-2", "mnli", "mnli-mm", "qnli", "rte", 'sts-b', 'qqp', 'cola', 'mrpc']:
            self.source = 'glue_data'
        else:
            self.source = 'amazon'
        self.teacher_model = 'teacher_utils/teacher_model/' + self.datasets

        self.emd_only = 0 if self.emd_only else 1
        self.data_path = './data/'
        self.data_src_path = f'./data/{self.source}/'
        self.word_emb_file = 'embeddings.txt'
        self.char_emb_file = 'char_embed.txt'
        self.word_vocab_file = 'vocab.txt'
        self.char_vocab_file = 'char_vocab.txt'
        self.path = os.path.join('searchs', self.name)
        self.plot_path = os.path.join(self.path, 'plots')
        # self.gpus = parse_gpus(self.gpus)
        if get_host_ip() == '192.168.193.24':
            self.student_bert = '/data/lxk/NLP/TinyBERT/zhh_emd/model/pytorch_bert_base_uncased'
        elif get_host_ip() == '192.168.193.20':
            self.student_bert = '/disk2/zhh/TinyBERT/model/bert_base_uncased'
        print("=============MODEL CONFIG==============")
        print("task={}\nlayer={}\nhidden={}\nbatch_size={}\nepochs={}".format(self.datasets,self.layers, self.init_channels, self.batch_size, self.epochs))
        print("=============KD CONFIG==============")
        print("use KD={}\nuse emd={}\nkd ALPHA={}\none_step={}\nrandom epoch={}\nfreeze epoch={}\ntrain alpha only={}\nemd only={}".format(self.use_kd, self.use_emd, self.kd_alpha, self.one_step, self.random_sample_epoch, self.freeze_alpha, self.alpha_only, self.emd_only))


class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--name', default='demo')
        parser.add_argument('--datasets', type=str, default='MRPC', help='input dataset')
        parser.add_argument('--saved_dataset', default='MRPC-nas', help='input dataset')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        parser.add_argument('--limit', type=int, default=128, help='max length')
        parser.add_argument('--char_limit', type=int, default=12, help='max length')
        parser.add_argument('--lr', type=float, default=1e-3, help='lr for weights')
        parser.add_argument('--lr_min', type=float, default=5e-4, help='minimum lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--n_gpu', default=1, help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=20, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=256)
        parser.add_argument('--layers', type=int, default=4, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=0, help='# of workers')
        parser.add_argument('--d_vec', type=int, default=300, help='# of nodes')
        parser.add_argument('--d_cvec', type=int, default=50, help='# of nodes')
        parser.add_argument('--n_classes', type=int, default=2, help='# of classes')
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')
        parser.add_argument('--max_seq_length', type=int, default=128, help='# of layers')
        parser.add_argument('--teacher_type', type=str, default='roberta', help='Pre-trained teacher model selected in the list: bert, roberta, gpt2')


        parser.add_argument('--use_kd', type=str2bool, default=True, help='# of layers')
        parser.add_argument('--use_emd', type=str2bool, default=True, help='# of layers')
        parser.add_argument('--kd_alpha', type=float, default=1.0, help='# of layers')
        parser.add_argument('--emd_rate', type=float, default=0.0, help='# of layers')
        parser.add_argument('--update_emd', type=str2bool, default=True, help='# of layers')
        parser.add_argument('--local_rank', type=int, default=-1, help='# of training epochs')
        parser.add_argument('--emd_only', type=str2bool, default=True, help='# of layers')

        parser.add_argument('--genotype', default="Genotype(normal=[[('stdconv_3', 0), ('stdconv_5', 1)], [('stdconv_7', 1), ('dilconv_5', 0)], [('stdconv_5', 1), ('dilconv_3', 0)]], normal_concat=range(2, 5), reduce=[], reduce_concat=range(2, 5))", type=str,help='Cell genotype')
        parser.add_argument('--filegt', default=-1, type=int,help='Cell genotype')
        parser.add_argument('--randgt', action='store_true',help='Cell genotype')
        parser.add_argument('--hidn2attn', type=bool, default=False, help='# of layers')
        parser.add_argument('--sep_alpha', type=bool, default=False, help='# of layers')
        parser.add_argument('--tokenizer_name', type=bool, default=None, help='tokenizer_name')
        parser.add_argument('--use_fast_tokenizer', type=bool, default=True, help='# of layers')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))


        self.teacher_model = teacher_model_dict[self.teacher_type]
        
        # self.teacher_model = 'teacher_utils/teacher_model/' + self.datasets
        self.data_path = './data/'

        if self.datasets.lower() in ["sst-2", "mnli", "mnli-mm", "qnli", "rte", 'sts-b', 'qqp', 'cola', 'mrpc']:
            self.source = 'glue_data'
        else:
            self.source = 'amazon'

        if self.genotype == "" and self.filegt == -1 and not self.randgt:
            raise Exception("No genotypes choosen")
        elif self.filegt != -1 and self.genotype == "":
            with open('train_genotyps.txt') as f:
                self.genotype = f.readlines()[self.filegt]
        elif self.randgt:
            self.gen_geno_random()
        self.emd_only = 0 if self.emd_only else 1
        self.data_src_path = './data/src/'
        self.word_emb_file = 'embeddings.txt'
        self.char_emb_file = 'char_embed.txt'
        self.word_vocab_file = 'vocab.txt'
        self.char_vocab_file = 'char_vocab.txt'
        self.path = os.path.join('augments', self.name)
        os.makedirs(self.path, exist_ok=True)
        self.genotype = gt.from_str(self.genotype)

    def gen_geno_random(self):
        import math
        import random
        PRIMITIVES = [
        'stdconv_3',
        'stdconv_5',
        'stdconv_7',
        'skip_connect',
        'dilconv_3',
        'dilconv_5',
        'dilconv_7',
        'avg_pool_3x3',
        'max_pool_3x3',]
        s = [random.choice(PRIMITIVES) for _ in range(6)]
        c = []
        for i in range(6):
            c.append(random.randint(0, math.ceil(i / 2)))
        geno = "Genotype(normal=[[('{}', {}), ('{}', {})], [('{}', {}), ('{}', {})], [('{}', {}), ('{}', {})]], normal_concat=range(2, 5), reduce=[], reduce_concat=range(2, 5))".format(s[0], c[0], s[1], c[1], s[2], c[2], s[3], c[3], s[4], c[4], s[5], c[5])
        self.genotype = geno
