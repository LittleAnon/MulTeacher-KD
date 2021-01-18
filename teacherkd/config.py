""" Config class for search/augment """
import argparse
import os
from functools import partial
from defaultconfig import read_ini
from teacherkd import genotypes as gt
common_ini_section = 'common'
train_ini_section = 'train'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(BASE_DIR)

def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(
        name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


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


class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")

        parser.add_argument('--datasets', type=str,
                            default='MRPC', help='input dataset')

        parser.add_argument('--batch_size', type=int,
                            default=32, help='batch size')
        parser.add_argument('--max_seq_length', type=int,
                            default=128, help='# of layers')

        parser.add_argument('--lr', type=float,
                            default=0.025, help='lr for weights')
        parser.add_argument('--lr_min', type=float,
                            default=5e-4, help='minimum lr for weights')

        parser.add_argument('--weight_decay', type=float,
                            default=3e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float,
                            default=5., help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int,
                            default=50, help='print frequency')
        parser.add_argument('--n_gpu', default=1, help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=20,
                            help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=256)
        parser.add_argument('--layers', type=int,
                            default=4, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int,
                            default=0, help='# of workers')

        parser.add_argument('--n_classes', type=int,
                            default=2, help='numbers of classes')

        parser.add_argument('--drop_path_prob', type=float,
                            default=0.2, help='probability of an path to be zeroed.(我觉得在结构固定时没用)')

        parser.add_argument('--teacher_type', type=str, default=None,
                            help='Pre-trained teacher model selected in the list: bert, roberta, gpt2,rAg(roberta+gpt2)')
        parser.add_argument('--student_type', type=str, default=None,
                            help='student model selected in the list: cnn,transform')

        parser.add_argument('--use_kd', type=str2bool,
                            default=True, help='Whether to use the teacher model')
        parser.add_argument('--use_emd', type=str2bool,
                            default=True, help='Whether to use embedding')
        parser.add_argument('--kd_alpha', type=float,
                            default=1.0, help='only use if use_kd == True, The coefficient of knowledge distillation')

        parser.add_argument('--quantity', type=float,
                            default=1, help='only use if use_emd == True,(0.00001~10000) the amount of information transfered from teacher model to student model')
        parser.add_argument('--update_emd', type=str2bool,
                            default=True, help='whether to update embedding')
        parser.add_argument('--local_rank', type=int,
                            default=-1, help='local rank is the local id for GPUs in the same node.')
        parser.add_argument('--emd_only', type=str2bool,
                            default=True, help='True: only use embedding,False:do not use embeddings')
        parser.add_argument('--momentum', type=float,
                            default=0.9,
                            help='Momentum is a stochastic optimization method that adds a momentum term to regular stochastic gradient descent')
        parser.add_argument(
            '--genotype', default="Genotype(normal=[[('stdconv_3', 0), ('stdconv_5', 1)], [('stdconv_7', 1), ('dilconv_5', 0)], [('stdconv_5', 1), ('dilconv_3', 0)]], normal_concat=range(2, 5), reduce=[], reduce_concat=range(2, 5))", type=str, help='Cell genotype')
        parser.add_argument('--filegt', default=-1,
                            type=int, help='Cell genotype file')
        parser.add_argument('--randgt', action='store_true',
                            help='Cell genotype')
        parser.add_argument('--hidn2attn', type=bool,
                            default=False, help='change the hidden states to attentions')

        parser.add_argument('--tokenizer_name', type=bool,
                            default=None, help=' tokenizer name')
        parser.add_argument('--use_fast_tokenizer', type=bool,
                            default=True, help='whether to use fast tokenizer')
        parser.add_argument('--default_conf_name', type=str,
                            default='suzhan', help='default init config file name')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        ini_config = read_ini.read_properties_from_config_ini(args.default_conf_name)

        teacher_model_suffix = self.datasets.lower()
        teacher_model_dict = {
            'roberta': ini_config[common_ini_section]['roberta_finetune_model_path'].format(teacher_model_suffix),
            'gpt2': ini_config[common_ini_section]['gpt2_finetune_model_path'].format(teacher_model_suffix),
            'bert': ini_config[common_ini_section]['bert_finetune_model_path'].format(teacher_model_suffix),
            'rAg': [ini_config[common_ini_section]['roberta_finetune_model_path'].format(teacher_model_suffix),
                    ini_config[common_ini_section]['gpt2_finetune_model_path'].format(teacher_model_suffix)]

        }

        self.teacher_model = teacher_model_dict[self.teacher_type]
        self.ini_config = ini_config

        self.data_path = ini_config[train_ini_section]['data_base_dir']

        if teacher_model_suffix in ["sst-2", "mnli", "mnli-mm", "qnli", "rte", 'sts-b', 'qqp', 'cola', 'mrpc']:
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
        self.path = ini_config[train_ini_section]['local_save_dir']
        self.genotype = gt.from_str(self.genotype)

    def gen_geno_random(self):
        '''
        生成随机的CNN结构
        :return:
        '''
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
            'max_pool_3x3', ]
        s = [random.choice(PRIMITIVES) for _ in range(6)]
        c = []
        for i in range(6):
            c.append(random.randint(0, math.ceil(i / 2)))
        geno = "Genotype(normal=[[('{}', {}), ('{}', {})], [('{}', {}), ('{}', {})], [('{}', {}), ('{}', {})]], normal_concat=range(2, 5), reduce=[], reduce_concat=range(2, 5))".format(
            s[0], c[0], s[1], c[1], s[2], c[2], s[3], c[3], s[4], c[4], s[5], c[5])
        self.genotype = geno

