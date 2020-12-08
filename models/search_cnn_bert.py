""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
import genotypes as gt
from torch.nn.parallel._functions import Broadcast
import logging
from models import blocks
import copy
import math

import json
import sys



def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i + len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """

    def __init__(self, nodes, C_pp, C_p, C, all_return=False):
        """
        Args:
            nodes: # of intermediate nodes
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.nodes = nodes

        self.preproc1 = blocks.Initialized_Conv1d(C_p, C, kernel_size=1)
        self.preproc0 = blocks.Initialized_Conv1d(C_pp, C, kernel_size=1)

        self.all_return = all_return
        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2 + i):  # include 1 input nodes
                # reduction should be used only for input node
                stride = 1
                op = blocks.MixedOp(C, stride)
                self.dag[i].append(op)

    def forward(self, s0, s1, w_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)
        # print(s0.shape, s1.shape)
        # states = [s0, s1]
        states = [s0, s1]
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)
        s_out = torch.cat(states[2:], dim=1)
        return s_out


class SearchCNN(nn.Module):
    """ Search CNN model """

    def __init__(self, config, n_classes,):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            nodes: # of intermediate nodes in Cell
            stem_multiplier
        """

        super().__init__()

        C_in = config.bert_config.hidden_size
        C = config.init_channels
        self.n_layers = config.layers
        nodes = config.nodes
        stem_multiplier=config.stem_multiplier
        bert_config = config.bert_config
        self.use_kd=config.use_kd
        self.all_return = config.all_return

        # self.C_in = C_in
        # self.C = C
        self.n_classes = n_classes
        # self.n_layers = n_layers
        # self.class_nums = class_nums
        C_cur = stem_multiplier * C
        # print(word_mat.shape)
        # print("C_out = ", C_cur)

        self.stem = blocks.BertEmbeddings(bert_config, C)

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        # C_pp, C_p, C_cur = C_cur, C_cur, C
        C_pp, C_p, C_cur = C_in, C_in, C
        self.cells = nn.ModuleList()
        for i in range(self.n_layers):
            print(C_pp, C_p, C_cur)
            cell = SearchCell(nodes, C_pp, C_p, C_cur)
            self.cells.append(cell)
            C_cur_out = C_cur * nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(C_p, self.n_classes)
        if self.use_kd:
            self.fit_dense = nn.Linear(C_cur_out, 768)
        # if task_type is not None:
        #     self.merge = nn.Linear(C_p * 4, C_p * 2)
        # self.ls = nn.Sigmoid(dim=-1)
    def forward(self, x, weights_normal, layer_out=False):
        # sent_num = x[0].shape[1]
        # sent_words = torch.split(x[0], 1, dim=1)
        # sent_chars = torch.split(x[1], 1, dim=1)
        # sent_words = [torch.squeeze(x, dim=1) for x in sent_words]
        student_layer_out = []
        # sent_chars = [torch.squeeze(x, dim=1) for x in sent_chars]
        input_ids, input_mask, segment_ids, seq_lengths = x
        s0 = s1 = self.stem(input_ids, segment_ids)
        # if len(sent_words) == 2:
        #     s0s = [self.stem(sent_words[0], sent_chars[0]) for i in range(sent_num)]
        #     s1s = [self.stem(sent_words[1], sent_chars[1]) for i in range(sent_num)]
        # else:
        #     s0s = s1s = [self.stem(sent_words[0], sent_chars[0])]
        outs = []
        # for s0, s1, mask in zip(s0s, s1s, sent_masks):
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, weights_normal)
            if self.use_kd:
                student_layer_out.append(self.fit_dense(s1.permute(0,2,1)))
        # out = s1[:, :, 0]
        out = self.gap(s1).squeeze(-1)
        # out_max = torch.sum(s1.transpose(2, 1) * input_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(input_mask, dim=1, keepdim=True)
        # out = torch.cat([out_mean, out_max], dim=-1)
        logits = self.linear(out)
        # outs.append(out)
        # if len(outs) == 1:
        #     logits = self.linear(outs[0])
        # else:
        #     logits = self.linear(self.merge(torch.cat(outs, dim=-1)))
        if layer_out:
            return logits, student_layer_out
        else:
            return logits
    
def _get_onehot_mask(log_alpha):
    uni = torch.ones_like(log_alpha)
    m = torch.distributions.one_hot_categorical.OneHotCategorical(uni)
    one_hot = m.sample()
    return one_hot

class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, config, n_classes, output_mode):
        super().__init__()
        n_layers = config.layers
        self.nodes = config.nodes
        device_ids = config.gpus
        
        self.output_mode = output_mode
        self.n_classes = config.n_classes
        # self.all_return = config.all_return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        # self.class_nums = class_nums
        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)
        self.alpha_normal = nn.ParameterList()

        self._alphas = []

        for i in range(self.nodes):
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.ones(i + 2, n_ops)))
        # self.alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
        # setup alphas list
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))
        self.net = SearchCNN(config, n_classes,)
        self.apply(init_weights)
    def forward(self, x, train=True, one_step=False, random_sample=False, layer_out=False):
        # weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        # weights_normal = [F.gumbel_softmax(alpha,tau=1e-3,hard=True) for alpha in self.alpha_normal]
        if random_sample:
            _softmax = lambda x: _get_onehot_mask(x)
        elif not one_step or not train:
            _softmax = nn.Softmax(dim=-1)
        else:
            _softmax = lambda x: F.gumbel_softmax(x, tau=1e-3, hard=True, dim=-1)
            

        weights_normal = [_softmax(alpha) for alpha in self.alpha_normal]
        # if len(self.device_ids) == 1 or not train:
        return self.net(x, weights_normal, layer_out=layer_out)
        
        # tmp_devices = self.device_ids
        # wnormal_copies = broadcast_list(weights_normal, tmp_devices)
        # xs = list(zip(*[nn.parallel.scatter(i, tmp_devices) for i in x]))
        # replicas = nn.parallel.replicate(self.net, tmp_devices)
        # layer_out = [layer_out] * len(tmp_devices)
        # outputs = nn.parallel.parallel_apply(
        #     replicas, list(zip(xs, wnormal_copies, layer_out)), devices=tmp_devices)
        # return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, ys):
        y = ys
        logits = self.forward(X)
        y = y.long() if self.output_mode == 'classification' else y.float()

        if self.output_mode == "classification":
            loss_fct = nn.CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, self.n_classes), y.view(-1))
        elif self.output_mode == "regression":
            loss_fct = nn.MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), y.view(-1))

        return tmp_eval_loss

    def crit(self, logits, y):
        y = y.long() if self.output_mode == 'classification' else y.float()

        if self.output_mode == "classification":
            loss_fct = nn.CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, self.n_classes), y.view(-1))
        elif self.output_mode == "regression":
            loss_fct = nn.MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), y.view(-1))

        return tmp_eval_loss

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alphas in self.alpha_normal:
            logger.info(F.softmax(alphas, dim=-1))
            # logger.info(alpha)

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        # gene_reduce = gt.parse(self.alpha_reduce, k=2)
        gene_reduce = []
        concat = range(1, 1 + self.nodes)  # concat all intermediate nodes

        return gt.Genotype(
            normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        tmp = [p for n, p in self.net.named_parameters() if p.requires_grad and 'alpha' not in n]
        return tmp

    def named_weights(self):
        tmp = []
        for n, p in self.net.named_parameters():
            if p.requires_grad and 'alpha' not in n:
                tmp.append((n, p))
        return tmp

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LSTM):
            nn.init.xavier_uniform_(m.weight_ih_l0.data)
            nn.init.orthogonal_(m.weight_hh_l0.data)
            nn.init.constant_(m.bias_ih_l0.data, 0.0)
            nn.init.constant_(m.bias_hh_l0.data, 0.0)
            hidden_size = m.bias_hh_l0.data.shape[0] // 4
            m.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

            if (m.bidirectional):
                nn.init.xavier_uniform_(m.weight_ih_l0_reverse.data)
                nn.init.orthogonal_(m.weight_hh_l0_reverse.data)
                nn.init.constant_(m.bias_ih_l0_reverse.data, 0.0)
                nn.init.constant_(m.bias_hh_l0_reverse.data, 0.0)
                m.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
