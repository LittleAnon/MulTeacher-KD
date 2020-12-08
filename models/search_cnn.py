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
from modeling import BertLayerNorm
import json
import sys

def convert_to_attn(hidns, mask):
    if type(hidns[0]) is not tuple:
        hdim = hidns[0].shape[-1]
        attns = [torch.matmul(x, x.transpose(2, 1)) / sqrt(hdim) for x in hidns]
        mask = mask.unsqueeze(1)
        mask = (1.0 - mask) * -10000.0
        attns = [softmax(x + mask, dim=-1) for x in attns]
    else:
        hidns = [torch.stack(x, dim=1) for x in hidns]
        hdim = hidns[0][0].shape[-1]
        attns = [torch.matmul(x, x.transpose(-1, -2)) / sqrt(hdim) for x in hidns]
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = (1.0 - mask) * -10000.0
        attns = [softmax(x + mask, dim=-1) for x in attns]
    return attns


def replace_masked(tensor, mask, value):
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i + len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention "
                             "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.key(hidden_states)
        mixed_key_layer = self.key(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        # attention_scores = nn.functional.relu(attention_scores)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attenotin_scores = attention_scores + attention_mask

        return attenotin_scores


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """

    def __init__(self, nodes, C_pp, C_p, C):
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

        self.concat_conv1 = blocks.Initialized_Conv1d(C_p, C*3, kernel_size=1)
        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2 + i):  # include 1 input nodes
                # reduction should be used only for input node
                stride = 1
                op = blocks.MixedOp(C, stride)
                self.dag[i].append(op)
        self.batch_norm = BertLayerNorm(C*3)
    def forward(self, s0, s1, w_dag):
        s1_out = self.concat_conv1(s1)

        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)
        s_out = torch.cat(states[2:], dim=1)

        s_out = self.batch_norm((s_out +  s1_out).permute(0, 2, 1)).permute(0, 2, 1)
        return s_out


class SearchCNN(nn.Module):
    """ Search CNN model """

    def __init__(self, config, n_classes):
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
        self.nodes = config.nodes
        stem_multiplier = config.stem_multiplier
        # bert_config = config.bert_config
        self.use_kd = config.use_kd
        self.all_return = config.use_emd
        self.return_node_out = config.s_att_type == "node"
        self.n_classes = n_classes
        C_cur = stem_multiplier * C
        # print(word_mat.shape)
        # print("C_out = ", C_cur)
        self.attn_cells = nn.ModuleList()
        self.stem = blocks.BertEmbeddings(config.bert_config)
        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_in, C_in, C
        self.sep_alpha = config.sep_alpha
        self.mul_att_out = config.mul_att_out
        if self.mul_att_out:
            self.attn_block = nn.ModuleList(
                [MultiHeadSelfAttention(C * 3, 12) for _ in range(self.n_layers)])
        self.cells = nn.ModuleList()
        for i in range(self.n_layers):
            print(C_pp, C_p, C_cur)
            cell = SearchCell(self.nodes, C_pp, C_p, C_cur)
            C_cur_out = C_cur * self.nodes
            C_pp, C_p = C_p, C_cur_out
            self.cells.append(cell)
        self.hidn2attn = config.hidn2attn
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(C_p * 2, self.n_classes)
        if self.all_return and not self.hidn2attn:
            self.fit_dense = nn.Linear(C_cur_out, 768)

    def forward(self, x, weights_normal):
        input_ids, mask, segment_ids, seq_lengths = x
        att_mask = mask.unsqueeze(1).unsqueeze(2)
        att_mask = (1.0 - att_mask) * -10000.0
        if self.return_node_out:
            student_node_out = []
        student_layer_out = []
        s0 = s1 = self.stem(input_ids, segment_ids)
        for cell_id, cell in enumerate(self.cells):
            if self.sep_alpha:
                s0, s1 = s1, cell(s0, s1,
                                weights_normal[cell_id*self.nodes:(cell_id + 1)*self.nodes])
            else:
                s0, s1 = s1, cell(s0, s1, weights_normal)
            if self.mul_att_out and self.all_return:
                s1_attention = self.attn_block[cell_id](s1.permute(0, 2, 1), att_mask)
                student_layer_out.append(s1_attention)
            elif self.all_return:
                if self.hidn2attn:
                    student_layer_out.append(s1.permute(0, 2, 1))
                else:
                    hidden_value = self.fit_dense(s1.permute(0, 2, 1))
                    student_layer_out.append(hidden_value)
                if self.return_node_out:
                    student_node_out.append(torch.chunk(s1.permute(0, 2, 1), 3, dim=-1))
        s1 = s1.permute(0, 2, 1)
        mask = mask.unsqueeze(-1).repeat(1, 1, s1.shape[-1])
        max_mid, _ = replace_masked(s1, mask, -1e7).max(dim=1)
        mean_mid = torch.sum(s1 * mask, dim=1) / torch.sum(mask, dim=1)

        out = torch.cat([max_mid, mean_mid], dim=-1)
        logits = self.linear(out)
        if self.return_node_out:
            return logits, student_layer_out, student_node_out
        if self.all_return:
            return logits, student_layer_out
        else:
            return logits


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, config, n_classes, output_mode):
        super().__init__()
        n_layers = config.layers
        self.nodes = config.nodes
        self.output_mode = output_mode
        self.n_classes = config.n_classes
        n_ops = len(gt.PRIMITIVES)
        self.alpha_normal = nn.ParameterList()
        self._alphas = []
        self.config = config

        self.sep_alpha = config.sep_alpha
        if self.sep_alpha:
            for _ in range(n_layers):
                for i in range(self.nodes):
                    self.alpha_normal.append(nn.Parameter(1e-3 * torch.ones(i + 2, n_ops)))
        else:
            for i in range(self.nodes):
                self.alpha_normal.append(nn.Parameter(1e-3 * torch.ones(i + 2, n_ops)))

        # setup alphas list
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))
        self.net = SearchCNN(config, n_classes)
        self.apply(init_weights)

    def forward(self,
                x,
                train=True,
                one_step=False,
                random_sample=False,
                freeze=False,
                alpha_only=False):
        if random_sample:
            _softmax = lambda xx: self._get_onehot_mask(xx) * nn.functional.softmax(xx, dim=-1)
        # elif not alpha_only and  random_sample:
        #     _softmax = lambda xx: self._get_onehot_mask(xx)
        elif alpha_only:
            _softmax = nn.Softmax(dim=-1)
        elif freeze:
            _softmax = nn.Softmax(dim=-1)
        elif one_step:
            _softmax = lambda xx: self._get_categ_mask(xx)
            # _softmax = lambda xx: nn.functional.gumbel_softmax(xx, tau=True, dim=-1)
        else:
            _softmax = nn.Softmax(dim=-1)
        weights_normal = [_softmax(alpha / self.config._temp) for alpha in self.alpha_normal]
        return self.net(x, weights_normal)

    def loss(self, X, ys):
        y = ys
        logits = self.forward(X)
        if self.config.use_emd:
            logits, _ = logits
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
        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alphas in self.alpha_normal:
            logger.info(F.softmax(alphas / self.config._temp, dim=-1).detach().cpu().numpy())
            # logger.info(alpha)
        logger.info("# Alpha - no softmax")
        for alphas in self.alpha_normal:
            logger.info(alphas.detach().cpu().numpy())

    def format_alphas(self):
        ds, ds_out = [], []
        for alphas in self.alpha_normal:
            soft_alphas = F.softmax(alphas / self.config._temp, dim=-1)
            for i in range(len(alphas)):
                ds.append({f"{x}": y for x, y in enumerate(alphas[i])})
                ds_out.append({f"{x}": y for x, y in enumerate(soft_alphas[i])})
        return ds, ds_out

    def generate_test_alpha_mask(self,):
        new_alpha = [torch.zeros_like(x) for x in self.alpha_normal]
        for index, edges in enumerate(self.alpha_normal):
            # edges: Tensor(n_edges, n_ops)
            edge_max, primitive_indices = torch.topk(edges[:, :-1], 1)  # ignore 'none'
            topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), 2)
            for edge_idx in topk_edge_indices:
                # print(edge_idx, primitive_indices[edge_idx][0])
                new_alpha[index][edge_idx][primitive_indices[edge_idx][0]] = 1
        return new_alpha

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = []
        concat = range(2, 2 + self.nodes)  # concat all intermediate nodes

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

    def _get_onehot_mask(self, log_alpha):
        uni = torch.ones_like(log_alpha)
        m = torch.distributions.one_hot_categorical.OneHotCategorical(uni)
        one_hot = m.sample()
        return one_hot

    def _get_categ_mask(self, log_alpha):
        # log_alpha 2d one_hot 2d
        u = torch.zeros_like(log_alpha).uniform_()
        softmax = torch.nn.Softmax(-1)
        one_hot = softmax((log_alpha + (-(-(u.log())).log())) / self.config._temp)
        return one_hot


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
        m.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0

        if (m.bidirectional):
            nn.init.xavier_uniform_(m.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(m.weight_hh_l0_reverse.data)
            nn.init.constant_(m.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(m.bias_hh_l0_reverse.data, 0.0)
            m.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0
