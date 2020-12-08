""" CNN for network augmentation """
import torch
import torch.nn as nn
from models.augment_cells import AugmentCell
from models import blocks
import json
import sys
from models.search_cnn import MultiHeadSelfAttention
def replace_masked(tensor, mask, value):
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """

    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        # assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool1d(5, stride=input_size - 5, padding=0, count_include_pad=False),  # 2x2 out
            nn.Conv1d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(128, 512, kernel_size=2, bias=False), # 1x1 out
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))
        self.linear = nn.Linear(128, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits


class AugmentCNN(nn.Module):
    """ Augmented CNN model """

    def __init__(self, config,n_classes, output_mode, auxiliary):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        C_in = config.hidden_size
        self.C = config.init_channels
        self.n_classes = n_classes
        self.n_layers = config.layers
        self.genotype = config.genotype
        self.output_mode = output_mode
        # aux head position
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1
        # self.aux_pos = -1
        stem_multiplier = 1
        C_cur = stem_multiplier * self.C
        if config.teacher_type == 'gpt2':
            self.stem = blocks.Gpt2Embeddings(config.gpt_config)
        elif config.teacher_type == 'bert':
            self.stem = blocks.BertEmbeddings(config.bert_config)
        elif config.teacher_type == 'roberta':
            self.stem = blocks.RobertaEmbeddings(config.roberta_config)
        self.hidn2attn = config.hidn2attn
        self.attn_cells = nn.ModuleList()
        # self.self_attention_output = config.self_attention_output
        C_pp, C_p, C_cur = C_in, C_in, self.C

        self.cells = nn.ModuleList()
        for i in range(self.n_layers):
            cell = AugmentCell(self.genotype, C_pp, C_p, C_cur, False, False)
            self.cells.append(cell)
            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out
            # if self.self_attention_output:
            #     self.attn_cells.append(MultiHeadSelfAttention(C_cur_out, 6))


        self.gap = nn.AdaptiveMaxPool1d(1)
        # self.linear1 = nn.Linear(C_p, C_p)
        # self.linear2 = nn.Linear(C_p, n_classes)
        self.linear = nn.Linear(C_cur_out * 2, self.n_classes)

        self.use_kd = config.use_kd
        self.all_return = config.use_emd
        if self.all_return:
            self.fit_dense = nn.Linear(C_cur_out, 768)

    def forward(self, x):
        input_ids, mask, segment_ids, seq_lengths = x
        
        student_layer_out = []
        s0 = s1 = self.stem(input_ids, segment_ids)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
            # if self.self_attention_output:
            #     s1 = self.attn_cells[cell_id](s1.permute(0, 2, 1), att_mask).permute(0, 2, 1)
            if self.all_return:
                if self.hidn2attn:
                    student_layer_out.append(s1.permute(0, 2, 1))
                else:
                    hidden_value = self.fit_dense(s1.permute(0,2,1))
                    student_layer_out.append(hidden_value)
        s1 = s1.permute(0,2,1)
        mask = mask.unsqueeze(-1).repeat(1, 1, s1.shape[-1])
        max_mid, _ = replace_masked(s1, mask, -1e7).max(dim=1)
        mean_mid = torch.sum(s1 * mask, dim=1) / torch.sum(mask, dim=1)

        out = torch.cat([max_mid, mean_mid], dim=-1)
        logits = self.linear(out)
        if self.all_return:
            return logits, student_layer_out
        else:
            return logits

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, blocks.DropPath_):
                module.p = p
