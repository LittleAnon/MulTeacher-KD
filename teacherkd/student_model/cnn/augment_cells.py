""" CNN cell for network augmentation """
import torch
import torch.nn as nn
from teacherkd.student_model.cnn import blocks
import teacherkd.genotypes as gt


from teacherkd.student_model.embeddings import BertLayerNorm


def dag_to_task_weight(gene, task_genes):
    task_weight = {i: [] for i in range(len(task_genes))}
    for task_id, task_gene in enumerate(task_genes):
        for layer_num, task_layer in enumerate(task_gene):
            print(task_layer)
            layer_cells = [gene[layer_num].index(x) for x in task_layer]
            layer_weight = [0] * len(gene[layer_num])
            for i in layer_cells:
                layer_weight[i] = 1
            task_weight[task_id].append(layer_weight)
    return task_weight

class AugmentCell(nn.Module):
    """ Cell for augmentation
    Each edge is discrete.
    """
    def __init__(self, genotype, C_pp, C_p, C, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = len(genotype.normal)


        self.preproc1 = blocks.Initialized_Conv1d(C_p, C, kernel_size=1)
        self.preproc0 = blocks.Initialized_Conv1d(C_pp, C, kernel_size=1)

        self.concat_conv1 = blocks.Initialized_Conv1d(C_p, C*3, kernel_size=1)

        gene = genotype.normal
        self.concat = genotype.normal_concat

        self.dag = gt.to_dag(C, gene, reduction)
        self.batch_norm = BertLayerNorm(C*3)

    def forward(self, s0, s1):
        s1_out = self.concat_conv1(s1)

        # s0 = self.preproc0(s0)
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)
        s_out = torch.cat([states[i] for i in self.concat], dim=1)
        #s_out = self.batch_norm((s_out).permute(0, 2, 1)).permute(0, 2, 1)
        # s_out = ((s_out + s1_out).permute(0, 2, 1)).permute(0, 2, 1)

        return s_out
