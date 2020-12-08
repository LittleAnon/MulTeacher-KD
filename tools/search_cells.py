""" CNN cell for architecture search """
import torch
import torch.nn as nn
from models import blocks


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        
        # self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        # self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)
        self.preproc = blocks.Initialized_Conv1d(C_p,C,kernel_size=1)
        # self.preproc1 = blocks.Initialized_Conv1d(C_p,C,kernel_size=1)
        
        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(1+i): # include 1 input nodes
                # reduction should be used only for input node
                stride = 1
                op = blocks.MixedOp(C, stride)
                self.dag[i].append(op)

    def set_grad_mode(self, choice):
        connections, options = choice
        for i in range(self.n_nodes):
            for j in range(2+i): # include 1 input nodes
                if connections[i]==j:
                    self.dag[i][j].set_active(options[i])
                else:
                    self.dag[i][j].set_active(-1)

    def forward(self, s0, s1, w_dag):
        # s1 = self.preproc(s1)
        # states = [s0, s1]
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            # print("s_cur shape",s_cur.size())
            states.append(s_cur)

        s_out = torch.cat(states[1:], dim=1)
        return s_out

