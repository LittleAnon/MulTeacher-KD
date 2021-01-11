import numpy as np
import torch
import torch.nn as nn
from pyemd import emd_with_flow



class Emd_Evaluator():
    def __init__(self, stu_layer_num, tea_layer_num, device, T=1, attn=False, weight_rate=1, add_softmax=True):
        self.s_weight = np.zeros(stu_layer_num)
        self.t_weight = np.zeros(tea_layer_num)
        self.s_layer_num = stu_layer_num
        self.t_layer_num = tea_layer_num
        self.d_method = nn.MSELoss()
        self.device = device
        self.Temperature = T
        self.add_softmax = add_softmax
        self.weight_rate = weight_rate

        self.init_weight()
        self.attn = attn
        if self.attn:
            self.s_att_weight = np.zeros(stu_layer_num)
            self.t_att_weight = np.zeros(tea_layer_num)
        # self.conv = nn.Conv1d(768, 768, 1).to(device)

    def init_weight(self):
        self.s_weight = np.ones(self.s_layer_num) / \
            self.s_layer_num * self.weight_rate
        self.t_weight = np.ones(self.t_layer_num) / \
            self.t_layer_num * self.weight_rate

    def assert_length(self, student_reps, teacher_reps):
        if len(student_reps) != self.s_weight:
            if student_reps == self.s_weight + 1:
                student_reps = student_reps[1:]
            else:
                raise ValueError("student layer output length error")
        if len(teacher_reps) == self.t_weight:
            if teacher_reps == self.t_weight + 1:
                teacher_reps = teacher_reps[1:]
            else:
                raise ValueError("student layer output length error")
        return student_reps, teacher_reps

    def build_distance_matrix(self, student_reps, teacher_reps):
        for i in range(self.s_layer_num):
            student_rep = student_reps[i]
            for j in range(self.t_layer_num):
                teacher_rep = teacher_reps[j + 1]
                # + torch.sum(torch.abs(student_rep - teacher_rep))
                tmp_loss = self.d_method(student_rep, teacher_rep)
                self.distance_matrix[i][j + self.s_layer_num] = self.distance_matrix[j +
                                                                                     self.s_layer_num][i] = tmp_loss

    def loss(self, student_reps, teacher_reps, return_distance=False):
        _s_weight = np.concatenate(
            (self.s_weight, np.zeros_like(self.t_weight)))
        _t_weight = np.concatenate(
            (np.zeros_like(self.s_weight), self.t_weight))
        totol_num = self.s_layer_num + self.t_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).to(self.device)
        for i in range(self.s_layer_num):
            student_rep = student_reps[i]
            for j in range(self.t_layer_num):
                teacher_rep = teacher_reps[j + 1]
                tmp_loss = self.d_method(student_rep, teacher_rep)
                distance_matrix[i][j + self.s_layer_num] = distance_matrix[j +
                                                                           self.s_layer_num][i] = tmp_loss
        _, trans_matrix = emd_with_flow(
            _s_weight, _t_weight, distance_matrix.detach().cpu().numpy().astype('float64'))
        d = torch.sum(torch.tensor(trans_matrix).to(
            self.device) * distance_matrix)
        if return_distance:
            return d, trans_matrix, distance_matrix
        else:
            return d

    def attn_loss(self, student_atts, teacher_atts, return_distance=False):
        if len(teacher_atts[0].shape) == 4 and len(student_atts[0].shape) == 3:
            teacher_atts = [torch.mean(x, dim=1) for x in teacher_atts]
        elif len(student_atts[0].shape) == 4 and len(teacher_atts[0].shape) == 4:
            if student_atts[0].shape[1] != teacher_atts[0].shape[1]:
                teacher_atts = [torch.mean(x, dim=1) for x in teacher_atts]
                student_atts = [torch.mean(x, dim=1) for x in student_atts]
        elif len(student_atts[0].shape) == 4 and len(teacher_atts[0].shape) == 3:
            student_atts = [torch.mean(x, dim=1) for x in student_atts]
        _s_weight = np.concatenate(
            (self.s_weight, np.zeros_like(self.t_weight)))
        _t_weight = np.concatenate(
            (np.zeros_like(self.s_weight), self.t_weight))
        totol_num = self.s_layer_num + self.t_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).to(self.device)
        for i in range(self.s_layer_num):
            student_att = student_atts[i]
            for j in range(self.t_layer_num):
                teacher_att = teacher_atts[j]
                tmp_loss = self.d_method(student_att, teacher_att)
                distance_matrix[i][j + self.s_layer_num] = distance_matrix[j +
                                                                           self.s_layer_num][i] = tmp_loss
        _, trans_matrix = emd_with_flow(
            _s_weight, _t_weight, distance_matrix.detach().cpu().numpy().astype('float64'))
        d = torch.sum(torch.tensor(trans_matrix).to(
            self.device) * distance_matrix)
        if return_distance:
            return d, trans_matrix, distance_matrix
        else:
            return d

    def update_weight(self, trans_matrix, distance_matrix):
        distance_matrix = distance_matrix.detach().cpu().numpy().astype('float64')

        trans = np.sum(trans_matrix * distance_matrix, -1)[:self.s_layer_num]
        tmp_s = np.divide(trans, self.s_weight, out=np.zeros_like(
            trans), where=self.s_weight != 0)
        weight_sum = np.ones_like(trans) * np.sum(tmp_s)
        self.s_weight = np.divide(
            weight_sum, tmp_s, out=np.zeros_like(weight_sum), where=tmp_s != 0)

        trans = np.sum(np.transpose(trans_matrix) *
                       distance_matrix, -1)[self.s_layer_num:]
        tmp_t = np.divide(trans, self.t_weight, out=np.zeros_like(
            trans), where=self.t_weight != 0)
        weight_sum = np.ones_like(trans) * np.sum(tmp_t)
        self.t_weight = np.divide(
            weight_sum, tmp_t, out=np.zeros_like(weight_sum), where=tmp_t != 0)
        self.s_weight = self.s_weight / np.sum(self.s_weight)
        self.t_weight = self.t_weight / np.sum(self.t_weight)
        if self.add_softmax:
            self.s_weight = self.s_weight / np.sum(self.s_weight)
            self.t_weight = self.t_weight / np.sum(self.t_weight)
            # self.s_weight = softmax(self.s_weight / self.Temperature)
            # self.t_weight = softmax(self.t_weight / self.Temperature)
        self.s_weight = self.s_weight * self.weight_rate
        self.t_weight = self.t_weight * self.weight_rate


def distillation_loss(y, labels, teacher_scores, output_mode, T=1, alpha=0.5, reduction_nll='mean', reduce_T=1):
    if output_mode == "classification":
        if teacher_scores is not None:
            student_likelihood = torch.nn.functional.log_softmax(y / T, dim=-1)
            # student_likelihood = torch.nn.functional.softmax(y / T, dim=-1)
            targets_prob = torch.nn.functional.softmax(
                teacher_scores / T, dim=-1)
            d_loss = (- targets_prob * student_likelihood).mean() * \
                T * T / reduce_T
        else:
            d_loss = 0.0
        crit = nn.CrossEntropyLoss()
        nll_loss = crit(y, labels)
    elif output_mode == "regression":
        loss_mse = nn.MSELoss()
        if teacher_scores is not None:
            d_loss = loss_mse(y.view(-1), teacher_scores.view(-1))
        else:
            d_loss = 0.0
        nll_loss = loss_mse(y.view(-1), labels.view(-1))
    else:
        raise ValueError(
            "output_mode must in \"classification\", \"regression\"")
    tol_loss = alpha * d_loss + (1.0 - alpha) * nll_loss
    return tol_loss, d_loss, nll_loss
