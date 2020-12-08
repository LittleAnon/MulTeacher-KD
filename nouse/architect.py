""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
import numpy as np
from kdTool import distillation_loss
import torch.distributed.autograd
from math import sqrt

# import torchviz
def convert_to_attn(hidns, mask):
    if type(hidns[0]) is not tuple:
        hdim = hidns[0].shape[-1]
        attns = [torch.matmul(x, x.transpose(2, 1)) / sqrt(hdim) for x in hidns]
        mask = mask.unsqueeze(1)
        mask = (1.0 - mask) * -10000.0
        attns = [torch.nn.functional.softmax(x + mask, dim=-1) for x in attns]
    else:
        hidns = [torch.stack(x, dim=1) for x in hidns]
        hdim = hidns[0][0].shape[-1]
        attns = [torch.matmul(x, x.transpose(-1, -2)) / sqrt(hdim) for x in hidns]
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = (1.0 - mask) * -10000.0
        attns = [torch.nn.functional.softmax(x + mask, dim=-1) for x in attns]
    return attns

class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, t_net, config, emd_tool=None):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.t_net = t_net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = config.w_momentum
        self.w_weight_decay = config.w_weight_decay
        self.use_kd = config.use_kd
        self.use_emd = config.use_emd
        self.emd_tool = emd_tool
        self.config = config
        self.emd_only = config.emd_only
    def virtual_step(self, trn_X, trn_y, tt_logits, tt_hid, xi, w_optim, mask):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        logits = self.net(trn_X, False, False)
        if self.use_emd:
            if self.config.s_att_type == "node":
                logits, student_reps, node_out = logits
            else:
                logits, student_reps = logits

        # if self.use_kd:
        #     rep_loss = 0
        #     if self.use_emd:
        #         if self.config.hidn2attn:
        #             student_reps = convert_to_attn(student_reps, mask)
        #             tt_hid = convert_to_attn(tt_hid, mask)
        #         rep_loss = self.emd_tool.loss(student_reps, tt_hid)
        #     kd_loss, _, _ = distillation_loss(logits, trn_y, tt_logits, self.net.output_mode, alpha=self.config.kd_alpha)
        #     loss = rep_loss * self.config.emd_rate #+  kd_loss * self.emd_only
        # else:
        loss = self.net.crit(logits, trn_y)
        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights(), allow_unused=True)
        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for (n, w), vw, g in zip(self.net.named_weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))
                
            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, trn_t, val_t, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        mask = val_X[1]
        if self.use_kd and self.use_emd:
            tt_logits, tt_hid = trn_t
            vt_logits, vt_hid = val_t
        else:
            tt_logits, tt_hid, vt_logits, vt_hid = None, None, None, None
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, tt_logits, tt_hid, xi, w_optim, mask)
        
        logits = self.v_net(val_X)
        if self.use_emd:
            if self.config.s_att_type == "node":
                logits, student_reps, node_out = logits
            else:
                logits, student_reps = logits
        if self.use_kd:
            rep_loss = 0
            if self.use_emd:
                if self.config.hidn2attn:
                    student_reps = convert_to_attn(student_reps, mask)
                    vt_hid = convert_to_attn(vt_hid, mask)
                rep_loss = self.emd_tool.loss(student_reps, vt_hid)
            kd_loss, _, _ = distillation_loss(logits, val_y, vt_logits, self.net.output_mode, alpha=self.config.kd_alpha)
            # loss = kd_loss * self.emd_only + rep_loss * self.config.emd_rate
            loss = rep_loss * self.config.emd_rate + self.v_net.crit(logits, val_y) * self.config.alpha_ac_rate
        else:
            loss = self.v_net.crit(logits, val_y)

        # loss = self.v_net.loss(val_Xs, val_ys)
        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights, allow_unused=True)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]
        hessian = self.compute_hessian(dw, trn_X, trn_y, tt_logits, tt_hid, mask)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h

    def compute_hessian(self, dw, trn_X, trn_y, tt_logits, tt_hid, mask):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
    
        logits = self.net(trn_X, False, False)
        if self.use_emd:
            if self.config.s_att_type == "node":
                logits, student_reps, node_out = logits
            else:
                logits, student_reps = logits
        # if self.use_kd:
        #     rep_loss = 0
        #     if self.use_emd:
        #         if self.config.hidn2attn:
        #             student_reps = convert_to_attn(student_reps, mask)
        #             tt_hid = convert_to_attn(tt_hid, mask)
        #         rep_loss = self.emd_tool.loss(student_reps, tt_hid)
        #     kd_loss, _, _ = distillation_loss(logits, trn_y, tt_logits, self.net.output_mode, alpha=self.config.kd_alpha)
        #     # loss = kd_loss * self.emd_only + rep_loss * self.config.emd_rate
        #     loss = rep_loss * self.config.emd_rate
        # else:
        loss = self.v_net.crit(logits, trn_y)

        dalpha_pos = torch.autograd.grad(loss, self.net.alphas(), allow_unused=True) # dalpha { L_trn(w+) }
        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        logits = self.net(trn_X, False, False)
        if self.use_emd:
            if self.config.s_att_type == "node":
                logits, student_reps, node_out = logits
            else:
                logits, student_reps = logits
        # if self.use_kd:
        #     rep_loss = 0
        #     if self.use_emd:
        #         if self.config.hidn2attn:
        #             student_reps = convert_to_attn(student_reps, mask)
        #             tt_hid = convert_to_attn(tt_hid, mask)
        #         rep_loss = self.emd_tool.loss(student_reps, tt_hid)
        #     kd_loss, _, _ = distillation_loss(logits, trn_y, tt_logits, self.net.output_mode, alpha=self.config.kd_alpha)
        #     # loss = kd_loss * self.emd_only + rep_loss * self.config.emd_rate
        #     loss = rep_loss * self.config.emd_rate
        # else:
        loss = self.v_net.crit(logits, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas(), allow_unused=True)  # dalpha { L_trn(w-) }
        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian