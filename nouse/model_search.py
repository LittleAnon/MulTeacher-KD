""" Search cell """
import os
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
import utils
from utils import bert_batch_split, get_acc_from_pred

from models.search_cnn import SearchCNNController
from modeling import TinyBertForSequenceClassification
from nouse.architect import Architect
from config import SearchConfig

from kdTool import Emd_Evaluator, softmax
from dist_util_torch import init_gpu_params, set_seed, FileLogger
from torch.nn.functional import softmax
from math import sqrt
acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte", 'books']
corr_tasks = ["sts-b"]
mcc_tasks = ["cola"]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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

class SearchModel():

    def __init__(self):
        self.config = SearchConfig()
        self.writer = None
        if self.config.tb_dir != "":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.config.tb_dir, flush_secs=20)
        init_gpu_params(self.config)
        set_seed(self.config)
        self.logger = FileLogger('./log', self.config.is_master, self.config.is_master)
        self.load_data()

        self.model = SearchCNNController(self.config, self.n_classes, self.output_mode)
        self.load_model()
        self.init_kd_component()

        if self.config.n_gpu > 0:
            self.model.to(f"cuda:{self.config.local_rank}")
        if self.config.n_gpu > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.config.local_rank], find_unused_parameters=True)
        self.model_to_print = self.model if self.config.multi_gpu is False else self.model.module
        self.architect = Architect(self.model, self.teacher_model, self.config, self.emd_tool)
        mb_params = utils.param_size(self.model)
        self.logger.info("Model size = {:.3f} MB".format(mb_params))

        self.init_optim()

    def init_kd_component(self):
        self.teacher_model, self.emd_tool = None, None
        if self.config.use_kd:
            self.teacher_model = TinyBertForSequenceClassification.from_pretrained(
                self.config.teacher_model, num_labels=self.n_classes)
            self.teacher_model = self.teacher_model.to(device)
            self.teacher_model.eval()
            if self.config.use_emd:
                att_emd = self.config.t_att_type != "" or self.config.s_att_type != ""
                self.emd_tool = Emd_Evaluator(self.config.layers, 12, device, attn=att_emd, weight_rate=self.config.weight_rate, add_softmax=self.config.add_softmax)

    def load_data(self):
        # set seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.benchmark = True
        self.task_name = self.config.datasets
        self.train_loader, self.arch_loader, self.eval_loader, self.output_mode, self.n_classes, self.config = utils.load_glue_dataset(
            self.config)
        self.logger.info(f"train_loader length {len(self.train_loader)}")

    def init_optim(self):
        no_decay = ["bias", "LayerNorm.weight"]
        self.w_optim = torch.optim.SGD([
            p for n, p in self.model.named_parameters()
            if not any(nd in n for nd in no_decay) and p.requires_grad and 'alpha' not in n
        ],
                                       self.config.w_lr,
                                       momentum=self.config.w_momentum,
                                       weight_decay=self.config.w_weight_decay)
        if self.config.alpha_optim.lower() == 'adam':
            self.alpha_optim = torch.optim.Adam([
                p for n, p in self.model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad and 'alpha' in n
            ],
                                                self.config.alpha_lr,
                                                weight_decay=self.config.alpha_weight_decay)
        elif self.config.alpha_optim.lower() == 'sgd':
            self.alpha_optim = torch.optim.SGD([
                p for n, p in self.model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad and 'alpha' in n
            ],
                                               self.config.alpha_lr,
                                               weight_decay=self.config.alpha_weight_decay)
        else:
            raise NotImplementedError("no such optimizer")

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.w_optim, self.config.epochs, eta_min=self.config.w_lr_min)

    def load_model(self):
        if self.config.restore != "":
            old_params_dict = dict()
            for k, v in self.model.named_parameters():
                old_params_dict[k] = v
            self.model.load_state_dict(torch.load(self.config.restore), strict=False)
            for k, v in self.model.named_parameters():
                if torch.sum(v) != torch.sum(old_params_dict[k]):
                    print(k + " not restore")
            del old_params_dict
        else:
            utils.load_embedding_weight(self.model,
                                        'teacher_utils/bert_base_uncased/pytorch_model.bin', True)

    def save_checkpoint(self, dump_path, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        if not self.is_master:
            return
        mdl_to_save = self.model.module if hasattr(self.model, "module") else self.model
        state_dict = mdl_to_save.state_dict()
        state_dict = {k:v for k, v in state_dict.items() if 'alpha' in k}
        torch.save(state_dict, os.path.join(dump_path, checkpoint_name))

    def main(self):
        # training loop
        best_top1 = 0.
        is_best = False
        best_genotype = ""
        for epoch in range(self.config.epochs):
            lr = self.lr_scheduler.get_last_lr()[-1]

            self.logger.info("Epoch {}".format(epoch))
            self.logger.info("Learning Rate {}".format(lr))

            self.train_sep(lr, epoch)
            self.model_to_print.print_alphas(self.logger)
            self.lr_scheduler.step()

            self.logger.info('valid')
            cur_step = (epoch + 1) * len(self.train_loader)
            top1 = self.validate(epoch, cur_step, "val")

            # genotype
            genotypes = self.model_to_print.genotype()
            # if config.is_master:
            self.logger.info("========genotype========\n" + str(genotypes))
            # # save
            is_best = best_top1 <= top1
            if is_best:
                best_top1 = top1
                best_genotype = genotypes
                if self.config.save_supernet != "" and self.config.alpha_ep != -1 and epoch < self.config.alpha_ep:
                    torch.save(self.model_to_print.state_dict(), self.config.save_supernet)
            # if epoch == self.config.alpha_ep and not is_best:
            #     torch.load(self.config.save_supernet)
            self.logger.info("Present best Prec@1 = {:.4%}".format(best_top1))
        if self.config.tb_dir != "":
            self.writer.close()
        self.logger.info("Final best Prec@1 = " + str(best_top1))
        self.logger.info("Best Genotype = " + str(best_genotype))

    def train(self, lr, epoch):
        top1 = utils.AverageMeter()
        losses = utils.AverageMeter()

        self.model.train()
        total_num_step = len(self.train_loader)
        cur_step = epoch * len(self.train_loader)
        valid_iter = iter(self.arch_loader)

        random_search = epoch < self.config.random_sample_epoch
        freeze = epoch < self.config.freeze_alpha
        update_alpha = (not freeze and not random_search) or self.config.alpha_only

        if self.config.multi_gpu:
            torch.distributed.barrier()

        for step, data in enumerate(self.train_loader):
            trn_X, trn_y = bert_batch_split(data, self.config.local_rank)

            try:
                v_data = next(valid_iter)
            except StopIteration:
                valid_iter = iter(self.arch_loader)
                v_data = next(valid_iter)

            val_X, val_y = None, None
            if self.config.one_step is not True:
                val_X, val_y = bert_batch_split(v_data, self.config.local_rank)

            out_att = self.config.t_att_type != ""
            trn_t, val_t = None, None
            if self.config.use_kd:
                with torch.no_grad():
                    teacher_out = self.teacher_model(trn_X, attention_out=out_att)
                    if out_att:
                        teacher_logits, teacher_attns, teacher_reps = teacher_out
                        teacher_attns = [softmax(x, dim=-1) for x in teacher_attns]
                    else:
                        teacher_logits, teacher_reps = teacher_out
                    trn_t = (teacher_logits, teacher_reps)

                    if self.config.one_step is not True:
                        v_teacher_logits, v_teacher_reps = self.teacher_model(val_X)
                        val_t = (v_teacher_logits, v_teacher_reps)

            N = trn_X[0].size(0)

            self.alpha_optim.zero_grad()
            if not self.config.one_step and update_alpha:
                self.architect.unrolled_backward(trn_X, trn_y, val_X, val_y, trn_t, val_t, lr,
                                                 self.w_optim)
                self.alpha_optim.step()
            if self.config.multi_gpu:
                torch.distributed.barrier()
            self.w_optim.zero_grad()

            logits = self.model(
                trn_X,
                random_sample=random_search,
                one_step=self.config.one_step,
                freeze=freeze,
                alpha_only=self.config.alpha_only)

            if self.config.use_emd:
                if self.config.s_att_type == "node":
                    logits, s_layer_out, node_out = logits
                else:
                    logits, s_layer_out = logits
                # if self.config.t_att_type != "":
                #     if self.config.s_att_type == "hid":
                #         s_atts = convert_to_attn(s_layer_out, trn_X[1])  # [N, D, D]
                #     elif self.config.s_att_type == "cut":
                #         s_layer_out_cut = []
                #         for one_layer_out in s_layer_out:
                #             s_layer_out_cut.append(torch.chunk(one_layer_out, 12, dim=-1))
                #         s_atts = convert_to_attn(s_layer_out_cut, trn_X[1]) # [N, H, D, D]
                #     elif self.config.s_att_type == "node":
                #         s_atts = convert_to_attn(node_out, trn_X[1]) # [N, H, D, D]
                #     else:
                #         raise NotImplementedError
            # if self.config.use_kd:
            #     kd_loss, _, _ = distillation_loss(
            #         logits, trn_y, teacher_logits, self.output_mode, alpha=self.config.kd_alpha)
            #     rep_loss, att_loss = 0.0, 0.0
                
            #     if self.config.use_emd:
            #         if self.config.mul_att_out:
            #             s_layer_out = [softmax(x, dim=-1) for x in s_layer_out]
            #             rep_loss, flow, distance = self.emd_tool.attn_loss(
            #             s_layer_out, teacher_attns, return_distance=True)
            #         else:
            #             if self.config.hidn2attn:
            #                 s_layer_out = convert_to_attn(s_layer_out, trn_X[1])
            #                 teacher_reps = convert_to_attn(teacher_reps, trn_X[1])
            #             rep_loss, flow, distance = self.emd_tool.loss(
            #                 s_layer_out, teacher_reps, return_distance=True)
            #             if self.config.t_att_type != "" or self.config.s_att_type != "":
            #                 att_loss, flow, distance = self.emd_tool.attn_loss(
            #                 s_atts, teacher_attns, return_distance=True)
            #             if self.config.update_emd:
            #                 self.emd_tool.update_weight(flow, distance)
            #     loss = kd_loss * self.config.emd_only + (rep_loss + att_loss * 100)* self.config.emd_rate
            # else:
            loss = self.model_to_print.crit(logits, trn_y)

            l1_loss = 0

            loss.backward()

            no_decay = ["bias", "LayerNorm.weight"]
            # gradient clipping
            # if not self.config.alpha_only:
            clip = clip_grad_norm_(self.model_to_print.weights(), self.config.w_grad_clip)
            self.w_optim.step()
            # if self.config.one_step and update_alpha:
            #     self.alpha_optim.step()
            if self.config.tb_dir != "":
                ds, ds2 = self.model.format_alphas()
                for layer_index, dsi in enumerate(ds):
                    self.writer.add_scalars(f'layer-{layer_index}-alpha', dsi, global_step=cur_step)
                for layer_index, dsi in enumerate(ds2):
                    self.writer.add_scalars(
                        f'layer-{layer_index}-softmax_alpha', dsi, global_step=cur_step)
                self.writer.add_scalar('loss', loss, global_step=cur_step)
                self.writer.add_scalar("EMD", rep_loss, global_step=cur_step)
                self.writer.add_scalar("l1 loss", l1_loss, global_step=cur_step)

            preds = logits.detach().cpu().numpy()
            result, train_acc = get_acc_from_pred(self.output_mode, self.task_name, preds,
                                            trn_y.detach().cpu().numpy())

            losses.update(loss.item(), N)
            top1.update(train_acc, N)
            # model.print_alphas(logger)

            if self.config.eval_during_train:
                if step % self.config.print_freq == 0 or step == total_num_step - 1:
                    self.validate(epoch, cur_step, mode="train_dev")
            if step % self.config.print_freq == 0 or step == total_num_step - 1:
                self.logger.info(
                    "Train: , [{:2d}/{}] Step {:03d}/{:03d} Loss {:.3f}, Prec@(1,5) {top1.avg:.1%}"
                    .format(
                        epoch + 1,
                        self.config.epochs,
                        step,
                        total_num_step - 1,
                        losses.avg,
                        top1=top1))
            cur_step += 1
        self.logger.info("{:.4%}".format(top1.avg))

    def train_sep(self, lr, epoch):
        top1 = utils.AverageMeter()
        losses = utils.AverageMeter()
        self.model.train()
        total_num_step = len(self.train_loader)
        cur_step = epoch * len(self.train_loader)
        valid_iter = iter(self.arch_loader)
        for step, data in enumerate(self.train_loader):
            trn_X, trn_y = bert_batch_split(data, self.config.local_rank)
            try:
                v_data = next(valid_iter)
            except StopIteration:
                valid_iter = iter(self.arch_loader)
                v_data = next(valid_iter)
            val_X, val_y = None, None
            if self.config.one_step is not True:
                val_X, val_y = bert_batch_split(v_data, self.config.local_rank)

            with torch.no_grad():
                teacher_logits, teacher_reps = self.teacher_model(trn_X, attention_out=False)
                # teacher_logits, teacher_attns, teacher_reps = teacher_out
                # # teacher_attns = [softmax(x, dim=-1) for x in teacher_attns]
                # trn_t = (teacher_logits, teacher_reps)
            N = trn_X[0].size(0)

            self.alpha_optim.zero_grad()
            self.w_optim.zero_grad()
            logits, s_layer_out = self.model(trn_X)
            if epoch % self.config.alpha_ep != 0:
                if self.config.hidn2attn:
                    s_layer_out = convert_to_attn(s_layer_out, trn_X[1])
                    teacher_reps = convert_to_attn(teacher_reps, trn_X[1])
                rep_loss, flow, distance = self.emd_tool.loss(
                    s_layer_out, teacher_reps, return_distance=True)
                if self.config.update_emd:
                    self.emd_tool.update_weight(flow, distance)
                loss = rep_loss * self.config.emd_rate + self.model_to_print.crit(logits, trn_y) * self.config.alpha_ac_rate
                loss.backward()
                self.alpha_optim.step()
            elif epoch % self.config.alpha_ep == 0:
                # if self.config.hidn2attn:
                #     s_layer_out = convert_to_attn(s_layer_out, trn_X[1])
                #     teacher_reps = convert_to_attn(teacher_reps, trn_X[1])
                # rep_loss, flow, distance = self.emd_tool.loss(
                #     s_layer_out, teacher_reps, return_distance=True)
                # if self.config.update_emd:
                #     self.emd_tool.update_weight(flow, distance)
                # loss = rep_loss * self.config.emd_rate + self.model_to_print.crit(logits, trn_y)
                loss = self.model_to_print.crit(logits, trn_y)
                loss.backward()
                # gradient clipping
                clip = clip_grad_norm_(self.model_to_print.weights(), self.config.w_grad_clip)
                self.w_optim.step()

            preds = logits.detach().cpu().numpy()
            result, train_acc = get_acc_from_pred(self.output_mode, self.task_name, preds,
                                            trn_y.detach().cpu().numpy())

            losses.update(loss.item(), N)
            top1.update(train_acc, N)

            if self.config.eval_during_train:
                if step % self.config.print_freq == 0 or step == total_num_step - 1:
                    self.validate(epoch, cur_step, mode="train_dev")
            if step % self.config.print_freq == 0 or step == total_num_step - 1:
                self.logger.info(
                    "Train: , [{:2d}/{}] Step {:03d}/{:03d} Loss {:.3f}, Prec@(1,5) {top1.avg:.1%}"
                    .format(
                        epoch + 1,
                        self.config.epochs,
                        step,
                        total_num_step - 1,
                        losses.avg,
                        top1=top1))
                if epoch % self.config.alpha_ep != 0 and self.config.update_emd:
                    self.logger.info("s weight:{}".format(self.emd_tool.s_weight))
                    self.logger.info("t weight:{}".format(self.emd_tool.t_weight))
            cur_step += 1
        self.logger.info("{:.4%}".format(top1.avg))

    def train_sep_dev(self, lr, epoch):
        top1 = utils.AverageMeter()
        losses = utils.AverageMeter()
        self.model.train()
        total_num_step = len(self.train_loader)
        cur_step = epoch * len(self.train_loader)
        current_loader = self.arch_loader if epoch % 2 == 1 else self.train_loader
        for step, data in enumerate(current_loader):
            trn_X, trn_y = bert_batch_split(data, self.config.local_rank)

            with torch.no_grad():
                teacher_logits, teacher_reps = self.teacher_model(trn_X, attention_out=False)
                # teacher_logits, teacher_attns, teacher_reps = teacher_out
                # # teacher_attns = [softmax(x, dim=-1) for x in teacher_attns]
                # trn_t = (teacher_logits, teacher_reps)
            N = trn_X[0].size(0)

            self.alpha_optim.zero_grad()
            self.w_optim.zero_grad()
            logits, s_layer_out = self.model(trn_X)
            if epoch >= self.config.alpha_ep:
                if self.config.hidn2attn:
                    s_layer_out = convert_to_attn(s_layer_out, trn_X[1])
                    teacher_reps = convert_to_attn(teacher_reps, trn_X[1])
                rep_loss, flow, distance = self.emd_tool.loss(
                    s_layer_out, teacher_reps, return_distance=True)
                if self.config.update_emd:
                    self.emd_tool.update_weight(flow, distance)
                loss = rep_loss * self.config.emd_rate + self.model_to_print.crit(logits, trn_y) * self.config.alpha_ac_rate
                loss.backward()
                self.alpha_optim.step()
            elif epoch < self.config.alpha_ep:
                # if self.config.hidn2attn:
                #     s_layer_out = convert_to_attn(s_layer_out, trn_X[1])
                #     teacher_reps = convert_to_attn(teacher_reps, trn_X[1])
                # rep_loss, flow, distance = self.emd_tool.loss(
                #     s_layer_out, teacher_reps, return_distance=True)
                # if self.config.update_emd:
                #     self.emd_tool.update_weight(flow, distance)
                # loss = rep_loss * self.config.emd_rate + self.model_to_print.crit(logits, trn_y)
                loss = self.model_to_print.crit(logits, trn_y)
                loss.backward()
                # gradient clipping
                clip = clip_grad_norm_(self.model_to_print.weights(), self.config.w_grad_clip)
                self.w_optim.step()

            preds = logits.detach().cpu().numpy()
            result, train_acc = get_acc_from_pred(self.output_mode, self.task_name, preds,
                                            trn_y.detach().cpu().numpy())

            losses.update(loss.item(), N)
            top1.update(train_acc, N)

            if self.config.eval_during_train:
                if step % self.config.print_freq == 0 or step == total_num_step - 1:
                    self.validate(epoch, cur_step, mode="train_dev")
            if step % self.config.print_freq == 0 or step == total_num_step - 1:
                self.logger.info(
                    "Train: , [{:2d}/{}] Step {:03d}/{:03d} Loss {:.3f}, Prec@(1,5) {top1.avg:.1%}"
                    .format(
                        epoch + 1,
                        self.config.epochs,
                        step,
                        total_num_step - 1,
                        losses.avg,
                        top1=top1))
            cur_step += 1
        self.logger.info("{:.4%}".format(top1.avg))

    def validate(self, epoch, cur_step, mode="dev"):
        # eval_labels = eval_labels.detach().cpu().numpy()
        eval_labels = []
        preds = []
        self.model.eval()

        total_loss, total_emd_loss = 0, 0

        with torch.no_grad():
            for step, data in enumerate(self.eval_loader):
                X, y = bert_batch_split(data, self.config.local_rank)
                N = X[0].size(0)
                logits = self.model(X, train=False)
                rep_loss = 0
                if self.config.use_emd:
                    logits, s_layer_out = logits
                    _, teacher_attns, teacher_reps = self.teacher_model(X, attention_out=True)
                    if self.config.mul_att_out:
                        rep_loss, flow, distance = self.emd_tool.attn_loss(
                        s_layer_out, teacher_attns, return_distance=True)
                    else:
                        if self.config.hidn2attn:
                            s_layer_out = convert_to_attn(s_layer_out, X[1])
                            teacher_reps = convert_to_attn(teacher_reps, X[1])
                        rep_loss, flow, distance = self.emd_tool.loss(
                            s_layer_out, teacher_reps, return_distance=True)
                    total_emd_loss += rep_loss.item()
                loss = self.model_to_print.crit(logits, y)
                total_loss += loss.item()
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                eval_labels.extend(y.detach().cpu().numpy())
            preds = preds[0]
            result, acc = get_acc_from_pred(self.output_mode, self.task_name, preds, eval_labels)

        self.logger.info(mode + ": [{:2d}/{}] Final Prec@1 {} Loss {}, EMD loss: {}".format(
            epoch + 1, self.config.epochs, result, total_loss, total_emd_loss))

        return acc

if __name__ == "__main__":
    architecture = SearchModel()
    architecture.main()
