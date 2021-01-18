# coding=utf-8
""" Training augmented model """
from sklearn.metrics import recall_score
from transformers import RobertaForSequenceClassification, GPT2ForSequenceClassification, BertForSequenceClassification
import os
from math import sqrt
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn

import teacherkd.utils as utils
# from transformers.data.metrics import glue_compute_metrics as compute_metrics
from teacherkd.metric_utils import compute_metrics
from teacherkd.config import AugmentConfig
from teacherkd.utils import init_gpu_params
from teacherkd.file_logger import FileLogger
from teacherkd.kdTool import Emd_Evaluator, distillation_loss
from teacherkd.student_model.cnn.augment_cnn import AugmentCNN
from teacherkd.student_model.transform.augment_transform import AugmentTransform
from teacherkd.report.reporter import generate_report_by_metrics
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

task_metric_dict = {
            "cola":'mcc',
            "sst-2":'acc',
            "mnli":'acc',
            "mnli-mm":'acc',
            "qnli":'acc',
            "rte":'acc',
            "books":'acc',
            "mrpc":'f1',
            "qqp":'f1',
            "sts-b":'corr',
        }

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def convert_to_attn(hidns, mask):
    if type(hidns[0]) is not tuple:
        hdim = hidns[0].shape[-1]
        attns = [torch.matmul(x, x.transpose(2, 1)) / sqrt(hdim)
                 for x in hidns]
        mask = mask.unsqueeze(1)
        mask = (1.0 - mask) * -10000.0
        attns = [torch.nn.functional.softmax(x + mask, dim=-1) for x in attns]
    else:
        hidns = [torch.stack(x, dim=1) for x in hidns]
        hdim = hidns[0][0].shape[-1]
        attns = [torch.matmul(x, x.transpose(-1, -2)) /
                 sqrt(hdim) for x in hidns]
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = (1.0 - mask) * -10000.0
        attns = [torch.nn.functional.softmax(x + mask, dim=-1) for x in attns]
    return attns


def main():
    config = AugmentConfig()
    init_gpu_params(config)
    logger = FileLogger('./log', config.is_master, config.is_master)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    ############# LOADING DATA /START ###############
    task_name = config.datasets
    train_dataloader, _, eval_dataloader, output_mode, n_classes, config = utils.load_glue_dataset(
        config)
    logger.info(f"train_loader length {len(train_dataloader)}")

    ############# LOADING DATA /END ###############

    ############### BUILDING MODEL /START ###############
    #cnn, transform
    if config.student_type == 'cnn':
        model = AugmentCNN(config, n_classes, output_mode, auxiliary=False)
    elif config.student_type == 'transform':
        model = AugmentTransform(config, n_classes ,output_mode)
    else:
        raise ValueError("unknown student model type.")
    pre_d, new_d = {}, {}
    for k, v in model.named_parameters():
        pre_d[k] = torch.sum(v)
    if config.teacher_type == 'gpt2':
        utils.load_gpt2_embedding_weight(model,
                                         config.teacher_model)
    elif config.teacher_type == 'bert':
        utils.load_bert_embedding_weight(model,
                                         config.teacher_model)
    elif config.teacher_type == 'roberta':
        utils.load_roberta_embedding_weight(model,
                                            config.teacher_model)
    elif config.teacher_type == 'rAg':
        utils.load_roberta_embedding_weight(model, config.teacher_model[0])

    for k, v in model.named_parameters():
        new_d[k] = torch.sum(v)

    logger.info("=" * 10 + "alter" + "=" * 10)
    for k in pre_d.keys():
        if pre_d[k] != new_d[k]:
            logger.info(k)
    del pre_d, new_d

    model = model.to(device)
    emd_tool = None
    if config.use_emd and config.use_kd:
        emd_tool = Emd_Evaluator(config.layers, 12, device,weight_rate=config.quantity)

    if not config.use_kd:
        teacher_model = None
    else:

        if config.teacher_type == 'gpt2':
            teacher_model = GPT2ForSequenceClassification.from_pretrained(
                config.teacher_model, num_labels=n_classes)
        elif config.teacher_type == 'bert':
            teacher_model = BertForSequenceClassification.from_pretrained(
                config.teacher_model, num_labels=n_classes)
        elif config.teacher_type == 'roberta':
            teacher_model = RobertaForSequenceClassification.from_pretrained(
                config.teacher_model)
        if config.teacher_type == 'rAg':
            teacher_model = [RobertaForSequenceClassification.from_pretrained(
                config.teacher_model[0]), GPT2ForSequenceClassification.from_pretrained(config.teacher_model[1], num_labels=n_classes)]
            teacher_model[0] = teacher_model[0].to(device)
            teacher_model[0].eval()
            teacher_model[1] = teacher_model[1].to(device)
            teacher_model[1].eval()
        else:
            teacher_model = teacher_model.to(device)
            teacher_model.eval()

    # model size
    mb_params = utils.param_size(model)
    if config.is_master:
        logger.info("Model size = {:.3f} MB".format(mb_params))

    optimizer = torch.optim.SGD(
        model.parameters(), config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.epochs/2, eta_min=config.lr_min)
    torch.autograd.set_detect_anomaly(True)
    best_top1 = 0
    best_top1_epoch = 0

    ############### BUILDING MODEL /END ###############

    ############### TRAIN /START ###############
    # training loop

    train_check_step_mark_record = []

    train_epoch_mark_record = []
    valid_epoch_mark_record = []
    for epoch in range(config.epochs):
        if config.student_type == 'cnn':
            drop_prob = config.drop_path_prob * epoch / config.epochs
            model.drop_path_prob(drop_prob)

        # training
        train_metric_results = train(logger, config, train_dataloader, model, teacher_model, optimizer, epoch, task_name.lower(),
                  emd_tool=emd_tool)
        train_check_step_mark_record += [train_metric_results]
        this_epoch_final_metric = train_metric_results[-1]
        train_epoch_mark_record += [this_epoch_final_metric]
        lr_scheduler.step()
        # validation
        if config.teacher_type == 'rAg':
            cur_step = (epoch + 1) * len(train_dataloader[0])
            val_result = validate(
                logger, config, eval_dataloader[0], model, epoch, cur_step, task_name.lower(), "val")
        else:
            cur_step = (epoch + 1) * len(train_dataloader)
            val_result = validate(logger, config, eval_dataloader,
                            model, epoch, cur_step, task_name.lower(), "val")

        # top1 = validate(test_loader, model, criterion, epoch, cur_step, "test", len(task_types))
        # save
        valid_epoch_mark_record += [val_result]
        metric_name = task_metric_dict[task_name.lower()]
        top1 = val_result[metric_name]
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
            best_top1_epoch = epoch
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        logger.info("Present best {}@1 = {:.4%}".format(metric_name,best_top1))
        print("")

    logger.info("Final best {}@1 = {:.4%}".format(metric_name,best_top1))
    # 如果有老师，config里记一下老师的评估参数
    if config.teacher_type is not None:
        with open(config.teacher_model + "/eval_results_{}.txt".format(config.datasets.lower())) as teacher_metric:
            config.teacher_metric = ''.join(teacher_metric.readlines())
    # 记录更详细的最优评估（用的什么指标，最优值多少，是哪个epoch）
    config.best_mark = {'metric_name':metric_name,'best_score':best_top1,'best_epoch':best_top1_epoch}
    generate_report_by_metrics(config,
                               train_check_step_mark_record,#训练时每个checkStep的评估指标
                               train_epoch_mark_record,     #训练时每个epoch的评估指标
                               valid_epoch_mark_record)     #验证时每个epoch的评估指标
    ############### TRAIN /END ###############


def train(logger, config, train_loader, model, teacher_model, optimizer, epoch, task_name, emd_tool=None):
    top1 = utils.AverageMeter()
    losses = utils.AverageMeter()

    if config.teacher_type != 'rAg':
        train_loader = [train_loader]

    total_num_step = len(train_loader[0])
    cur_step = epoch * len(train_loader[0])
    cur_lr = optimizer.param_groups[0]['lr']

    logger.info("Epoch {} LR {}".format(epoch + 1, cur_lr))
    # writer.add_scalar('train/lr', cur_lr, cur_step)
    model.train()

    if model.output_mode == "classification":
        criterion = nn.CrossEntropyLoss()
    elif model.output_mode == "regression":
        criterion = nn.MSELoss()

    loader_data = [i for i in train_loader[0]]

    metric_results = []

    if config.teacher_type == 'rAg':
        loader_data_plus = [i for i in train_loader[0]]

    for step in range(loader_data.__len__()):
        data = [x.to("cuda", non_blocking=True) for x in loader_data[step]]

        input_ids, input_mask, segment_ids, label_ids, seq_lengths = data
#       [32,128]    [32,128]   [32,128]      [32]         [32]
        X = [input_ids, input_mask, segment_ids, seq_lengths]
        y = label_ids

        N = X[0].size(0)

        optimizer.zero_grad()
        # [32,2] , [[32,128,768],[32,128,768],[32,128,768],[32,128,768]]
        logits = model(X)
        if config.use_emd:
            logits, s_layer_out = logits
        if config.use_kd:
            # with torch.no_grad():
            #     check_ids = input_ids.cpu()  #[32,128]
            #     check_seg = segment_ids.cpu() #[32,128]
            #     mask_check = input_mask.cpu() #[32,128]

            if config.teacher_type == 'roberta' or config.teacher_type == 'gpt2':
                output_dict = teacher_model(
                    input_ids, attention_mask=input_mask, output_hidden_states=True, return_dict=True)

                teacher_logits, teacher_reps = output_dict.logits, output_dict.hidden_states
            elif config.teacher_type == 'bert':
                output_dict = teacher_model(
                    input_ids=input_ids,token_type_ids=segment_ids, attention_mask=input_mask, output_hidden_states=True, return_dict=True)
                teacher_logits, teacher_reps = output_dict.logits, output_dict.hidden_states
                # print("teacher_logits:",teacher_logits.shape)
                # print("len_logits",len(teacher_reps))
                # for item in teacher_reps:
                #     print(item.shape)
                # print(['*']*20)
            elif config.teacher_type == 'rAg':
                output_dict_r = teacher_model[0](
                    input_ids, attention_mask=input_mask, output_hidden_states=True, return_dict=True)
                teacher_logits_r = output_dict_r.logits
                teacher_reps_r = output_dict_r.hidden_states

                data_plus = [x.to("cuda", non_blocking=True)
                             for x in loader_data_plus[step]]
                input_ids_p, input_mask_p, segment_ids_p, label_ids_p, seq_lengths_p = data_plus
                output_dict_g = teacher_model[1](
                    input_ids_p, attention_mask=input_mask_p, output_hidden_states=True, return_dict=True)
                teacher_logits_g = output_dict_g.logits
                teacher_reps_g = output_dict_g.hidden_states
                teacher_logits = torch.add(teacher_logits_r, teacher_logits_g)

                teacher_logits = teacher_logits / 2
                teacher_reps = teacher_reps_r + teacher_reps_g

            # print(np.argmax(teacher_logits.detach().cpu().numpy(),axis=1))
            # print(label_ids.cpu().numpy())
            # print("#####################################################")
            kd_loss, _, _ = distillation_loss(
                logits, y, teacher_logits, model.output_mode, alpha=config.kd_alpha)
            rep_loss = 0
            if config.use_emd:
                if config.hidn2attn:
                    s_layer_out = convert_to_attn(s_layer_out, input_mask)
                    teacher_reps = convert_to_attn(teacher_reps, input_mask)
                rep_loss, flow, distance = emd_tool.loss(
                    s_layer_out, teacher_reps, return_distance=True)
                if config.update_emd:
                    emd_tool.update_weight(flow, distance)
            loss = kd_loss + rep_loss
            # loss = kd_loss
        else:
            loss = criterion(logits, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        preds = logits.detach().cpu().numpy()
        if model.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif model.output_mode == "regression":
            preds = np.squeeze(preds)        # top5.update(prec5.item(), N)
        result = compute_metrics(task_name, preds, y.detach().cpu().numpy())

        metric_name = task_metric_dict[task_name]
        train_acc = result[metric_name]
        losses.update(loss.item(), N)
        top1.update(train_acc, N)

        if step % config.print_freq == 0 or step == total_num_step - 1:
            logger.info(
                "Train: , [{:2d}/{}] Step {:03d}/{:03d} Loss {:.3f} {} {top1:.1%}"
                .format(epoch + 1, config.epochs, step, total_num_step - 1, losses.avg,metric_name, top1=train_acc))
            result['step'] = total_num_step * epoch + cur_step
            result['loss'] = losses.avg
            metric_results.append(result)
        cur_step += 1
    return metric_results


def validate(logger, config, data_loader, model, epoch, cur_step, task_name, mode="dev"):
    top1 = utils.AverageMeter()
    losses = utils.AverageMeter()
    if model.output_mode == "classification":
        criterion = nn.CrossEntropyLoss()
    elif model.output_mode == "regression":
        criterion = nn.MSELoss()
    eval_labels = []
    model.eval()
    preds = []
    epoch_ = epoch + 1
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            data = [x.to(device, non_blocking=True) for x in data]
            # input_ids, input_mask, segment_ids, label_ids, seq_lengths = data
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = data

            X = [input_ids, input_mask, segment_ids, seq_lengths]
            y = label_ids
            N = X[0].size(0)
            logits = model(X)
            if config.use_emd:
                logits, _ = logits
            loss = criterion(logits, y)
            correct = torch.sum(torch.argmax(logits, axis=1) == y)

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
            eval_labels.extend(y.detach().cpu().numpy())

        preds = preds[0]
        if model.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif model.output_mode == "regression":
            preds = np.squeeze(preds)
        print(preds)
        result = compute_metrics(task_name, preds, eval_labels)

        print(np.sum(preds == eval_labels), len(eval_labels), result)
        metric_name = task_metric_dict[task_name]
        acc = result[metric_name]
        result['epoch'] = epoch_
    logger.info(
        mode + ": [{:2d}/{}] Final {}@1 {:.4%}".format(epoch_, config.epochs, metric_name, acc))
    return result


if __name__ == "__main__":
    main()
