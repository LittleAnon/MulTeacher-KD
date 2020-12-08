""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from modeling import TinyBertForSequenceClassification
from nouse.architect import Architect
# from visualize import plot

from bert_fineturn.data_processor.glue import glue_compute_metrics as compute_metrics
from kdTool import Emd_Evaluator, distillation_loss
from dist_util_torch import set_seed, FileLogger
acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte", 'books']
corr_tasks = ["sts-b"]
mcc_tasks = ["cola"]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_checkpoint(model, is_master, dump_path, checkpoint_name: str = "checkpoint.pth"):
    """
    Save the current state. Only by the master process.
    """
    if not is_master:
        return
    mdl_to_save = model.module if hasattr(model, "module") else model
    # mdl_to_save.config.save_pretrained(self.dump_path)
    state_dict = mdl_to_save.state_dict()
    torch.save(state_dict, os.path.join(dump_path, checkpoint_name))

def main():
    config = SearchConfig()
    if config.tb_dir != "":
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.tb_dir, flush_secs=20)
    else:
        writer = None
    # init_gpu_params(config)
    config.local_rank = 0
    config.multi_gpu = False
    config.is_master = True
    set_seed(config)
    logger = FileLogger('./log', config.is_master, config.is_master)

    use_emd = config.use_emd
    print(config)
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta
    ############# LOADING DATA ###############
    task_name = config.datasets
    train_dataloader, arch_dataloader, eval_dataloader, output_mode, n_classes, config = utils.load_glue_dataset(
        config)
    logger.info(f"train_loader length {len(train_dataloader)}")
    ############### BUILDING MODEL ###############
    model = SearchCNNController(config, n_classes, output_mode)
    # utils.load_embedding_weight(model, 'teacher_utils/bert_base_uncased/pytorch_model.bin', True)

    if config.restore != "":
        old_params_dict = dict()
        for k, v in model.named_parameters():
            old_params_dict[k] = v
        model.load_state_dict(torch.load(config.restore), strict=False)
        for k, v in model.named_parameters():
            if torch.sum(v) != torch.sum(old_params_dict[k]):
                print(k + " not restore")
        del old_params_dict
    else:
        utils.load_embedding_weight(model, 'teacher_utils/bert_base_uncased/pytorch_model.bin', True)
    
    if config.n_gpu > 0:
        model.to(f"cuda:{config.local_rank}")
    if config.n_gpu > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank], find_unused_parameters=True)

    if not config.use_kd:
        teacher_model = None
    else:
        teacher_model = TinyBertForSequenceClassification.from_pretrained(
            config.teacher_model, num_labels=n_classes)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()

    

    emd_tool = None
    if config.use_emd and config.use_kd:
        emd_tool = Emd_Evaluator(config.layers, 12, device)

    if config.multi_gpu:
        architect = Architect(model.module, teacher_model, config, emd_tool)
    else:
        architect = Architect(model, teacher_model, config, emd_tool)

    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    no_decay = ["bias", "LayerNorm.weight"]
    # weights optimizer
    # if config.use_emd:
    #     opti_parameters = model.weights() + list(emd_tool.conv.parameters())
    # else:
    #     opti_parameters = model.weights()

    # w_optim = torch.optim.SGD(
    #     opti_parameters,
    #     config.w_lr,
    #     momentum=config.w_momentum,
    #     weight_decay=config.w_weight_decay)
    # alpha_optim = torch.optim.Adam(
    #     model.alphas(), config.alpha_lr, weight_decay=config.alpha_weight_decay)
    w_optim = torch.optim.SGD(
        [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad and 'alpha' not in n],
        config.w_lr,
        momentum=config.w_momentum,
        weight_decay=config.w_weight_decay)
    if config.alpha_optim.lower() == 'adam':
        alpha_optim = torch.optim.Adam([p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad and 'alpha' in n],
            config.alpha_lr, weight_decay=config.alpha_weight_decay)
    elif config.alpha_optim.lower() == 'sgd':
        alpha_optim = torch.optim.SGD([p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad and 'alpha' in n],
            config.alpha_lr, weight_decay=config.alpha_weight_decay)
    else:
        raise NotImplementedError("no such optimizer")

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    # training loop
    best_top1 = 0.
    is_best = False
    best_genotype = ""
    for epoch in range(config.epochs):
        lr = lr_scheduler.get_last_lr()[-1]
        # if config.is_master or not config.multi_gpu:
        logger.info("Epoch {}".format(epoch))
        logger.info("Learning Rate {}".format(lr))
        # training

        train(
            logger,
            config,
            train_dataloader,
            arch_dataloader,
            eval_dataloader,
            model,
            teacher_model,
            architect,
            w_optim,
            alpha_optim,
            lr,
            epoch,
            task_name.lower(),
            is_kd=config.use_kd,
            emd_tool=emd_tool,
            writer=writer)
        if config.multi_gpu:
            model.module.print_alphas(logger)
        else:
            model.print_alphas(logger)

        lr_scheduler.step()
        # validation
        # if config.is_master:
        logger.info('valid')
        cur_step = (epoch + 1) * len(train_dataloader)

        top1 = validate(logger, config, eval_dataloader, model, teacher_model, epoch, cur_step, task_name.lower(), "val", emd_tool=emd_tool)

        # genotypea
        if config.multi_gpu:
            genotypes = model.module.genotype()
        else:
            genotypes = model.genotype()
        # if config.is_master:
        logger.info("========genotype========\n" + str(genotypes))
        # # save
        is_best = best_top1 <= top1
        if is_best:
            best_top1 = top1
            best_genotype = genotypes
            if config.save_supernet != "":
                model_to_save = model.module if config.multi_gpu else model
                torch.save(model_to_save.state_dict(), config.save_supernet)
            # save_checkpoint(model, config.is_master, 'pretrain_full_model',  'RTE.bin')
        # if config.is_master:
        logger.info("Present best Prec@1 = {:.4%}".format(best_top1))
        # torch.distributed.barrier()
        # utils.save_checkpoint(model.state_dict(), config.path, True)
    # if config.is_master:]
    if config.tb_dir != "":
        writer.close()
    logger.info("Final best Prec@1 = " + str(best_top1))
    logger.info("Best Genotype = " + str(best_genotype))

def train(logger,
          config,
          train_loader,
          arch_loader,
          eval_loader,
          model,
          teacher_model,
          architect,
          w_optim,
          alpha_optim,
          lr,
          epoch,
          task_name,
          is_kd=True,
          emd_tool=None,
          writer=None):

    top1 = utils.AverageMeter()
    losses = utils.AverageMeter()

    output_mode = model.module.output_mode if config.multi_gpu else model.output_mode
    model.train()
    total_num_step = len(train_loader)
    cur_step = epoch * len(train_loader)
    valid_iter = iter(arch_loader)
    random_search = epoch < config.random_sample_epoch
    freeze = epoch < config.freeze_alpha
    update_alpha = (not freeze and not random_search) or config.alpha_only

    model_to_print = model.module if config.multi_gpu else model
    if config.multi_gpu:
        torch.distributed.barrier()

    for step, data in enumerate(train_loader):
        data = [x.to(f"cuda:{config.local_rank}", non_blocking=True) for x in data]

        input_ids, input_mask, segment_ids, label_ids, seq_lengths = data
        trn_X = [input_ids, input_mask, segment_ids, seq_lengths]
        trn_y = label_ids
        try:
            v_data = next(valid_iter)
        except StopIteration:
            valid_iter = iter(arch_loader)
            v_data = next(valid_iter)

        v_data = [x.to(f"cuda:{config.local_rank}", non_blocking=True) for x in v_data]

        v_input_ids, v_input_mask, v_segment_ids, v_label_ids, v_seq_lengths = v_data
        val_X = [v_input_ids, v_input_mask, v_segment_ids, v_seq_lengths]
        val_y = v_label_ids

        trn_t, val_t = None, None
        if is_kd:
            with torch.no_grad():
                teacher_logits, teacher_reps = teacher_model(
                    input_ids, segment_ids, input_mask, attention_out=False)
                v_teacher_logits, v_teacher_reps = teacher_model(
                    v_input_ids, v_segment_ids, v_input_mask, attention_out=False)
                trn_t = (teacher_logits, teacher_reps)
                val_t = (v_teacher_logits, v_teacher_reps)
        N = trn_X[0].size(0)
        alpha_optim.zero_grad()
        if not config.one_step and update_alpha:
            architect.unrolled_backward(trn_X, trn_y, val_X, val_y, trn_t, val_t, lr, w_optim)
            alpha_optim.step()
        if config.multi_gpu:
            torch.distributed.barrier()
        w_optim.zero_grad()
        logits = model(trn_X, random_sample=random_search, one_step=config.one_step, freeze=freeze, alpha_only=config.alpha_only)
        if config.use_emd:
            logits, s_layer_out = logits
        if is_kd:
            kd_loss, _, _ = distillation_loss(
                logits, trn_y, teacher_logits, output_mode, alpha=config.kd_alpha)
            rep_loss = 0
            if config.use_emd:
                rep_loss, flow, distance = emd_tool.loss(s_layer_out, teacher_reps, return_distance=True)
                if config.update_emd:
                    emd_tool.update_weight(flow, distance)
            loss = kd_loss * config.emd_only + rep_loss * config.emd_rate
        else:
            loss = model_to_print.crit(logits, trn_y)
        l1_loss = 0
        if config.l1 != 0:
            l1_loss = sum(torch.sum(torch.abs(x)) for x in model.alpha_normal)
            loss += l1_loss * config.l1

        loss.backward()

        no_decay = ["bias", "LayerNorm.weight"]
        # gradient clipping
        if not config.alpha_only:
            clip = nn.utils.clip_grad_norm_(model_to_print.weights(), config.w_grad_clip)
            w_optim.step()
        if config.one_step and update_alpha:
            alpha_optim.step()
        if config.tb_dir != "":
            ds, ds2 = model.format_alphas()
            for layer_index, dsi in enumerate(ds):
                writer.add_scalars(f'layer-{layer_index}-alpha', dsi, global_step=step + epoch * total_num_step)
            for layer_index, dsi in enumerate(ds2):
                writer.add_scalars(f'layer-{layer_index}-softmax_alpha', dsi, global_step=step + epoch * total_num_step)
            writer.add_scalar('loss', loss, global_step=step + epoch * total_num_step)
            writer.add_scalar("EMD", rep_loss, global_step=step + epoch * total_num_step)
            writer.add_scalar("l1 loss", l1_loss, global_step=step + epoch * total_num_step)
        preds = logits.detach().cpu().numpy()
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(task_name, preds, trn_y.detach().cpu().numpy())

        if task_name in acc_tasks:
            train_acc = result['acc']
        elif task_name in corr_tasks:
            train_acc = result['corr']
        elif task_name in mcc_tasks:
            train_acc = result['mcc']

        losses.update(loss.item(), N)
        top1.update(train_acc, N)
        # model.print_alphas(logger)

        if config.eval_during_train:
            if step % config.print_freq == 0 or step == total_num_step - 1:
                validate(logger, config, eval_loader, model, teacher_model, epoch, cur_step, task_name, mode="train_dev", emd_tool=None)
        if step % config.print_freq == 0 or step == total_num_step - 1:
            logger.info(
                "Train: , [{:2d}/{}] Step {:03d}/{:03d} Loss {:.3f}, Prec@(1,5) {top1.avg:.1%}"
                .format(epoch + 1, config.epochs, step, total_num_step - 1, losses.avg, top1=top1))
        cur_step += 1
    logger.info("{:.4%}".format(top1.avg))


def validate(logger, config, data_loader, model, teacher_model, epoch, cur_step, task_name, mode="dev", emd_tool=None):
    # eval_labels = eval_labels.detach().cpu().numpy()
    eval_labels = []
    preds = []
    model.eval()

    model_to_print = model.module if config.multi_gpu else model
    total_loss, total_emd_loss = 0, 0

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            data = [x.to("cuda", non_blocking=True) for x in data]
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = data
            X = input_ids, input_mask, segment_ids, seq_lengths
            y = label_ids
            N = X[0].size(0)
            logits = model(X, train=False)
            rep_loss = 0
            if config.use_emd:
                logits, s_layer_out = logits
                _, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)
                rep_loss, flow, distance = emd_tool.loss(s_layer_out, teacher_reps, return_distance=True)
                total_emd_loss += rep_loss.item()
            loss = model_to_print.crit(logits, y)
            total_loss += loss.item()

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
            eval_labels.extend(y.detach().cpu().numpy())
        preds = preds[0]
        if model_to_print.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif model_to_print.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(task_name, preds, eval_labels)

        if task_name == "cola":
            acc = result['mcc']
        elif task_name in ["sst-2", "mnli", "mnli-mm", "qnli", "rte", "books"]:
            acc = result['acc']
        elif task_name in ["mrpc", "qqp"]:
            acc = result['f1']
        elif task_name == "sts-b":
            acc = result['corr']
    logger.info(mode +
                ": [{:2d}/{}] Final Prec@1 {} Loss {}, EMD loss: {}".format(epoch + 1, config.epochs, result, total_loss, total_emd_loss))

    return acc


if __name__ == "__main__":
    main()
