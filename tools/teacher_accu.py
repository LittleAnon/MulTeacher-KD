""" Training augmented model """
import os
import sys
sys.path.append('./')
import torch
import torch.nn as nn
import numpy as np
from config import SearchConfig
import utils
from models.augment_cnn import AugmentCNN
from vocabulary import Vocabulary
from modeling import TinyBertForSequenceClassification, BertConfig

import random
from bert_fineturn.data_processor.glue import glue_processors as processors
from bert_fineturn.data_processor.glue import glue_compute_metrics as compute_metrics
from bert_fineturn.data_processor.glue import glue_output_modes as output_modes
from tokenization import BertTokenizer
from dataset import MultiTaskBatchSampler, get_tensor_data
import tqdm
from kdTool import Emd_Evaluator, softmax, distillation_loss
from dist_util_torch import init_gpu_params, set_seed, logger

acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte", "books"]
corr_tasks = ["sts-b"]
mcc_tasks = ["cola"]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    config = SearchConfig()
    init_gpu_params(config)
    use_emd = config.use_emd
    
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # load embeddings

    ############# LOADING DATA ###############
    task_name = config.datasets
    s_train_loader, s_eval_loader, s_arch_loader, t_train_loader, t_eval_loader, t_arch_loader, (word_mat, char_mat), output_mode, n_classes = utils.load_both(config, logger)
    logger.info(f"train_loader length {len(s_train_loader)}")
    model = None
    teacher_model = TinyBertForSequenceClassification.from_pretrained(
        config.teacher_model, num_labels=n_classes)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    best_top1 = 0
    # training loop
    validate(config, t_eval_loader, teacher_model, 0, 0, task_name.lower(), output_mode)

def validate(config, data_loader, model, epoch, cur_step, task_name, output_mode, mode="dev"):
    top1 = utils.AverageMeter()
    losses = utils.AverageMeter()
    if output_mode == "classification":
        criterion = nn.CrossEntropyLoss()
    elif output_mode == "regression":
        criterion = nn.MSELoss()
    eval_labels = []
    model.eval()
    preds = []
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            data = [x.to(device, non_blocking=True) for x in data]
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = data
            # X = data[:-1]
            logits, _, _, _ = model(input_ids, segment_ids, input_mask)
            y = label_ids
            N = input_ids.size(0)
            # logits = model(X)
            loss = criterion(logits, y)
            correct = torch.sum(torch.argmax(logits, axis=1) == y)

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
            eval_labels.extend(y.detach().cpu().numpy())

        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(task_name, preds, eval_labels)
        print(np.sum(preds == eval_labels), len(eval_labels), result)
        if task_name == "cola":
            acc = result['mcc']
        elif task_name in ["sst-2", "mnli", "mnli-mm", "qnli", "rte", 'books']:
            acc = result['acc']
        elif task_name in ["mrpc", "qqp"]:
            acc = result['f1']
        elif task_name == "sts-b":
            acc = result['corr']

    logger.info(mode + ": [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, acc))
    return acc


if __name__ == "__main__":
    main()
