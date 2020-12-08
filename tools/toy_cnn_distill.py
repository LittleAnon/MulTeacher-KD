import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append('.')
from kdTool import distillation_loss

from bert_fineturn.data_processor.glue import glue_compute_metrics as compute_metrics
from modeling import TinyBertForSequenceClassification, BertConfig
import models.blocks as blocks
from config import SearchConfig
import utils

from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def replace_masked(tensor, mask, value):
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add

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
            m.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

            if (m.bidirectional):
                nn.init.xavier_uniform_(m.weight_ih_l0_reverse.data)
                nn.init.orthogonal_(m.weight_hh_l0_reverse.data)
                nn.init.constant_(m.bias_ih_l0_reverse.data, 0.0)
                nn.init.constant_(m.bias_hh_l0_reverse.data, 0.0)
                m.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0


class ToyTextCNN(nn.Module):
    def __init__(self, config):
        super(ToyTextCNN, self).__init__()
        self.n_classes = config.n_classes
        self.bert_config = config.bert_config
        self.hidden_state = config.bert_config.hidden_size
        self.layers = config.layers
        self.stem = blocks.BertEmbeddings(self.bert_config)
        self.convs = nn.ModuleList([blocks.SepConv(self.hidden_state, self.hidden_state, 3, 1, 1, affine=True) for _ in range(2)])
        self.linear = nn.Linear(self.hidden_state * 2, self.n_classes)
        self.apply(init_weights)
    def forward(self, x):
        input_ids, mask, segment_ids, seq_lengths = x
        student_layer_out = []
        hidden_value = self.stem(input_ids, segment_ids)
        for conv in self.convs:
            hidden_value = conv(hidden_value)
            student_layer_out.append(hidden_value.permute(0, 2, 1))
        hidden_value = hidden_value.permute(0, 2, 1)
        mask = mask.unsqueeze(-1).repeat(1, 1, hidden_value.shape[-1])
        max_mid, _ = replace_masked(hidden_value, mask, -1e7).max(dim=1)
        mean_mid = torch.sum(hidden_value * mask, dim=1) / torch.sum(mask, dim=1)

        out = torch.cat([max_mid, mean_mid], dim=-1)
        logits = self.linear(out)
        return logits, student_layer_out

if __name__ == "__main__":
    config = SearchConfig()
    config.is_master = True
    config.multi_gpu = False
    train_dataloader, arch_dataloader, eval_dataloader, output_mode, n_classes, config = utils.load_glue_dataset(
        config)
    model = ToyTextCNN(config)
    utils.load_embedding_weight(model, 'teacher_utils/bert_base_uncased/pytorch_model.bin')

    model = model.cuda()
    if not config.use_kd:
        teacher_model = None
    else:
        teacher_model = TinyBertForSequenceClassification.from_pretrained(
            config.teacher_model, num_labels=n_classes)
        teacher_model = teacher_model.to("cuda")
        teacher_model.eval()
    mse = nn.MSELoss()

    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(10):
        losses = 0.0
        model.train()
        for step, data in enumerate(train_dataloader):
            data = [x.to(f"cuda", non_blocking=True) for x in data]
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = data
            trn_X = [input_ids, input_mask, segment_ids, seq_lengths]
            trn_y = label_ids
            logits, student_layer_out = model(trn_X)
            optim.zero_grad()

            if config.use_kd:
                tl, tr = teacher_model(trn_X, attention_out=False)
                loss, _, _ = distillation_loss(logits, trn_y, tl, "classification", alpha=0)
                mse_loss = mse(student_layer_out[1], tr[11]) #mse(student_layer_out[0], tr[5]) + 
                loss += mse_loss * 0.01
            else:
                loss = crit(logits, trn_y)
            loss.backward()
            optim.step()
            accu = torch.sum(torch.argmax(logits, dim=-1) == trn_y) / 32.0
            losses += loss.item()
        print(losses)
        model.eval()
        with torch.no_grad():
            preds, eval_labels =[], []
            total_accu = []
            for step, data in enumerate(eval_dataloader):
                data = [x.to(f"cuda", non_blocking=True) for x in data]
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = data
                trn_X = [input_ids, input_mask, segment_ids, seq_lengths]
                y = label_ids
                logits, _ = model(trn_X)
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                eval_labels.extend(y.detach().cpu().numpy())
            preds = preds[0]
            preds = np.argmax(preds, axis=1)

            result = compute_metrics(config.datasets.lower(), preds, eval_labels)
            print(result)
            