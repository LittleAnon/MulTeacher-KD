from torch.utils.data import Dataset, ConcatDataset, BatchSampler, TensorDataset
import torch
import numpy as np
import random


class OneInputDataset(Dataset):

    def __init__(self, npz_file, task_id=0, sents_num=1, loss_type=1, name=None):
        data = np.load(npz_file)
        self.data_idxs = torch.tensor(data["data_idxs"]).long()
        self.data_char_idxs = torch.tensor(data["data_char_idxs"]).long()
        self.labels = torch.tensor(data["labels"])
        self.ids = torch.tensor(data["ids"]).long()
        self.task_id = task_id
        self.num = len(data["labels"])
        self.sents_num = sents_num
        self.loss_type = loss_type
        self.name = name
        self.class_num = len(np.unique(data["labels"])) if name != 'sts-b' else 1
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.data_idxs[idx], self.data_char_idxs[idx], self.labels[idx]

    def get_task_id(self):
        return self.task_id

class MultiTaskBatchSampler(BatchSampler):

    def __init__(self, datasets, batch_size, mix_opt, extra_task_ratio):
        self._datasets = datasets
        self._batch_size = batch_size
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        train_data_list = []
        for dataset in datasets:
            train_data_list.append(self._get_shuffled_index_batches(len(dataset), batch_size))
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [list(range(i, min(i + batch_size, dataset_len))) for i in range(0, dataset_len, batch_size)]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(int(len(train_data) * self._extra_task_ratio) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(self._train_data_list, self._mix_opt, self._extra_task_ratio)
        for local_task_idx in all_indices:
            task_id = self._datasets[local_task_idx].get_task_id()
            batch = next(all_iters[local_task_idx])
            yield [(task_id, sample_id) for sample_id in batch]

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        # if len(train_data_list) > 1 and extra_task_ratio > 0:
        #     main_indices = [0] * len(train_data_list[0])
        #     extra_indices = []
        #     for i in range(1, len(train_data_list)):
        #         extra_indices += [i] * len(train_data_list[i])
        #     random_picks = int(min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices)))
        #     extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
        #     if mix_opt > 0:
        #         extra_indices = extra_indices.tolist()
        #         random.shuffle(extra_indices)
        #         all_indices = extra_indices + main_indices
        #     else:
        #         all_indices = main_indices + extra_indices.tolist()

        # else:
        for i in range(len(train_data_list)):
            all_indices += [i] * int(len(train_data_list[i]) * extra_task_ratio)
        # if mix_opt > 0:
        #     random.shuffle(all_indices)
        # all_indices += [0] * len(train_data_list[0])
        # if mix_opt < 1:
        random.shuffle(all_indices)
        return all_indices


class MultiTaskDataset(Dataset):

    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        for dataset in datasets:
            task_id = dataset.get_task_id()
            assert task_id not in task_id_2_data_set_dic, "Duplicate task_id %s" % task_id
            task_id_2_data_set_dic[task_id] = dataset

        self._task_id_2_data_set_dic = task_id_2_data_set_dic

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, idx):
        task_id, sample_id = idx
        return task_id, self._task_id_2_data_set_dic[task_id][sample_id]

def get_tensor_data(output_mode, features,mul_teacher=False):

    # if not mul_teacher:
    #     features = features[0]

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                all_seq_lengths)
    return tensor_data, all_label_ids

def get_tensor_data_new(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths_a = torch.tensor([f.seq_length_a for f in features], dtype=torch.long)
    all_input_ids_a = torch.tensor([f.input_ids_a for f in features], dtype=torch.long)
    all_input_mask_a = torch.tensor([f.input_mask_a for f in features], dtype=torch.long)
    all_segment_ids_a = torch.tensor([f.segment_ids_a for f in features], dtype=torch.long)
    all_seq_lengths_b = torch.tensor([f.seq_length_b for f in features], dtype=torch.long)
    all_input_ids_b = torch.tensor([f.input_ids_b for f in features], dtype=torch.long)
    all_input_mask_b = torch.tensor([f.input_mask_b for f in features], dtype=torch.long)
    all_segment_ids_b = torch.tensor([f.segment_ids_b for f in features], dtype=torch.long)
    all_seq_lengths_all = torch.tensor([f.seq_length_all for f in features], dtype=torch.long)
    all_input_ids_all = torch.tensor([f.input_ids_all for f in features], dtype=torch.long)
    all_input_mask_all = torch.tensor([f.input_mask_all for f in features], dtype=torch.long)
    all_segment_ids_all = torch.tensor([f.segment_ids_all for f in features], dtype=torch.long)

    tensor_data = TensorDataset(all_input_ids_a, all_input_ids_b, all_input_ids_all,
                                all_input_mask_a, all_input_mask_b, all_input_mask_all,
                                all_segment_ids_a, all_segment_ids_b, all_segment_ids_all,
                                all_label_ids,
                                all_seq_lengths_a, all_seq_lengths_b, all_seq_lengths_all)

    return tensor_data, all_label_ids

if __name__ == "__main__":
    data = OneInputDataset('data/mt_100long/books/dev.npz')