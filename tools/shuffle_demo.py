adddimport numpy as np
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

seed = np.random.randint(0, 10000)

class SeedRandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None, seed=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        if seed != None:
            self.order_list = np.arange(len(self.data_source))
            np.random.seed(seed)
            np.random.shuffle(self.order_list)
        else:
            self.order_list = torch.randperm(len(self.data_source)).tolist()
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(self.order_list)

    def __len__(self):
        return self.num_samples

    def resample(self,seed=0):
        np.random.seed(seed)
        np.random.shuffle(self.order_list)

if __name__ == "__main__":
    a = range(10)
    b = range(10, 20)
    sa = SeedRandomSampler(a, seed=10)
    sb = SeedRandomSampler(b, seed=10)  
    la = DataLoader(a, sampler=sa, batch_size=2)
    lb = DataLoader(b, sampler=sb, batch_size=2)
    print(list(la))
    print(list(lb))
    la.sampler.resample()
    lb.sampler.resample()
    print(list(la))
    print(list(lb))
