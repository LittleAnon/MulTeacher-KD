import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import sys
import random

# sys.path.append('/disk2/lxk/paperRelated/darts-KD')
# from torch.nn.parallel import DistributedDataParallel as DDP
# from dist_util_torch import init_gpu_params, set_seed, logger
# import argparse



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '45236'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)
    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    # print(f"Running basic DDP example on rank {rank}.")
    # setup(rank, world_size)
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--n_gpu", type=int, default=0)
    args = parser.parse_args()
    init_gpu_params(args)
    # create model and move it to GPU with id rank
    model = ToyModel().to(args.local_rank)
    ddp_model = DDP(model, device_ids=[args.local_rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    for epoch in range(10):
        for step in range(10):
            optimizer.zero_grad()
            outputs = ddp_model(torch.randn(20, 10))
            labels = torch.randn(20, 5).to(rank)
            loss_fn(outputs, labels).backward()
            optimizer.step()
        if rank not in [-1, 0]:
            torch.distributed.barrier()
        if rank == 0:
            print("hello")
            torch.distributed.barrier()
    cleanup()

def warm_up_test():
    t = torch.tensor([0.0], requires_grad=True)
    optim = torch.optim.SGD([t], lr=1.)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 10)
    for e in range(10):
        optim.step()
        lr_scheduler.step()
        print("-", lr_scheduler.last_epoch, lr_scheduler.get_lr()[0], optim.param_groups[0]['lr'])

def load_test():
    from torch.utils.data.sampler import RandomSampler, SequentialSampler
    from torch.utils.data import DataLoader

    data = range(10)
    data2 = data
    rs = RandomSampler(data)
    seq = SequentialSampler(data)
    l1 = DataLoader(data, sampler=rs, batch_size=2)
    l2 = DataLoader(data, sampler=seq, batch_size=2)
    print(list(l1))
    print(list(l2))
# def run_demo(demo_fn, world_size):


if __name__ == "__main__":
    # from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler, SequentialSampler
    # from torch.utils.data.distributed import DistributedSampler
    # from torch.utils.data import DataLoader
    # import numpy as np
    # torch.manual_seed(0)
    # np.random.seed(0)
    # data = range(10)
    # order = list(range(len(data)))
    # random.shuffle(order)
    # data2 = [x + 10 for x in range(10)]
    # sampler = OrderdedSampler(data, order)
    # dataloader = DataLoader(data, sampler=sampler, batch_size=2)
    # dataloader2 = DataLoader(data2, sampler=sampler, batch_size=2)
    # print(list(dataloader))
    # print(list(dataloader2))
    import torch.distributed.autograd as dist_autograd
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch import optim
    from torch.distributed.optim import DistributedOptimizer
    from torch.distributed.rpc import RRef

    import torch.distributed as dist
    import torch.utils.data.distributed
    import argparse
    # ......
    parser = argparse.ArgumentParser(description='PyTorch distributed training on cifar-10')
    parser.add_argument('--rank', default=0,
                        help='rank of current process')
    parser.add_argument('--word_size', default=2,
                        help="word size")
    parser.add_argument('--init_method', default='tcp://127.0.0.1:23456',
                        help="init-method")
    args = parser.parse_args()

    dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.word_size)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)

    net = Net()
    net = net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net)
