# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Utils to train DistilBERT
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import json
import logging
import os
import socket

# import git
import numpy as np
import torch


import logging


class FileLogger:
  def __init__(self, output_dir, is_master=False, is_rank0=False):
    self.output_dir = output_dir

    # Log to console if rank 0, Log to console and file if master
    if not is_rank0: self.logger = NoOp()
    else: self.logger = self.get_logger(output_dir, log_to_file=is_master)

  def get_logger(self, output_dir, log_to_file=True):
    logger = logging.getLogger('imagenet_training')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    if log_to_file:
      vlog = logging.FileHandler(output_dir+'/verbose.log')
      vlog.setLevel(logging.INFO)
      vlog.setFormatter(formatter)
      logger.addHandler(vlog)

      eventlog = logging.FileHandler(output_dir+'/event.log')
      eventlog.setLevel(logging.WARN)
      eventlog.setFormatter(formatter)
      logger.addHandler(eventlog)

      time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
      debuglog = logging.FileHandler(output_dir+'/debug.log')
      debuglog.setLevel(logging.DEBUG)
      debuglog.setFormatter(time_formatter)
      logger.addHandler(debuglog)
      
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)
    return logger

  def console(self, *args):
    self.logger.debug(*args)

  def event(self, *args):
    self.logger.warn(*args)

  def info(self, *args):
    self.logger.info(*args)

# no_op method/object that accept every signature
class NoOp:
  def __getattr__(self, *args):
    def no_op(*args, **kwargs): pass
    return no_op


# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,
# )
# logger = logging.getLogger(__name__)

# def git_log(folder_path: str):
#     """
#     Log commit info.
#     """
#     repo = git.Repo(search_parent_directories=True)
#     repo_infos = {
#         "repo_id": str(repo),
#         "repo_sha": str(repo.head.object.hexsha),
#         "repo_branch": str(repo.active_branch),
#     }

#     with open(os.path.join(folder_path, "git_log.json"), "w") as f:
#         json.dump(repo_infos, f, indent=4)


def init_gpu_params(params):
    """
    Handle single and multi-GPU / multi-node.
    """
    if params.n_gpu <= 1:
        params.local_rank = 0
        params.master_port = -1
        params.is_master = True
        params.multi_gpu = False
        return

    assert torch.cuda.is_available()

    print("Initializing GPUs")
    if params.n_gpu > 1:
        assert params.local_rank != -1

        params.world_size = int(os.environ["WORLD_SIZE"])
        params.n_gpu_per_node = int(os.environ["N_GPU_NODE"])
        params.global_rank = int(os.environ["RANK"])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.multi_gpu = True

        assert params.n_nodes == int(os.environ["N_NODES"])
        assert params.node_id == int(os.environ["NODE_RANK"])

    # local job (single GPU)
    else:
        assert params.local_rank == -1

        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1
        params.multi_gpu = False

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in multi-node distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1

    # summary
    PREFIX = f"--- Global rank: {params.global_rank} - "
    print(PREFIX + "Number of nodes: %i" % params.n_nodes)
    print(PREFIX + "Node ID        : %i" % params.node_id)
    print(PREFIX + "Local rank     : %i" % params.local_rank)
    print(PREFIX + "World size     : %i" % params.world_size)
    print(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    print(PREFIX + "Master         : %s" % str(params.is_master))
    print(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    print(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    print(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if params.multi_gpu:
        print("Initializing PyTorch distributed")
        torch.distributed.init_process_group(
            init_method="env://", backend="nccl", 
        )


def set_seed(args):
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
