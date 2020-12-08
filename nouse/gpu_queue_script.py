import os
import sys
import time
from datetime import datetime


def train_step_2():
    cmds = []
    f = open("train_genotyps.txt")
    for index, line in enumerate(f.readlines()):
        line = line.strip()
        file = line[:line.find(".log")] + ".log"
        # while os.path.exists('step2_' + file):
        #     file = file[:-4] + '+.log'
        geno = "\"" + line[line.find("Geno"):] + "\""
        cmds.append(
            "nohup python augment.py --use_emd False --use_kd False --genotype {} > step2_{} 2>&1 &"
            .format(geno, file))
    return cmds


def train_step_1():
    cmds = []
    emd_rate = [0, 0.1, 1, 100, 500, 1000]
    hidden_attention = [True, False]
    alpha_opt = ['sgd', 'adam']
    alpha_lr = {'sgd': [1, 0.5, 0.1, 0.025], 'adam': [1e-4, 5e-4, 1e-3, 5e-3]}
    
    training_script = "nohup python model_search.py --kd_alpha 0.0 --w_lr 1e-3 --epochs 20 --hidn2attn {} --alpha_optim {} --alpha_ep 2 --alpha_ac_rate 1 --add_op cnn --alpha_lr {} --emd_rate {} > _EMD_SEP_TRAIN_HID2ATT_{}_OPT_{}_ALR_{}_ER_{}.log 2>&1 &"
    for o in alpha_opt:
        for l in alpha_lr[opt]:
            for e in emd_rate:
                for h in hidden_attention:
                    cmds.append(training_script.format(h, o, l, e, h, o, l, e))
    return cmds


# cmd = train_step_1()

# print('\n'.join(cmd))
# exit(0)

cmd = ['nohup python model_search.py --kd_alpha 0.0 --w_lr 1e-3 --epochs 20 --emd_rate 1 --update_emd True --hidn2attn True --alpha_ep 2 --alpha_ac_rate 0 --add_op cnn --alpha_lr 1e-3 --weight_rate 10   --add_softmax True > EMD_SEP_TRAIN_WEIGHT_10_ADDSOFT_TRUE_.log 2>&1 &',
'nohup python model_search.py --kd_alpha 0.0 --w_lr 1e-3 --epochs 20 --emd_rate 1 --update_emd True --hidn2attn True --alpha_ep 2 --alpha_ac_rate 0 --add_op cnn --alpha_lr 1e-3 --weight_rate 50   --add_softmax True > EMD_SEP_TRAIN_WEIGHT_50_ADDSOFT_TRUE_.log 2>&1 &',
'nohup python model_search.py --kd_alpha 0.0 --w_lr 1e-3 --epochs 20 --emd_rate 1 --update_emd True --hidn2attn True --alpha_ep 2 --alpha_ac_rate 0 --add_op cnn --alpha_lr 1e-3 --weight_rate 100  --add_softmax True > EMD_SEP_TRAIN_WEIGHT_100_ADDSOFT_TRUE_.log 2>&1 &',
'nohup python model_search.py --kd_alpha 0.0 --w_lr 1e-3 --epochs 20 --emd_rate 1 --update_emd True --hidn2attn True --alpha_ep 2 --alpha_ac_rate 0 --add_op cnn --alpha_lr 1e-3 --weight_rate 500  --add_softmax True > EMD_SEP_TRAIN_WEIGHT_500_ADDSOFT_TRUE_.log 2>&1 &',
'nohup python model_search.py --kd_alpha 0.0 --w_lr 1e-3 --epochs 20 --emd_rate 1 --update_emd True --hidn2attn True --alpha_ep 2 --alpha_ac_rate 0 --add_op cnn --alpha_lr 1e-3 --weight_rate 1000 --add_softmax True > EMD_SEP_TRAIN_WEIGHT_1000_ADDSOFT_TRUE_.log 2>&1 &',
'nohup python model_search.py --alpha_optim sgd --kd_alpha 0.0 --w_lr 1e-3 --epochs 20 --emd_rate 1 --update_emd True --hidn2attn True --alpha_ep 2 --alpha_ac_rate 0 --add_op cnn --alpha_lr 0.1 --weight_rate 10   --add_softmax True > EMD_SEP_SDG_TRAIN_WEIGHT_10_ADDSOFT_TRUE_.log 2>&1 &',
'nohup python model_search.py --alpha_optim sgd --kd_alpha 0.0 --w_lr 1e-3 --epochs 20 --emd_rate 1 --update_emd True --hidn2attn True --alpha_ep 2 --alpha_ac_rate 0 --add_op cnn --alpha_lr 0.1 --weight_rate 50   --add_softmax True > EMD_SEP_SDG_TRAIN_WEIGHT_50_ADDSOFT_TRUE_.log 2>&1 &',
'nohup python model_search.py --alpha_optim sgd --kd_alpha 0.0 --w_lr 1e-3 --epochs 20 --emd_rate 1 --update_emd True --hidn2attn True --alpha_ep 2 --alpha_ac_rate 0 --add_op cnn --alpha_lr 0.1 --weight_rate 100  --add_softmax True > EMD_SEP_SDG_TRAIN_WEIGHT_100_ADDSOFT_TRUE_.log 2>&1 &',
'nohup python model_search.py --alpha_optim sgd --kd_alpha 0.0 --w_lr 1e-3 --epochs 20 --emd_rate 1 --update_emd True --hidn2attn True --alpha_ep 2 --alpha_ac_rate 0 --add_op cnn --alpha_lr 0.1 --weight_rate 500  --add_softmax True > EMD_SEP_SDG_TRAIN_WEIGHT_500_ADDSOFT_TRUE_.log 2>&1 &',
'nohup python model_search.py --alpha_optim sgd --kd_alpha 0.0 --w_lr 1e-3 --epochs 20 --emd_rate 1 --update_emd True --hidn2attn True --alpha_ep 2 --alpha_ac_rate 0 --add_op cnn --alpha_lr 0.1 --weight_rate 1000 --add_softmax True > EMD_SEP_SDG_TRAIN_WEIGHT_1000_ADDSOFT_TRUE_.log 2>&1 &',
]


def gpu_info():
    gpu_status = [
        x.strip('|') for x in os.popen('nvidia-smi | grep python').read().strip().split('\n')
    ]
    gpu_status = [x.split() for x in gpu_status if x != '']
    if len(gpu_status) == 0:
        return {}
    device2pid = {int(x[0]): x[1] for x in gpu_status if x[1] not in ['45957', '45958', '45959']}
    return device2pid


def process_info():
    running_process = os.popen('ps -aux | grep python').read().strip().split('\n')
    running_process = [x for x in running_process if ("search" in x or "augment" in x)]
    running_process = [x.split() for x in running_process]
    return running_process

ALL_GPUS = 8


def narrow_setup(interval=10):
    all_gpus = set(range(ALL_GPUS))
    pid_occupy_history = dict()
    while len(cmd) > 0:
        time.sleep(interval)
        r_pids = process_info()
        if len(r_pids) < ALL_GPUS:
            occupy = gpu_info()
            if len(occupy) < ALL_GPUS:
                using_gpus = occupy.keys()
                waiting_to_load = set([x[1] for x in r_pids]) - set(occupy.values())
                tmp = []
                for x in waiting_to_load:
                    if x in pid_occupy_history:
                        tmp.append(pid_occupy_history[x])
                using_gpus = list(using_gpus) + tmp
                empty_gpu = all_gpus - set(using_gpus)
                empty_gpu = list(empty_gpu)[0]
                running_script = "CUDA_VISIBLE_DEVICES={} ".format(empty_gpu) + cmd.pop()
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")

                print("Running Time =", current_time, running_script)
                sys.stdout.flush()
                os.system(running_script)
                new_pid = set([x[1] for x in process_info()]) - set([x[1] for x in r_pids])
                pid_occupy_history[new_pid.pop()] = empty_gpu


if __name__ == '__main__':
    narrow_setup()