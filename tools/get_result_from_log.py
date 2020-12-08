import glob
import os
import numpy as np
import re
import argparse
fw = open('result.txt', 'w')

def get_step_1_result(path):
    all_filse = sorted(glob.glob(path))
    for file in all_filse:
        final = 0
        gt = ""
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                if "Final best Prec@1" in line:
                    final = line.strip().split('=')[-1].replace('%', "")
                if "Best Genotype" in line:
                    gt = line.strip().split('Best Genotype = ')[-1]
            epoch = 0

            for line in lines:
                if str(final) in line:
                    try:
                        epoch = int(
                            re.search('\[\s*[0-9]*/[0-9]*\]', line).group().split('/')[0][1:])
                        break
                    except:
                        break

        print(file.split('/')[-1], round(float(final), 5) * 100, epoch, gt, sep='\t', file=fw)


def get_step_2_result(path):
    all_filse = sorted(glob.glob(path))
    for file in all_filse:
        with open(file) as f:
            final = 0
            for line in f.readlines():
                if "Final best Prec@1" in line:
                    final = line.strip().split('=')[-1].replace('%', "")
            print(
                file.split('/')[-1],
                round(float(final), 5),
                sep='\t',
                file=fw,
            )


def get_step_1_EMD_result(path):
    all_files = sorted(glob.glob(path))
    for file in all_files:
        try:
            with open(file) as f:
                genos, emd = [], []
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if "EMD loss" in line:
                        line = line.strip().split("EMD loss: ")[-1].strip()
                        emd_loss = float(line)
                        emd.append(emd_loss)
                        f.readline()
                        while True:
                            line = f.readline()
                            if 'Genotype' in line:
                                break
                            if not line:
                                break
                        genos.append(line.strip())
                min_emd_epoch = np.argmin(emd)
                print(
                    file,
                    min_emd_epoch,
                    emd[min_emd_epoch],
                    genos[min_emd_epoch],
                )
        except:
            continue

def draw_step_2_results(path):
    import numpy as np
    import matplotlib.pyplot as plt
    all_accs = {}
    all_files = sorted(glob.glob('./snas_one_step_*.log'))
    for file in all_files:
        print(file)
        with open(file) as f:
            accus = []
            for line in f.readlines():
                if "Final Prec@1" in line:
                    line = line.strip().split("Final Prec@1")[1].split("Loss")[0]
                    line = eval(line)
                    accus.append(line['acc'])
            all_accs[file] = accus
    for k, i in all_accs.items():
        plt.plot(range(len(i)), i, label=k, alpha=0.5)
    plt.legend()
    plt.savefig('darts_result3.png', dpi=400)


def get_step1_geno_from_file(path):
    if not os.path.exists(path):
        files = glob.glob(path)
    else:
        files = [path]
    for file in files:
        with open(file) as f:
            for line in f.readlines():
                if 'Genotype' in line and 'Best' not in line:
                    print(line.strip())

def draw_training(path):
    import re
    # import matplotlib.pyplot as plt
    import torch
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('runs/show_result_add')
    # fw = open('result.txt', 'w')
    _a = dict()
    # plt.figure(figsize=(24,4))
    pattern = "[\[|\d|\.|\]| |\+|e|\-]+"
    all_files = sorted(glob.glob(path))
    for f in all_files:
        vals, tmp, alphas, flag = [], [[]], [], False
        loss, emd_loss = [], []
        with open(f) as fo:
            print(f)
            for line in fo.readlines():
                x = re.search("val: \[.*\] Final Prec@1 ", line)
                if '# Alpha - normal' in line:
                    flag = True
                if flag:
                    if "# Alpha - no softmax" in line:
                        flag = False
                        sa = []
                        for t in tmp:
                            if len(t) > 0:
                                t = re.sub("\s+", ',', ",".join(t))
                                t = re.sub("\]\[", '],[', t)
                                sa.append(np.array(eval(t)))
                        if len(sa) > 0:
                            for i in sa:
                                print(i.shape)
                            alphas.append(np.concatenate(sa, axis=0))
                        tmp = [[]]
                    rs = re.findall(pattern, line.strip())
                    if len(rs) > 1:
                        len_rs = [len(x) for x in rs]
                        maxlen_rs = len_rs.index(max(len_rs))
                        select_rs = rs[maxlen_rs]
                    else:
                        select_rs = rs[0]
                    if "[[" in select_rs:
                        select_rs = select_rs[select_rs.find('[['):]
                    if "]]" in select_rs:
                        select_rs = select_rs[:select_rs.find(']]') + 2]
                    if '[' in select_rs or ']' in select_rs:
                        tmp[-1].append(select_rs)
                        if "]]" in select_rs:
                            tmp.append([])

                elif x is not None:
                    vals.append(eval(line.strip()[x.end():].split('Loss')[0]))
                    line = line.strip().split("Loss")[-1]
                    line = line.split("EMD loss")
                    loss.append(eval(line[0].strip(':')))
                    emd_loss.append(eval(line[1].strip(':')))
        alpha_diff, alpha_none = [], []
        pre_struc = None
        change_ep = []
        for ep_index, alpha in enumerate(alphas):
            alpha, _none = alpha[:, :-1], alpha[:, -1]
            alpha_diff.append(np.average(np.max(alpha, axis=0) - np.min(alpha, axis=0)))
            struc = np.argmax(alpha)
            if pre_struc is None:
                pre_struc = struc
            elif pre_struc != struc:
                change_ep.append(ep_index)
                pre_struc = struc
        result = [x['f1'] for x in vals]

        if result[1::2] == result[::2]:
            result = result[::2]
            loss = loss[::2]
            emd_loss = emd_loss[::2]

        best_ep = result.index(max(result))
        print(f, [round(x, 3) for x in (alpha_diff[1::5] + [alpha_diff[-1]])], best_ep, round(max(result), 4), change_ep, sep='\t', file=fw)
        # _a[f] = np.array([alpha_diff, alpha_none, result, loss, emd_loss])
    # for step in range(len(alpha_diff)):
    #     for type_index, type_name in enumerate(['alpha_diff', 'alpha_none', 'result', 'loss', 'emd_loss']):
    #         tmp = {}
    #         for i in _a.keys():
    #             tmp[i] = torch.tensor(_a[i][type_index][step])
    #         writer.add_scalars(type_name, tmp, global_step=step)
    #         print(type_name, tmp)
if __name__ == "__main__":
    # get_step_1_result("*alr*")
    # get_step_2_result("*alr*")
    draw_training("EMD_SEP_TRAIN_HARD_1000_OP_cnn_alr_1.0_0.log")
    exit(0)
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--type',
        metavar="T",
        type=int,
        required=True,
        help='1: get step one files result \n 2: get step two files result\n3: get step one EMD result\n4: draw step two results to plt\n5: get all step one genotypes from file'
    )
    parser.add_argument(
        '--path',
        metavar="P",
        type=str,
        required=True,
        help='file path, support glob Unix style pathname pattern expansion')
    args = parser.parse_args()
    func_dict = {
        1: get_step_1_result,
        2: get_step_2_result,
        3: get_step_1_EMD_result,
        4: draw_step_2_results,
        5: get_step1_geno_from_file
    }
    func_dict[args.type](args.path)