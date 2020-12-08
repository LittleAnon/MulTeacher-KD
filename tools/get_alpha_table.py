# import torch
import os
from alpha_parse import parse
import torch
import torch.nn as nn

def process_alpha(string):
    try:
        s = string.strip().replace("device='cuda:0',", "").replace("tensor", "").replace("\n", "").replace("grad_fn=<SoftmaxBackward>", "")
        s = eval(s)[0]
    except:
        print(string)
        exit(0)
    return s

def get_next(it):
    tmp = ""
    while True:
        try:
            x = it.__next__()
        except StopIteration:
            break
        tmp = tmp + x
        if ")" in x:
            break
    return tmp
fs = [
    "searchs/conv3_att_rnn/conv3_att_rnn.log",
    "searchs/conv3_attn/conv3_attn.log",
    "searchs/conv3_attn_hid384_n2/conv3_attn_hid384_n2.log",
    "searchs/conv3_attn_hid384_n4/conv3_attn_hid384_n4.log",
    "searchs/conv3_attn_hid576_n6/conv3_attn_hid576_n6.log",
    # "searchs/conv3_attn_hid768_n8/conv3_attn_hid768_n8.log",
    "searchs/conv3_rnn/conv3_rnn.log",
    "searchs/conv3_skip_att_rnn/conv3_skip_att_rnn.log",
    ]
def a1():
    for file in fs:
        print(os.path.basename(file)[:-4], end= "|")
        f = open(file)

        genos = os.path.basename(file)[:-4].split("_")
        alter_dict = {"conv3":"conv_3", "conv5":"conv_5", "conv7":"conv_7", "att":"self_att","skip":"skip_connect"}
        genos =[alter_dict[x] if x in alter_dict else x for x in genos]
        genos = genos + ['none']

        line_iter = iter(f.readlines())
        alpah0 = []
        alpah1 = []
        alpah2 = []
        test_reult = []
        while True:
            try:
                line = line_iter.__next__()
            except StopIteration:
                break
            if line.strip() == "# Alpha - normal":
                alpah0.append(process_alpha(get_next(line_iter)))
                alpah1.append(process_alpha(get_next(line_iter)))
                alpah2.append(process_alpha(get_next(line_iter)))
            elif "{'acc':" in line:
                test_reult.append(eval(line.strip().split(':')[-1][:-1]))
        for i in [50, 150, 250]:
            i = i - 1
            print("换行".join("{}".format(x) for x in alpah0[i] + alpah1[i] + alpah2[i]), end="|")
            print(test_reult[i], end="|")
        # print()
        j = test_reult.index(max(test_reult))
        print("换行".join("{}".format(x) % x for x in alpah0[j] + alpah1[j] + alpah2[j]), "|", max(test_reult), "|", j, "|")

def a2():
    index = 0
    for file in fs:
        f = open(file)
        # genos = ['conv_3', 'self_att', 'none']
        genos = os.path.basename(file)[:-4].split("_")
        alter_dict = {"conv3":"conv_3", "conv5":"conv_5", "conv7":"conv_7", "att":"self_att","skip":"skip_connect"}
        genos =[alter_dict[x] if x in alter_dict else x for x in genos]
        genos = genos + ['none']
        line_iter = iter(f.readlines())
        alpah0 = []
        alpah1 = []
        alpah2 = []
        test_reult = []
        while True:
            try:
                line = line_iter.__next__()
            except StopIteration:
                break
            if line.strip() == "# Alpha - normal":
                
                alpah0.append(process_alpha(get_next(line_iter)))
                alpah1.append(process_alpha(get_next(line_iter)))
                alpah2.append(process_alpha(get_next(line_iter)))
            elif "M | test:" in line:
                test_reult.append(eval(line_iter.__next__().strip().split('=')[-1][:-1]))
        for i in [50, 150, 250]:
            i = i - 1
        # print()
        j = test_reult.index(max(test_reult))

        b = nn.ParameterList()
        b.append(nn.Parameter(torch.tensor(alpah0[j])))
        b.append(nn.Parameter(torch.tensor(alpah1[j])))
        b.append(nn.Parameter(torch.tensor(alpah2[j])))
        print("CUDA_VISIBLE_DEVICES=" + str(index % 8) + " nohup python augment.py --genotypes \"{'books': Genotype(normal=" + str(parse(b, 1, genos)) + ", normal_concat=range(1, 4))}\"" + " >" + os.path.basename(file)[:-4] + "_best_top1.log 2>&1 &")
        index += 1
        print("CUDA_VISIBLE_DEVICES=" + str(index % 8) + " nohup python augment.py --genotypes \"{'books': Genotype(normal=" + str(parse(b, 2, genos)) + ", normal_concat=range(1, 4))}\"" + " >" + os.path.basename(file)[:-4] + "_best_top2.log 2>&1 &")
        index += 1

        b = nn.ParameterList()
        b.append(nn.Parameter(torch.tensor(alpah0[249])))
        b.append(nn.Parameter(torch.tensor(alpah1[249])))
        b.append(nn.Parameter(torch.tensor(alpah2[249])))
        print("CUDA_VISIBLE_DEVICES=" + str(index % 8) + " nohup python augment.py --genotypes \"{'books': Genotype(normal=" + str(parse(b, 1, genos)) + ", normal_concat=range(1, 4))}\"" + " >" + os.path.basename(file)[:-4] + "_final_top1.log 2>&1 &")
        index += 1

        print("CUDA_VISIBLE_DEVICES=" + str(index % 8) + " nohup python augment.py --genotypes \"{'books': Genotype(normal=" + str(parse(b, 2, genos)) + ", normal_concat=range(1, 4))}\"" + " >" + os.path.basename(file)[:-4] + "_final_top2.log 2>&1 &")
        index += 1
def a3():
    for file in fs:
        f = open(file)
        for line in f.readlines():
            if "Final best Prec@1" in line:
                line = line.strip().split('=')
                line = line[-1]
                print(os.path.basename(file)[:-4], line)
                break

def a4():
    f = open("searchs/geno_record.txt")
    d = dict()
    for line in f.readlines():
        line = line.strip().split(',', -1)
        g = ','.join(line[:-1])
        t = line[-1]
        if g not in d:
            d[g] = [t.strip()]
        else:
            d[g].append(t.strip())
        print(g, t)
    for k, v in d.items():
        if len(v) > 1:
            print(v)
a1()