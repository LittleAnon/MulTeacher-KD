
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds,average='macro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }



def json_data(data):
    uids = data['uids']
    predictions = data['scores']
    previous = ""
    rank_list = []
    data_list = []
    for i, uid in enumerate(uids):
        qid = uid.split('-')[0]
        aid = "-".join(uid.split('-')[1:])
        if qid != previous:
            ordered_rank_list = sorted(rank_list, key=lambda x: x[1], reverse=True)
            rank_dict = {}
            for j, value in enumerate(ordered_rank_list):
                rank_dict[value[0]] = j + 1
            for j, value in enumerate(rank_list):
                data_list.append([previous, value[0], rank_dict[value[0]]])
            previous = qid
            rank_list = []
        rank_list.append((aid, predictions[2*i+1]))
    ordered_rank_list = sorted(rank_list, key=lambda x: x[1], reverse=True)
    rank_dict = {}
    for j, value in enumerate(ordered_rank_list):
        rank_dict[value[0]] = j+1
    for j, value in enumerate(rank_list):
        data_list.append([previous, value[0], rank_dict[value[0]]])
    return data_list

def eval_map_mrr(data_list, gold_file):
    dic = {}
    fin = open(gold_file)
    for line in fin:
        line = line.strip()
        if not line:
            continue
        cols = line.split('\t')
        if cols[0] == 'QuestionID':
            continue

        q_id = cols[0]
        a_id = cols[4]

        if not q_id in dic:
            dic[q_id] = {}
        dic[q_id][a_id] = [cols[6], -1]
    fin.close()

    for cols in data_list:
        q_id = cols[0]
        a_id = cols[1]
        rank = int(cols[2])
        dic[q_id][a_id][1] = rank

    MAP = 0.0
    MRR = 0.0
    cnt = 0
    for q_id in dic:
        flag = False
        for k,v in dic[q_id].items():
            if v[0] == '1':
                flag = True
        if flag:
            cnt += 1
        else:
            continue

        sort_rank = sorted(dic[q_id].items(), key=lambda asd: asd[1][1], reverse=False)
        correct = 0
        total = 0
        AP = 0.0
        mrr_mark = False
        for i in range(len(sort_rank)):
            # compute MRR
            if sort_rank[i][1][0] == '1' and mrr_mark == False:
                MRR += 1.0 / float(i + 1)
                mrr_mark = True
            # compute MAP
            total += 1
            if sort_rank[i][1][0] == '1':
                correct += 1
                AP += float(correct) / float(total)
        if correct != 0:
            AP /= float(correct)
        else:
            AP = 0.0
        MAP += AP
    MAP /= float(cnt)
    MRR /= float(cnt)
    return MAP, MRR

def eval_qa(data, gold_file):
    data_list = json_data(data=data)
    MAP, MRR = eval_map_mrr(data_list, gold_file)
    return MAP, MRR



def compute_metrics(task_name, preds, labels):
    if 'wikiqa' not in task_name.lower():
        assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "trec":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wikiqa":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name in ["wikiqadev", "wikiqatest"]:
        data_list = json_data(preds)
        x = eval_map_mrr(data_list, labels)
        return {"map": x[0], "mrr":x[1]}
    elif task_name == "sick":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "sickreg":
        return pearson_and_spearman(preds, labels)
    elif task_name == "agnews":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "dbpedia":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)