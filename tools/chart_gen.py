import numpy as np
import re
strinfo = re.compile('\b+')
PRIMITIVES = [
    'stdconv_3',
    'stdconv_5',
    'stdconv_7',
    'skip_connect',
    'dilconv_3',
    'dilconv_5',
    'dilconv_7',
    'avg_pool_3x3',
    'max_pool_3x3',
    'none',
]
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

f = open('/data/lxk/NLP/darts-KD/alpha_print_one_.log')
result = dict()
shape_label_index = dict()
history = 0
for line in f.readlines():
    tmp = np.array(eval(line))
    if tmp.shape not in shape_label_index:
        shape_label_index[tmp.shape] = list(range(history, len(tmp) + history))
        history += len(tmp)
    for index, alpha in enumerate(tmp):
        if shape_label_index[tmp.shape][index] not in result:
            result[shape_label_index[tmp.shape][index]] = [softmax(alpha)]
        else:
            result[shape_label_index[tmp.shape][index]].append(softmax(alpha))
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 20),)

for k, v in result.items():
    plt.subplot(3,3,k+1)
    alpha = np.array(v)
    alpha = alpha.T

    for i, x in enumerate(alpha):
        y = range(len(x))
        plt.plot(y, x, label=PRIMITIVES[i])
    plt.legend()
plt.savefig('tools/new.jpg', dpi=300)
