
import matplotlib.pyplot as plt
import numpy as np


# mytemplate = Template(filename='report.html')
# render = mytemplate.render(name=("Python GUIs", "Python IDEs", "Python web scrapers"))
# with open('retest.html','w') as out:
#     out.write(render)

# train_epoch_mark_record
x_dot = [0, 50, 100, 114, 230, 280, 330, 344, 460, 510, 560, 574, 690, 740, 790, 804]
y_acc =  [0.25, 0.40625, 0.53125, 0.65, 0.75, 0.59375, 0.78125, 0.75, 0.65625, 0.78125, 0.71875, 0.85, 0.71875, 0.65625, 0.59375, 0.7]
y_loss =  [2.368673801422119, 4.853103302273095, 2.7758423269975303, 2.524854229606745, 0.610259473323822, 0.6389203685171464, 0.6398254346729505, 0.6404845453270649, 0.6763741374015808, 0.6385315621600431, 0.6281130856806689, 0.6276547682909305, 0.5953279137611389, 0.6420185747099858, 0.6408071957602359, 0.6448715168621184]
y_f1_micro =  [0.25, 0.40625, 0.53125, 0.65, 0.75, 0.59375, 0.78125, 0.75, 0.65625, 0.78125, 0.71875, 0.85, 0.71875, 0.65625, 0.59375, 0.7]
y_f1_macro =  [0.2, 0.28888888888888886, 0.3469387755102041, 0.393939393939394, 0.42857142857142855, 0.37254901960784315, 0.6157804459691252, 0.42857142857142855, 0.46907993966817496, 0.43859649122807015, 0.41818181818181815, 0.45945945945945943, 0.41818181818181815, 0.39622641509433965, 0.37254901960784315, 0.6000000000000001]
y_f1_binary =  [0.0, 0.0, 0.6938775510204082, 0.787878787878788, 0.8571428571428571, 0.7450980392156863, 0.8679245283018867, 0.8571428571428571, 0.7843137254901961, 0.8771929824561403, 0.8363636363636363, 0.9189189189189189, 0.8363636363636363, 0.7924528301886793, 0.7450980392156863, 0.8]


train_check_step_mark_record = [[{'acc': 0.25, 'f1_macro': 0.2, 'f1_micro': 0.25, 'f1_binary': 0.0, 'f1': 0.2, 'acc_and_f1': 0.225, 'step': 0, 'loss': 2.368673801422119}, {'acc': 0.40625, 'f1_macro': 0.28888888888888886, 'f1_micro': 0.40625, 'f1_binary': 0.0, 'f1': 0.28888888888888886, 'acc_and_f1': 0.34756944444444443, 'step': 50, 'loss': 4.853103302273095}, {'acc': 0.53125, 'f1_macro': 0.3469387755102041, 'f1_micro': 0.53125, 'f1_binary': 0.6938775510204082, 'f1': 0.3469387755102041, 'acc_and_f1': 0.439094387755102, 'step': 100, 'loss': 2.7758423269975303}, {'acc': 0.65, 'f1_macro': 0.393939393939394, 'f1_micro': 0.65, 'f1_binary': 0.787878787878788, 'f1': 0.393939393939394, 'acc_and_f1': 0.521969696969697, 'step': 114, 'loss': 2.524854229606745}],
                                    [{'acc': 0.75, 'f1_macro': 0.42857142857142855, 'f1_micro': 0.75,
                                     'f1_binary': 0.8571428571428571, 'f1': 0.42857142857142855,
                                     'acc_and_f1': 0.5892857142857143, 'step': 230, 'loss': 0.610259473323822},
                                    {'acc': 0.59375, 'f1_macro': 0.37254901960784315, 'f1_micro': 0.59375,
                                     'f1_binary': 0.7450980392156863, 'f1': 0.37254901960784315,
                                     'acc_and_f1': 0.4831495098039216, 'step': 280, 'loss': 0.6389203685171464},
                                    {'acc': 0.78125, 'f1_macro': 0.6157804459691252, 'f1_micro': 0.78125,
                                     'f1_binary': 0.8679245283018867, 'f1': 0.6157804459691252,
                                     'acc_and_f1': 0.6985152229845626, 'step': 330, 'loss': 0.6398254346729505},
                                    {'acc': 0.75, 'f1_macro': 0.42857142857142855, 'f1_micro': 0.75,
                                     'f1_binary': 0.8571428571428571, 'f1': 0.42857142857142855,
                                     'acc_and_f1': 0.5892857142857143, 'step': 344, 'loss': 0.6404845453270649}], [
                                    {'acc': 0.65625, 'f1_macro': 0.46907993966817496, 'f1_micro': 0.65625,
                                     'f1_binary': 0.7843137254901961, 'f1': 0.46907993966817496,
                                     'acc_and_f1': 0.5626649698340875, 'step': 460, 'loss': 0.6763741374015808},
                                    {'acc': 0.78125, 'f1_macro': 0.43859649122807015, 'f1_micro': 0.78125,
                                     'f1_binary': 0.8771929824561403, 'f1': 0.43859649122807015,
                                     'acc_and_f1': 0.6099232456140351, 'step': 510, 'loss': 0.6385315621600431},
                                    {'acc': 0.71875, 'f1_macro': 0.41818181818181815, 'f1_micro': 0.71875,
                                     'f1_binary': 0.8363636363636363, 'f1': 0.41818181818181815,
                                     'acc_and_f1': 0.5684659090909091, 'step': 560, 'loss': 0.6281130856806689},
                                    {'acc': 0.85, 'f1_macro': 0.45945945945945943, 'f1_micro': 0.85,
                                     'f1_binary': 0.9189189189189189, 'f1': 0.45945945945945943,
                                     'acc_and_f1': 0.6547297297297296, 'step': 574, 'loss': 0.6276547682909305}], [
                                    {'acc': 0.71875, 'f1_macro': 0.41818181818181815, 'f1_micro': 0.71875,
                                     'f1_binary': 0.8363636363636363, 'f1': 0.41818181818181815,
                                     'acc_and_f1': 0.5684659090909091, 'step': 690, 'loss': 0.5953279137611389},
                                    {'acc': 0.65625, 'f1_macro': 0.39622641509433965, 'f1_micro': 0.65625,
                                     'f1_binary': 0.7924528301886793, 'f1': 0.39622641509433965,
                                     'acc_and_f1': 0.5262382075471699, 'step': 740, 'loss': 0.6420185747099858},
                                    {'acc': 0.59375, 'f1_macro': 0.37254901960784315, 'f1_micro': 0.59375,
                                     'f1_binary': 0.7450980392156863, 'f1': 0.37254901960784315,
                                     'acc_and_f1': 0.4831495098039216, 'step': 790, 'loss': 0.6408071957602359},
                                    {'acc': 0.7, 'f1_macro': 0.6000000000000001, 'f1_micro': 0.7, 'f1_binary': 0.8,
                                     'f1': 0.6000000000000001, 'acc_and_f1': 0.65, 'step': 804,
                                     'loss': 0.6448715168621184}]]

train_epoch_mark_record = [
    {'acc': 0.65, 'f1_macro': 0.393939393939394, 'f1_micro': 0.65, 'f1_binary': 0.787878787878788,
     'f1': 0.393939393939394, 'acc_and_f1': 0.521969696969697, 'step': 114, 'loss': 2.524854229606745},
    {'acc': 0.75, 'f1_macro': 0.42857142857142855, 'f1_micro': 0.75, 'f1_binary': 0.8571428571428571,
     'f1': 0.42857142857142855, 'acc_and_f1': 0.5892857142857143, 'step': 344, 'loss': 0.6404845453270649},
    {'acc': 0.85, 'f1_macro': 0.45945945945945943, 'f1_micro': 0.85, 'f1_binary': 0.9189189189189189,
     'f1': 0.45945945945945943, 'acc_and_f1': 0.6547297297297296, 'step': 574, 'loss': 0.6276547682909305},
    {'acc': 0.7, 'f1_macro': 0.6000000000000001, 'f1_micro': 0.7, 'f1_binary': 0.8, 'f1': 0.6000000000000001,
     'acc_and_f1': 0.65, 'step': 804, 'loss': 0.6448715168621184}]
valid_epoch_mark_record = [{'acc': 0.6838235294117647, 'f1_macro': 0.40611353711790393, 'f1_micro': 0.6838235294117647,
                            'f1_binary': 0.8122270742358079, 'f1': 0.40611353711790393,
                            'acc_and_f1': 0.5449685332648343, 'epoch': 1},
                           {'acc': 0.6838235294117647, 'f1_macro': 0.40611353711790393, 'f1_micro': 0.6838235294117647,
                            'f1_binary': 0.8122270742358079, 'f1': 0.40611353711790393,
                            'acc_and_f1': 0.5449685332648343, 'epoch': 2},
                           {'acc': 0.6838235294117647, 'f1_macro': 0.40611353711790393, 'f1_micro': 0.6838235294117647,
                            'f1_binary': 0.8122270742358079, 'f1': 0.40611353711790393,
                            'acc_and_f1': 0.5449685332648343, 'epoch': 3},
                           {'acc': 0.6862745098039216, 'f1_macro': 0.4772372372372373, 'f1_micro': 0.6862745098039216,
                            'f1_binary': 0.8078078078078079, 'f1': 0.4772372372372373, 'acc_and_f1': 0.5817558735205794,
                            'epoch': 4}]


def generate_report_by_metrics(config, train_check_step_mark_record, train_epoch_mark_record, valid_epoch_mark_record):
    for train_epoch_metric in train_check_step_mark_record:
        for train_step_metric in train_epoch_metric:
            print(train_epoch_metric)


# generate_report_by_metrics(None,train_check_step_mark_record,train_epoch_mark_record,valid_epoch_mark_record)

# def draw_pic():
def draw_pic(x_check_step, y_step_loss, y_step_acc, y_step_f1_micro, y_step_f1_macro, y_step_f1_binary):
    plt.figure(figsize=(6, 5))
    # 第一个参数shape也就是我们网格的形状
    # 第二个参数loc,位置,这里需要注意位置是从0开始索引的
    # 第三个参数colspan跨多少列,默认是1
    # 第四个参数rowspan跨多少行,默认是1
    plt.subplots_adjust(
        wspace=0.5, hspace=2.5)
    shape = (15, 4)
    ax1 = plt.subplot2grid(shape, (0, 0), colspan=3, rowspan=2)
    # 如果为他设置一些属性的话，如plt.title,则用ax1的话
    # ax1.set_title(),同理可设置其他属性
    ax1.set_title("checkstep train loss")
    plt.plot(x_check_step, y_step_loss)

    ax2 = plt.subplot2grid(shape, (4, 0), colspan=3, rowspan=2)
    # 如果为他设置一些属性的话，如plt.title,则用ax1的话
    # ax1.set_title(),同理可设置其他属性
    ax2.set_title("checkstep train acc")
    plt.plot(x_check_step, y_step_acc)

    # ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=1)
    ax3 = plt.subplot2grid(shape, (7, 0), colspan=3, rowspan=2)
    ax3.set_title('f1_binary')

    plt.plot(x_dot, y_f1_binary)
    ax4 = plt.subplot2grid(shape, (10, 0), colspan=3, rowspan=2)
    ax4.set_title('f1_micro')
    plt.plot(x_dot, y_f1_micro)
    ax5 = plt.subplot2grid(shape, (13, 0), colspan=3, rowspan=2)
    ax5.set_title('f1_macro')
    plt.plot(x_dot, y_f1_macro)
    plt.show()


def draw_pic_f1():
    plt.xlabel("check_step")  # x轴上的名字
    plt.ylabel("f1")  # y轴上的名字

    plt.legend(loc='upper left')
    l1 = plt.plot(x_dot, y_f1_micro, color='green', label='micro-f1', linewidth=2)
    l2 = plt.plot(x_dot, y_f1_macro, color='red', label='macro-f1', linewidth=2)
    # l3 = plt.plot(x_dot, y_f1_binary, color='yellow',label = 'binary-f1', linewidth=2)
    # plt.legend(handles=[l1, l2], labels=['up', 'down'], loc='best')

    plt.show()


loss_dot = []
for i in range(x_dot.__len__()):
    x = x_dot[i]
    y = y_loss[i]
    loss_dot.append([x, y])

# draw_pic(x_dot,y_loss,y_acc,y_f1_micro,y_f1_macro,y_f1_binary)
draw_pic_f1()