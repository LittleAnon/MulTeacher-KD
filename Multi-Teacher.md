<!--
 * @Author: your name
 * @Date: 2021-03-06 10:12:51
 * @LastEditTime: 2021-03-30 20:47:05
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /MulTeacher-KD/Multi-Teacher.md
-->
# Abstract

大规模预训练语言模型已经在自然语言处理任务中取得了巨大的成功。预训练语言模型按照目标函数(objective function)的不同可以分为Masked language model(MLM) 和Autoregressive language model(ALM). 基于MLM的语言模型BERT在自然语言理解任务中有很好的表现，不过，对于文本生成任务, ALM更具有优势。在这篇文章中，为了结合MLM和ALM的优势，我们将预训练的语言模型与知识蒸馏构成一个多老师模型框架。**在蒸馏过程中，为了保留老师模型的多样性,我们并没有简单的将teacher loss进行简单的求平均，我们评估了老师模型在gradient space中的diversity以便我们可以找到训练学生网络最优的方向，从而让学生网络学到自然语言理解能力和自然语言生成能力**


# Introduction

预训练的语言模型已经在NLU和NLG任务中取得了巨大的提升。根据objective function的不同，预训练的语言模型分为MLM和ALM. 基于MLM训练的语言模型比如BERT在自然语言理解任务中会有优势，对于生成任务，ALM会更具有优势。这是因为在ALM中，模型模拟的是文本序列中的序列生成过程. 这说明MLM和ALM具有天然的diversity. 

在自然语言处理中，无论是自然语言生成，还是自然语言理解。其本质上还是在刻画自然语言序列。因此，如果能利用diversity的优势会使得模型具有更好的表达能力。基于此，有学者提出了Probabilistically Masked Language model(PMLM)来构建MLM和ALM之间的gap。PMLM基于概率分布来定义masked 序列。PMLM强调其在自然语言处理任务中的生成能力，同时，作为一个masked LM, PMLM也保留了其自然语言理解的能力。

在这篇文章中，为了利用MLM和ALM的优势，很自然的我们将其应用到知识蒸馏框架中来。因为老师模型的多样性就意味着学生模型具有更强的能力。在外面的框架中，问题的关键就是如何能把老师模型的diversity传授给学生模型。在外面的模型中，我们没有简单的奖老师模型的loss求平均来训练学生网络。受人类教育的启发，我们从gradient space 优化了蒸馏的过程。在学生的学习过程中，老师会提供给学生学习的目标和学习的方向。同理，在训练学生网络的过程中，梯度可以看成是老师提供的学习方向。


## Knowledge Distillation from Ensemble Teacher

给定一个老师模型和学生模型，老师模型和学生模型的logits是$a^t$ 和 $a^s$, KD的loss如下表示:

$\mathcal{L}_{kd}=\mathcal{H}(p^s,p^s)=\mathcal{H}(
\delta(a^s;T),\delta(a^t;T))=-\sum\limits_{k=1}^K p^t[k]\log p^s[k]=-\langle p^t,\log p^s\rangle$

对于ensemble模型，vanilla KD loss 是把所有的输出p^t做一个softened output. 假设 ensemble size 为$M$, Eq.(1) 可以写成:

$\mathcal{L}_{mkd}=\mathcal{H}(p^s,\frac{1}{M}\sum\limits_{m=1}^Mp_m^t)=-\langle\frac{1}{M}\sum\limits_{m=1}^Mp_m^t,\log p^s\rangle=\frac{1}{M}\sum\limits_{m=1}^M\mathcal{H}(p^s,p_m^t)$

因此，我们使用averaged KD loss来表明ensemble KD 方法。 除了logits-based 方法，也有很多方法探索intermediate representations 的蒸馏。令 $f^t$ 和 $f^s$为teacher 和 student的feature maps.最终的objective 可以写成:

$\mathcal{L}_{fkd}=\mathcal{D}(f^t(f^t),r^s(f^s))$

类比到ensemble我不做赘述，最终的optimization objective of ensemble knowledge distillation 如下所示:

$\mathcal{L}_{ens}=\mathcal{H}(y,\delta(a^s;1)) + \lambda\cdot\mathcal{L}_{mkd}+\beta\cdot\mathcal{L}_{mfkd}$

# Model


1、多老师模型的融合方式：

现阶段：

    融合方式比较粗糙，老师模型loss直接累加作为总的loss。

toDo:

    调研loss的训练方式(苏展)

2、架构搜索和EMD是否会对结果有提升

现阶段：

    实验的结果中EMD对最中对结果贡献不明确

toDo:

    1、EMD对最终的结果是否有提升，提升多少
    2、NAS开发

# Experiment

当前实验结果：

https://docs.qq.com/sheet/DT2hQQklWYm1kZldY?tab=BB08J2

当前实验代码：

https://github.com/anonNo2/MulTeacher-KD


