<!--
 * @Author: your name
 * @Date: 2021-03-06 10:12:51
 * @LastEditTime: 2021-03-23 11:51:02
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /MulTeacher-KD/Multi-Teacher.md
-->
# Abstract

大规模预训练语言模型已经在自然语言处理任务中取得了巨大的成功。预训练语言模型按照目标函数(objective function)的不同可以分为Masked language model(MLM) 和Autoregressive language model(ALM). 基于MLM的语言模型BERT在自然语言理解任务中有很好的表现，不过，对于文本生成任务, ALM更具有优势。在这篇文章中，为了结合MLM和ALM的优势，我们将预训练的语言模型与知识蒸馏构成一个多老师模型框架。老师模型基于MLM和ALM，通过知识蒸馏技术，学生模型可以同时具有自然语言理解和自然语言生成的能力。

# Introduction

预训练的语言模型已经在NLU和NLG任务中取得了巨大的提升。根据objective function的不同，预训练的语言模型分为MLM和ALM. 在MLM中，输入的句子序列有一部分会被[MASK]代替, 我们会预测这些被masked掉的token。例如，在BERT模型中，我们会随机15%的词来进行masked。预训练的MLM已经在NLU任务重取得了stoa结果。对于生成任务，研究者更倾向于ALM. ALM中会去模拟文本序列中的序列生成过程而不是预测masked tokens. 因此，ALM在自然语言生成任务中会表现的更好。

为了充分利用MLM和ALM的优势，有学者提出了Probabilistically Masked Language model(PMLM)来构建MLM和ALM之间的gap。PMLM基于概率分布来定义masked 序列。PMLM强调其在自然语言处理任务中的生成能力，同时，作为一个masked LM, PMLM也保留了其自然语言理解的能力。

在这篇文章中，为了利用MLM和ALM的优势，我们将预训练的语言模型与知识蒸馏相结合提出了一种多老师模型框架。对于老师模型，我们使用了原始的老师模型。比如MLM模型BERT，ALM GPT2. 学生模型我们使用CNN模型。通过知识蒸馏技术，学生模型同时学到了自然语言理解和自然语言生成知识。


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


