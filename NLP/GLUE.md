

​        自然语言处理（NLP）主要包括自然语言理解（NLU）和自然语言生成（NLG）。为了让NLU任务发挥最大的作用，来自纽约大学、华盛顿大学等机构创建了一个多任务的自然语言理解基准和分析平台，也就是GLUE（General Language Understanding Evaluation）。

GLUE包含九项NLU任务，语言均为英语。GLUE九项任务涉及到自然语言推断、文本蕴含、情感分析、语义相似等多个任务。像BERT、XLNet、RoBERTa、ERINE、T5等知名模型都会在此基准上进行测试。目前，大家要把预测结果上传到官方的网站上，官方会给出测试的结果。

GLUE的论文为：GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding[[1\]](https://zhuanlan.zhihu.com/p/135283598#ref_1)

GLUE的官网为：[https://gluebenchmark.com/](https://link.zhihu.com/?target=https%3A//gluebenchmark.com/)



### **CoLA**

CoLA(The Corpus of Linguistic Acceptability，语言可接受性语料库)，单句子分类任务，语料来自语言理论的书籍和期刊，每个句子被标注为是否合乎语法的单词序列。本任务是一个二分类任务，标签共两个，分别是0和1，其中0表示不合乎语法，1表示合乎语法。

样本个数：训练集8, 551个，开发集1, 043个，测试集1, 063个。

任务：可接受程度，合乎语法与不合乎语法二分类。

评价准则：Matthews correlation coefficient。

标签为1（合乎语法）的样例：

- She is proud.
- she is the mother.
- John thinks Mary left.
- Yes, she did.
- Will John not go to school?
- Mary noticed John's excessive appreciation of himself.

标签为0（不合语法）的样例：

- Mary sent.
- Yes, she used.
- Mary wonders for Bill to come.
- They are intense of Bill.
- Mary thinks whether Bill will come.
- Mary noticed John's excessive appreciation of herself.

注意到，这里面的句子看起来不是很长，有些错误是性别不符，有些是缺词、少词，有些是加s不加s的情况，各种语法错误。但我也注意到，有一些看起来错误并没有那么严重，甚至在某些情况还是可以说的通的。



### SST-2

SST-2(The Stanford Sentiment Treebank，斯坦福情感树库)，单句子分类任务，包含电影评论中的句子和它们情感的人类注释。这项任务是给定句子的情感，类别分为两类正面情感（positive，样本标签对应为1）和负面情感（negative，样本标签对应为0），并且只用句子级别的标签。也就是，本任务也是一个二分类任务，针对句子级别，分为正面和负面情感。

样本个数：训练集67, 350个，开发集873个，测试集1, 821个。

任务：情感分类，正面情感和负面情感二分类。

评价准则：accuracy。

标签为1（正面情感，positive）的样例：

- two central performances
- against shimmering cinematography that lends the setting the ethereal beauty of an asian landscape painting
- the situation in a well-balanced fashion
- a better movie
- at achieving the modest , crowd-pleasing goals it sets for itself
- a patient viewer

标签为0（负面情感，negative）的样例：

- a transparently hypocritical work that feels as though it 's trying to set the women 's liberation movement back 20 years
- so pat it makes your teeth hurt
- blood work is laughable in the solemnity with which it tries to pump life into overworked elements from eastwood 's dirty harry period .
- faced with the possibility that her life is meaningless , vapid and devoid of substance , in a movie that is definitely meaningless , vapid and devoid of substance
- monotone
- this new jangle of noise , mayhem and stupidity must be a serious contender for the title .

注意到，由于句子来源于电影评论，又有它们情感的人类注释，不同于CoLA的整体偏短，有些句子很长，有些句子很短，长短并不整齐划一。

 

