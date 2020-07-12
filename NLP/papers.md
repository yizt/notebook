[TOC]



### A neural probabilistic language model

https://www.researchgate.net/publication/221618573_A_Neural_Probabilistic_Language_Model



1. 网络的第一层(输入层)是将C(wt-n+1),...,C(wt-2),C(wt-1)这已知的n-1和单词的词向量首尾相连拼接起来，形成(n-1)w的向量，下面用x表示。

2. 网络的第二层(隐藏层)直接用d+Hx计算得到，d是一个偏置项。之后，用tanh作为激活函数。

3. 网络的第三层(输出层)一共有|V|个节点，每个节点yi表示下一个单词i的未归一化log概率。最后使用softmax函数将输出值y归一化成概率，最终y的计算公式如下：

    y = b + Wx + Utanh(d+Hx)

4. 最后，用随机梯度下降法把这个模型优化出来就可以了。





### Distributed Representations of Sentences and Documents



### Efficient estimation of word representations in vector space



### word2vec Parameter Learning Explained

https://arxiv.org/pdf/1411.2738





### Attention Is All You Need

<https://arxiv.org/pdf/1706.03762.pdf>

提出Transformer,仅仅依赖注意力机制，省去RNN和Conv；更好的并行，大大减少训练时长；RNN无法并行训练	；

Transformer依靠注意力机制捕获全局依赖





### ELMo:Deep contextualized word representations

<https://arxiv.org/pdf/1802.05365.pdf>

Embeddings from Language Models

基于深度双向语言模型，组合模型的不同层表示；

高质量的词表示面临两个挑战：a)词的复杂特性(句法、语义)，b)一词多义



### GPT:Improving Language Understanding by Generative Pre-Training

<https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf>

在未标注的语料上生成式与训练，在相关任务上监督微调；

提出一种半监督方法组合无监督预训练和监督微调。预训练使用语言模型作为目标函数，模型结构使用Transformer的decoder;

微调时语言模型作为辅助损失有助于提升精度

类别相关输入转换：将结构化输入转为有序序列；如：相似性；将两个句子按可能情况拼接(A+B，B+A)，产生的结果逐像素相加



总结：Transformer模型+长依赖文本





### BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

https://arxiv.org/pdf/1810.04805.pdf



 

 

 

 

 