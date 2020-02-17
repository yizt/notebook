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

 

 

 

 

 