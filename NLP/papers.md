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

传统的**机器翻译**都是包含encoder和decoder的复杂RNN或CNN

本文提出Transformer,仅仅依赖注意力机制，省去RNN和Conv；更好的并行，大大减少训练时长；RNN无法并行训练	；

Transformer依靠注意力机制捕获全局依赖;

为了保持序列顺序，增加位置enbedding信息作为输入。

用途：机器翻译

评价指标：BLEU  **Bilingual Evaluation Understudy**

输入：word embedding+position embedding



代码：https://github.com/tensorflow/tensor2tensor



### ELMo:Deep contextualized word representations

<https://arxiv.org/pdf/1802.05365.pdf>

ELMo: **E**mbeddings from **L**anguage **M**odels(CNN+LSTM)

基于深度双向**语言模型**，组合模型的不同层表示；

高质量的词表示面临两个挑战：a)词的复杂特性(句法、语义)，b)一词多义

不同于以往的一个词对应一个向量，是固定的。

在ELMo世界里，预训练好的模型不再只是向量对应关系，而是一个训练好的模型。

使用时，将一句话或一段话输入模型，模型会根据上下文来推断每个词对应的**词向量**。这样做之后明显的好处之一就是对于多义词，可以结合前后语境对多义词进行理解。比如apple，可以根据前后文语境理解为公司或水果。





### GPT:Improving Language Understanding by Generative Pre-Training

<https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf>



NLP 领域中只有小部分标注过的数据，而有大量的数据是未标注，如何只使用标注数据将会大大影响深度学习的性能，所以为了充分利用大量未标注的原始文本数据，需要利用无监督学习来从文本中提取特征，最经典的例子莫过于词嵌入技术。

但是词嵌入只能 word-level 级别的任务（同义词等），没法解决句子、句对级别的任务（翻译、推理等）。出现这种问题原因有两个：

- 首先，是因为不清楚下游任务，所以也就没法针对性的进行行优化；
- 其次，就算知道了下游任务，如果每次都要大改模型也会得不偿失。

为了解决以上问题，作者提出了 GPT 框架，用一种半监督学习的方法来完成语言理解任务，GPT 的训练过程分为两个阶段：Pre-training 和 Fine-tuning。目的是学习一种通用的 Representation 方法，针对不同种类的任务只需略作修改便能适应。



不同于word Embedding、ELMo 以无监督的方式学习到一些特征，然后利用这些特征喂给一些特定的有监督模型，这里是先无监督的pre−train模型，然后直接fine-tune预训练后的模型，迁移到一些特定的有监督任务上。相当于end-to-end训练。



在未标注的语料上生成式预训练，在相关任务上监督微调；

提出一种半监督方法组合无监督预训练和监督微调。预训练使用语言模型作为目标函数，模型结构使用Transformer的decoder;

微调时语言模型作为辅助损失有助于提升精度

类别相关输入转换：将结构化输入转为有序序列；如：相似性；将两个句子按可能情况拼接(A+B，B+A)，产生的结果逐像素相加



总结：Transformer模型+长依赖文本





### BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

https://arxiv.org/pdf/1810.04805.pdf

与GPT一样：在未标注的语料上生成式预训练，在相关任务上监督微调；

深度双向上下文+Transformer的Encoder;

Masked Token 预测+Next sentence prediction 



之前的模型都只能单向训练：Bert采用Masked LM(类似完形填空) 训练双向模型；

输入是：token embedding+segment embedding+position embedding

输出是：NSP+MLM（下一句预测和掩码语言模型）

fine-tune任务没有Mask,为缓解此问题，对于选中的token; 80%使用mask填充,10%随机旋转一个token填充，10%不改变。

 

### GPT-2: Language Models are Unsupervised Multitask Learners



与GPT的区别

A) 将layer normalization放到每个sub-block之前，并在最后一个Self-attention后再增加一个layer normalization。

B) 在Pretrain部分基本与GPT方法相同，在Fine-tune部分把第二阶段的Fine-tuning有监督训练具体NLP任务，换成了无监督训练具体任务，这样使得预训练和Fine-tuning的结构完全一致。不定义这个模型应该做什么任务，模型会自动识别出来需要做什么任务。

- 数据质量：GPT 2 更高，进行了筛选
- 数据广度：GPT 2 更广， 包含网页数据和各种领域数据
- 数据数量：GPT 2 更大，WebText，800 万网页
- 数据模型：模型更大，15 亿参数
- 结构变化：变化不大
- 两阶段 vs 一步到位：GPT 1 是两阶段模型，通过语言模型预训练，然后通过 Finetuning 训练不同任务参数。而 GPT 2 直接通过引入特殊字符，从而一步到位解决问题





参考文章：

[完全图解GPT-2：看完这篇就够了](https://www.sohu.com/a/336262203_129720)

[完全图解GPT-2：看完这篇就够了（二）](https://www.sohu.com/a/393893211_120054440)

[从word2vec开始，说下GPT庞大的家族系谱 ](https://www.sohu.com/a/422574962_129720?spm=smpc.author.fd-d.2.1602467430132AR5RYY0)

[按照时间线帮你梳理10种预训练模型](https://mp.weixin.qq.com/s/vQ0LVG6X0Zqy19yC1lqxoQ)

https://jalammar.github.io/illustrated-transformer/



### GPT与BERT区别

GPT的单向语言模型采用decoder部分，decoder的部分见到的都是不完整的句子；bert的双向语言模型则采用encoder部分，采用了完整句子。

解码器在自注意力（self-attention）层上还有一个关键的差异：它将后面的单词掩盖掉了。但并不像 BERT 一样将它们替换成特殊定义的单词<mask>，而是在自注意力计算的时候屏蔽了来自当前计算位置右边所有单词的信息。

 

### Character-Level Language Modeling with Deeper Self-Attention

https://arxiv.org/pdf/1808.04444.pdf

1. 使用**causal attention**将当前词后面的词mask掉。causal attention其实与transformer的decode部分中的masked attention是一样
2. **中间层的auxiliary losses** 除了在transformer的最上层进行下一个词的预测以外，我们在transformer的每个中间层也进行了下一个词的预测，这样中间层的每个位置都会有对应的损失函数，就叫做中间层的auxiliary losses
3. **Multiple Targets的auxiliary losses** 在序列中的每个位置，模型对下一个字符进行两次（或更多次）预测。对每次预测我们使用不同的分类器，这样我们对transformer的每一层的每个位置都会产出两个或多个损失函数，选择一共作为主要损失，其他的都称为字符辅助损失。



### Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

https://arxiv.org/pdf/1901.02860.pdf

Transformer 是 Google 提出的一种先进的 NLP 模型，在很多任务上都取得比传统 RNN 更好的效果。Transformer 使用了 Self-Attention 机制，让单词之间可以直接建立联系，因此编码信息和学习特征的能力比 RNN 强。但是 Transformer 在学习**长距离依赖信息**的能力仍然有一些限制，Transformer-XL 是一种语言模型，可以提高 Transformer 学习长期依赖信息的能力。



**vanilla Transformer 的缺点**

长期依赖受限：vanilla Transformer 将数据分段进行训练，每个 segment 中单词能得到的依赖关系限制在本这个 segment 中，无法使用长期依赖的信息。

分段数据语义不完整：vanilla Transformer 将数据分段的时候，是直接按照固定的长度划分语料库里面的句子，而不考虑句子的边界，使得分割出来的 segment 语义不完整。也就是说会将一个句子分到两个 segment 里面。

评估时候速度慢：在评估模型时，每预测一个单词，都要将该单词的上文重新计算一次，效率低下。

 

**Transformer-XL 主要有两个创新点：**

第一、提出了 Segment-Level Recurrence，在 Transformer 中引入了循环机制，在训练当前 segment 的时候，会保存并**使用上一个 segment 每一层的输出向量**。这样就可以利用到之前 segment 的信息，提高 Transformer 长期依赖的能力，在训练时**前一个 segment 的输出只参与前向计算**，而**不用进行反向传播**。

第二、提出 Relative Positional Encodings，Transformer 为了表示每一个单词的位置，会在单词的 Embedding 中加入位置 Embedding，位置 Embedding 可以用三角函数计算或者学习得到。但是在 Transformer-XL 中不能使用这种方法，因为每一个 segment 都会存在相同位置的 Embedding，这样两个 segment 中同样位置的 Embedding 是一样的。因此 Transformer-XL 提出了一种新的位置编码方式，**相对位置编码** (Relative Positional Encodings)。



### UIE(Universal Information Extraction)

[Unified Structure Generation for Universal Information Extraction](https://arxiv.org/pdf/2203.12277.pdf)

