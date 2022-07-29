### InstDisc : Unsupervised Feature Learning via Non-Parametric Instance Discrimination

<https://arxiv.org/pdf/1805.01978v1.pdf>

实例级别

非监督分为两类：a) 生成模型，b）自监督方法（预测上下文、统计数量、补全缺失部分、灰度转彩色、拼图）

训练时表征fi与网络参一同学习；然后更新vi



半监督学习：先非监督学习，然后在少量标注数据上fine-tune



层次softmax, NCE, Negative sample 



参考：

NCE:<http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf>





### IIC-Invariant Information Clustering for Unsupervised Image Classification and Segmentation

<https://arxiv.org/pdf/1807.06653.pdf>

互信息、最大化互信息

clustering degeneracy、noisy data 

协同聚类与互信息



### MoCo:Momentum contrast for unsupervised visual representation learning 

<https://arxiv.org/pdf/1911.05722.pdf>

非监督任务有两方面：pretext tasks 和loss functions 

we take two random “views” of the same image under random data augmentation to form a positive pair.  



1、大数据集，除了query图像和他的同源图像（augmentation）之外，其他的图像都是负样本。利用大量负样本字典，去学习一个pretext task。

2、字典要足够大，但是受限于字典，minibatch无法足够大，所以构建queue把二者解耦。

3、为了能让queue的key真实反映当前的encoder，要保证足够的连续性。所以要动量更新encoder。

  

 



### SimCLR: A simple framework for contrastive learning of visual representations

<https://arxiv.org/pdf/2002.05709.pdf>

无监督学习视觉表征有两类方法；a) 生成式：生成或建模输入空间,缺点：像素级计算量大；b)判别式，类似监督学习，不过输入和标签都来自未标注数据；关键是如何构建模拟监督任务。

1. 组合数据增广
2. 在表征和对比损失之间加入非线性变换
3. 对比交叉熵受益于biao'zhun
4. 大的Batch-Size



### MoCo V2:Improved Baselines with Momentum Contrastive Learning

<https://arxiv.org/pdf/2003.04297.pdf>

验证了SimCLR设计的有效性；将更多数据增广和非线性变换引入MoCo中，形成v2版本；

对比损失关键在于keys怎么维持；SimCLR放在Batch中，MoCo则放在队列中。