### Unsupervised Feature Learning via Non-Parametric Instance Discrimination

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



### A simple framework for contrastive learning of visual representations 

<https://arxiv.org/pdf/2002.05709.pdf>



### MoCo V2:Improved Baselines with Momentum Contrastive Learning

<https://arxiv.org/pdf/2003.04297.pdf>