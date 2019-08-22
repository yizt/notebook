Large-scale Bisample Learning on ID vs. Spot Face Recognition 工程实现



权重更新

1：初始化权重来源于每个特别嵌入特征的均值

2：支配队列和候选队列初始化；根据嵌入特征做K临近

 a)  生成嵌入特征(根据初始网络)

 b)  将嵌入特征加入faiss中；同时保存一份到hdf5中，作为初始原型W

 c)   使用faiss生成每个类别的K临近(k为300和100)

3：选择原型

​      随机选择B/2个类别;随机选择B/2个每个类别随机选择一个样本;



3：支配队列更新：