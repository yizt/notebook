[TOC]





## ML

### xgboost

#### 特性

正则化

二阶导数近似

尺寸收缩(类似学习率)

列下采样

近似算法(贪婪学习)

带权百分比摘要(Weighted Quantile Sketch)

稀疏模型处理



#### 系统实现方法

列快并行(column block) 预计算：近似时多个块(每个块包含部分行)，对local proposal algorithms；每个列的统计可以并行

Cache-aware获取

堆外计算(块压缩和分片) 



> 优点 

- XGB利用了二阶梯度来对节点进行划分，相对其他GBM来说，精度更加高。
- 利用局部近似算法对分裂节点的贪心算法优化，取适当的eps时，可以保持算法的性能且提高算法的运算速度。
- 在损失函数中加入了L1/L2项，控制模型的复杂度，提高模型的鲁棒性。
- 提供并行计算能力，主要是在树节点求不同的候选的分裂点的Gain Infomation（分裂后，损失函数的差值）
- Tree Shrinkage，column subsampling等不同的处理细节。

> 缺点

- 需要pre-sorted，这个会耗掉很多的内存空间（2 * #data * # features）
- 数据分割点上，由于XGB对不同的数据特征使用pre-sorted算法而不同特征其排序顺序是不同的，所以分裂时需要对每个特征单独做依次分割，遍历次数为#data * #features来将数据分裂到左右子节点上。
- 尽管使用了局部近似计算，但是处理粒度还是太细了
- 由于pre-sorted处理数据，在寻找特征分裂点时（level-wise），会产生大量的cache随机访问。

> 因此LightGBM针对这些缺点进行了相应的改进。

1. LightGBM基于histogram算法代替pre-sorted所构建的数据结构，利用histogram后，会有很多有用的tricks。例如histogram做差，提高了cache命中率（主要是因为使用了leaf-wise）。
2. 在机器学习当中，我们面对大数据量时候都会使用采样的方式（根据样本权值）来提高训练速度。又或者在训练的时候赋予样本权值来关于于某一类样本（如Adaboost）。LightGBM利用了GOSS来做采样算法。
3. 由于histogram算法对稀疏数据的处理时间复杂度没有pre-sorted好。因为histogram并不管特征值是否为0。因此我们采用了EFB来预处理稀疏数据。



#### 参考：

https://zhuanlan.zhihu.com/p/38516467









## 基础网络

### Network In Network



参考：https://www.cnblogs.com/yinheyi/p/6978223.html



### ResNet v1

#### 1. 介绍

网络深度非常重要；

深度导致难以训练已经较大程度解决:  Normalized 初始化、  Batch Normalization

深度带来另一个问题：网络退化，并非过拟合(训练和测试误差都增大)

更深的网络不应该比更浅的网络差(假定最后几层就学习一个恒等映射)；说明学习恒等映射很难

考虑学习一个残差(极致情况，权重全为0就可以了)；而且不会增加计算量和参数。

实验结果表明：更容易优化、性能随深度单调增加

#### 3. 深度残差学习

存在一个假说:多层非线性函数可以逼近复杂的函数; 那么同样可以逼近我们的残差

如果恒等映射是最优的，那么残差网络将权重学习为0就可以了

实验表明恒等映射通常有很小的反应

**网络结构**

a)相同的feature map大小,相同的channel, b) feature map 减半，深度加倍

c) 维度变化使用一个1*1的卷积来解决



#### 4. 实验

普通深度网络深度加深到一定程度后训练和验证误差都增加，**推测**是由于**收敛速率指数级降低** 

恒等映射 vs 投影映射 ；后者效果稍好，但会增加参数

Bottleneck架构：层数增加，参数没有增加

残差网络的相应更小

1202层和152层训练误差一样，但验证误差更高



### ResNet V2

​          探索传播方式，前向和后向信号可以直接从一个block传到另一个block。

#### 介绍

​         聚焦直接传递信息，不仅在残差单元内，而是整个网络；

​         预激活思想设计一个全新的残差单元



#### 分析残差网络

​        如果xl+1 ≡ yl; 信息可以传递到任何浅层单元

#### Skip Connections的影响

​       shortcut connections是信息传递的最直接路径



#### 激活分析

​      预激活有两个优点：容易优化、防止过拟合



### Non-local Neural Networks

地址: https://arxiv.org/pdf/1711.07971.pdf



### Deformable Convolutional Networks

https://arxiv.org/pdf/1703.06211.pdf

 

由尺度、姿态、视角和部分形变等因素引起的几何变化是目标识别和检测的主要挑战；目前解决方法分为两类

1.数据增广: 覆盖尽可能多的变换(如: 仿射变换)；缺点：无法应对不可预知的变换前款

2.transformation-invariant 的特征和算法(如：SIFT)；缺点：手工设计的变换无关特征和算法很困难，且无法应对复杂的变换

本文提出可形变卷积和可形变RoI Pooling; 通过添加位移来丰富空间采样，位移值通过任务学习。





### GCNet

https://arxiv.org/pdf/1904.11492



可以应用到所有的残差块；不像NL由于计算量大，只用于少数层

兼具NL的性能和SENet的轻量；比NL轻量，比NL、SENet效果都要好；

实验分析NL



### VoVNet

https://arxiv.org/pdf/1904.09730.pdf

本文是DenseNet的变种。DenseNet的shortcut是密集的，即当前层的feature会传到后面的所有层。虽然DenseNet因为shortcut的原因，模型参数变少，但是其对GPU的资源消耗人仍旧非常的大，并且因为密集的shortcut，会增加后层特征channel，增大计算量。作者提出了One-short Aggregation（文中称为VoVNet），其实就是改变了shortcut的连接方式，从原来的前层feature与后面每一层都连接，变成了前层feature只与最后一层的连接。大大减少了连接数量。作者声称同样的结构，性能差不多，但是比DenseNet快2倍，energy consumption也减少了1.6~4.1倍。

 



## 目标检测

### R-CNN

​          PASCAL VOC目标检测进入停滞期, mAP 58.3; 两个见解: CNN用到region proposal中来定位对象；监督预训练和领域精调。

#### 简介

​        进展缓慢； 2012 AlexNet 在ImageNet分类引起关注，CNN分类任务多大程度能够泛化到目标检测。

​        关注两个问题：CNN定位对象，小量标注，大容量模型。

​        解决方法：region proposal和监督预训练(以前是无监督预训练)；



#### R-CNN目标检测

​         三个模块 region proposal，cnn, svm分类

​         高效：特征少，共享计算

**训练过程**

​        监督预训练

​        领域精调: 1:3 正负样本比，



### SSD



低分辨率获取高精度，进一步促进速度提升

不同feature maps预测不同的尺寸

使用小的卷积过滤器到feature maps上



### CBNet: A Novel Composite Backbone Network Architecture for Object Detection

https://arxiv.org/pdf/1909.03625.pdf



### Retinanet

https://arxiv.org/pdf/1708.02002.pdf



### Cascade R-CNN: Delving into High Quality Object Detection 

https://arxiv.org/abs/1712.00726



### Cascade R-CNN: High Quality Object Detection and Instance Segmentation

https://arxiv.org/pdf/1906.09756v1.pdf

 

### CornerNet: Detecting Objects as Paired Keypoints 

https://arxiv.org/pdf/1808.01244.pdf



### ExtremeNet 



### FCOS

https://arxiv.org/pdf/1904.01355.pdf

anchor free，单阶段，类似语义分割逐像素做检测

基于anchor的检测器的缺点

a) 对anchor尺寸、长宽比、数量敏感;需要小心的调试优化超参数

b) 无法适应尺寸的剧烈变化、小物体检测困难；且预定的anchor边框丢失通用性

c) 引起正负anchor样本数的不平衡问题

d) 大量的anchor边框iou计算耗时且占用内存



1. 对于重叠的GT 边框，像素到底预测哪一个GT box;FPN可以缓解这个问题(对于歧义的像素，关联面积小的GT box)。
2. center-ness 预测像素偏离中心的程度，该分支可以过滤低质量的检测边框
3. fcos可以利用尽可能多的前景样本，不像基于anchor的检测器，只利用IoU较高的样本





### CenterNet: Objects as Points 

https://arxiv.org/pdf/1904.07850.pdf

关键词: anchor-free,没有分组,没有后处理(如NMS);

对应目标检测keypiont就是GT 边框中心点

建模过程

1. Keypoint: 计算GT边框的k的Keypoint在步长R为预测FeatureMap(H/R,W/R,C)上的坐标(x,y)，将通道维Ck通道(x,y)赋值为1，然后使用高斯核函数均匀化；使用FocalLoss回归
2. Keypoint坐标偏移: 使用L1 loss计算由于步长R带来的坐标偏移
3. Keypoint尺寸预测：使用L1 loss计算Keypoint尺寸

预测过程

1. 通过Keypoint预测结果找到最大的n个峰值，所谓峰值就是不小于领域所有值
2. 将峰值坐标偏移，再根据尺寸即可得到边框坐标



### CenterNet: Keypoint Triplets for Object Detection

https://arxiv.org/pdf/1904.08189.pdf



CornerNet受限于较弱的全局信息能力(对关键点检测敏感，对哪些关键点应该是一对不敏感)；

CenterNet增加一个关键点感知region的中心；同时增加两个策略center pooling,cascade corner pooling



### EfficientDet

https://arxiv.org/pdf/1911.09070.pdf





## 语义分割

RefineNet、PSPNet

### DeepLab v1

<https://arxiv.org/pdf/1412.7062v3.pdf>





#### 摘要

​        三个贡献: 

a)空洞卷积:控制分辨率、增大感受野、保持参数不变

b) ASPP-空洞金字塔池化: 鲁棒分割多尺寸对象，多重采样比例和视野，来捕捉不同尺寸的对象

c) 组合DCNN和PGM: 提升边界定位，卷积网络的下采样影响定位精度，通过组合DCNN最后一层响应和全连接CRF，提升定位性能



#### 简介

​        DCNN在视觉识别任务(图像分类、目标检测)取得很大成功，关键的因素是DCNN对于局部图像变换的内在不变性；可以学习数据的抽象表示，对于分类很好；但会妨碍像语义分割这种密集预测任务，抽象的空间信息在这里不合适。

​         DCNN遇到的三个挑战: 分辨率的降低、目标的多尺寸、	不变性带来的定位精度降低

第一个挑战: 去除最后几个池化层，使用滤波器上采样(就是空洞卷积)；组合多个空洞卷积，接一个双线性插值到原图分辨率。

第二个挑战：不同于多尺寸输入，使用多个并行的不同采样率的空洞卷积，称之为ASPP.

第三个挑战：一种方式使用skip-layers，最终用多层来预测最终分割结果。我们使用全连接CRF，捕获边缘细节，满足长距离依赖

​          DeepLab的三个优点：

a)速度：8FPS, 全连接CRF 0.5秒   b) 

b)精度：VOC 2012 79.7%

c) 简单：级联两个固定的模块DCNN和CRF





#### 相关工作

​         像素分类，完全抛弃分割；组合DCNN和局部CRF，将superpixs作为顶点对待，忽略远程依赖。我们的方法将像素作为顶点，捕获远程依赖，服从快速平均场推断。

​        其它论文的重要和有价值的

a) 端到端的训练，结构化预测：我们的CRF是后处理步骤，已经有人提出的端到端联合训练 ；使用一个卷积层近似密集CRF平均场推断的一个迭代；另一个方向是使用DCNN学习CRF的成对项 

b) 弱监督：无需整张图都有像素级别标注



#### 方法

##### 空洞卷积抽取特征和增大视野

​        一种修复分辨率减小的方法是使用反卷积，这需要额外的内存和耗时。我们使用空洞卷积，可以应用到任意层，满足任意分辨率，并无缝集成。

​         可以在任意高分辨率上计算DCNN最终的响应；例如：为了加倍特征响应的空间密度；将最后一个池化层的补充设置为1，接下来所有的卷积层的采样比设置为2。在所有层上使用这种方法，可以获得原始图像分辨率的响应，但是这计算量太大。采用混合方式，平滑效率/精度;最后两层使用空洞卷积, 然后8倍双线性插值，恢复到原图分辨率。双线性插值不需要学习任何参数，速度更快

​         空洞卷积可以在任意层，任意增大视野，计算和参数保持不变。大的采样率，对性能有提升。

##### ASPP表示多尺寸图像

​         DCNN有隐式代表图像尺寸的能力，明确说明对象尺寸可以提高DCNN成功处理大尺寸和小尺寸对象的能力。

​         第一种是多尺寸处理。

​         第二种是重采样单尺寸的卷积层特征；我们使用ASPP，泛化了DeepLab-LargeFOV。



##### 全连接CRF结构化预测

​         DCNN顶层节点只有很平滑的响应，只能预测对象的粗略位置；以前有两种方式来处理：一是多层预测，而是采用超像素表示，使用低级分割方法。

​         我们组合DCNN和全连接CRF，传统的CRF用于弱分类器的平滑，相邻像素倾向于相同类别。DCNN的特征图原本就很平滑，我们的目标不是平滑而是恢复局部结构的细节

​         第一项依赖像素位置和RGB颜色，第二项只依赖像素。第一项迫使相似颜色和位置的像素有相同的标签；第二项仅考虑像素位置。



#### 实验结果

​         ImageNet预训练模型、交叉熵损失函数、在8倍下采样的每个空间计算损失、DCNN和CRF分开训练。先介绍会议版本，接下来是最近结果



##### PASCAL VOC 2012

​         20个目标对象，一个背景类; 1, 464 (train)、1, 449 (val), and 1, 456 (test) 像素级别标注图像；性能度量是21个类别上像素级别IoU

**会议版本结果**

​         使用VGG16预训练模型，mini-batch 20，学习率0.001，每2000个迭代学习率缩减10倍，权重衰减0.0005、动量大小0.9。

​          最终使用3*3的卷积核、大的采样率12、fc6和fc7层神经元改为1000个，形成DeepLab-LargeFOV版本。CRF可以提升3~5个百分点。



**会议版本后的改进**

a)学习率策略

​        使用poly学习率，比sgd好1.17%

b)采用ASPP

​        多个并行的fc6-fc7-fc8分支，fc6 是3*3卷积，ASPP-S和 ASPP-L的采样率分别为{2, 4, 8, 12}和{6, 12, 18, 24}

CRF前ASPP-S比LargeFOV好1.22%；经过CRF后基本一样(也就是ASPP没有效果)。ASPP-L还是有提升的。

c)更深的网络和多尺寸处理

​      使用ResNet101; 分别输入scale = {0.5, 0.75,1}；在score map层融合最大响应；在MS-COCO上预训练，训练时随机缩放0.5~1.5; 最终79.7%的精度。

​     

##### PASCAL-Context

​          语义标注了整个图像包括对象和stuff；在最常见的59类+背景类上评估；训练数据4998；验证集5105；最好精度为45.7%



##### PASCAL-Person-Part

​       包含更多的尺寸和姿态，标注了人体的每一部分；融合标注为Head, Torso,Upper/Lower Arms and Upper/Lower Legs 6类+1个背景类；1716个训练集，1817个验证集。最好精度为64.94%。

​        但是LargeFOV or ASPP在这个数据集上没有作用。



##### Cityscapes

​       高质量的像素级别标注，来自50个城市的5000张街景图像。19个类别属于7个大类： ground, construction,object, nature, sky, human, and vehicle；

​        training, validation, and test 分别为 2975, 500, 1525 ；最好精度71.4。

​        图像原始分辨率为2048×1024；没有使用多尺寸。



##### 失败案例

​       不能捕获纤细物体的边界，如自行车、椅子



### DeeplabV2

<https://arxiv.org/pdf/1606.00915.pdf>

语义分割面临的三大问题：a)分辨率降低；b)多尺寸；c) 	平移不变带来的定位精度降低



### DeeplabV3

<https://arxiv.org/pdf/1706.05587.pdf>



### DeeplabV3+

<https://arxiv.org/pdf/1802.02611.pdf>





### 总结

​        空洞卷积、ASPP、组合DCNN和全连接CRF。





参考：https://blog.csdn.net/u013580397/article/details/78508392

[DeepLab(1,2,3)系列总结](https://blog.csdn.net/u011974639/article/details/79148719)

[DeepLab v3+](https://www.paperweekly.site/papers/notes/326)





### MNC Instance-aware semantic segmentation via Multi-task Network Cascades

https://arxiv.org/pdf/1512.04412.pdf

COCO 2015 语义分割冠军



### InstanceFCN Instance-sensitive Fully Convolutional Networks

<https://arxiv.org/abs/1603.08678>

不同于FCN只有一个score map,产生少量instance-sensitive score maps

动机：如果能够区分左边和右边，就能够使用score map区分实例；

使用一个实例相对位置分类器

assembling 模块是复制粘贴操作

局部连贯性：





### FCIS

https://arxiv.org/pdf/1611.07709.pdf

COCO 2016 语义分割冠军

第一个instance-aware的语义分割FCN;传统FCN需要检测和分割来做到instance-aware;之前的instance-aware语义分割使用FC实现，形变和固定尺寸表示损害分割精度，参数过多，计算量大，耗时长

position-sensitive score maps 可以实现平移变化

在提议框上执行ROI操作，而不是滑动窗口



### FCN Fully Convolutional Networks for Semantic Segmentation

<https://arxiv.org/pdf/1411.4038.pdf>

FC网络可以看做卷积网络的一种；

227 × 227 耗时1.2ms产生一个点输出;500 × 500耗时22ms,产生10 × 10网格输出；对应的反向传播2.4 ms和37ms；

分类网络产生的分割结果是粗糙的

Sampling in patchwise training can correct class imbalance [27, 8, 2] and mitigate the spatial correlation of dense patches [28, 16].

Whole image fully convolutional training is identical to patchwise training where each batch consists of all the receptive fields of the units below the loss for an image (or collection of images).

If the kept patches still have significant overlap, fully convolutional computation will still speed up training.





### Learning deconvolution network for semantic segmentation 

<https://arxiv.org/pdf/1505.04366>



fixed-size receptive field ,大对象分类不一致，小对象误分类或漏标



- a multi-layer deconvolution network 
- The trained network is applied to individual object proposals to obtain instance-wise segmentations 

### 

deconvolution 是形状生成器

unpooling 捕获example-specific 结构，跟踪原始定位；

deconvolutional 捕获class-specific 形状

Instance-wise segmentation 可以处理不同的尺寸



训练细节

BN

两阶段训练：参数多，样本少； we employ a two-stage training method to address this issue, where we train the network with easy examples first and fine-tune the trained network with more challenging examples later ；

容易样本，crop GT; 困难样本使用proposals 

预测：top50 proposal 以max方式融合

FCN擅长整体形状，DeconvNet 擅长细节

### ParseNet: Looking Wider to See Better

 <https://arxiv.org/pdf/1506.04579.pdf>





### Segnet

https://arxiv.org/pdf/1511.00561.pdf

encoder-decoder结构、non-linear upsampling 

non-linear upsampling: 提升边框刻画能力，减少参数，移植到其它encoder-decoder网络中

FCN encoder参数过多，decoder参数过少，训练困难；预测耗时

使用CRF是因为解码器不够好

Therefore, it is necessary to capture and store boundary information in the encoder feature maps before sub-sampling is performed 



### U-Net







### RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation

https://arxiv.org/pdf/1611.06612.pdf





### PSPNet





### ExFuse: Enhancing Feature Fusion for Semantic Segmentation

<https://arxiv.org/pdf/1804.03821.pdf>











### Object-Contextual Representations for Semantic Segmentation

<https://arxiv.org/pdf/1909.11065.pdf>









## 实时语义分割

### HarDNet: A Low Memory Traffic Network

https://arxiv.org/pdf/1909.00948.pdf



### ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation

https://arxiv.org/pdf/1803.06815.pdf



### ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation

<https://arxiv.org/pdf/1906.09826.pdf>







### ShelfNet for Fast Semantic Segmentation

https://arxiv.org/pdf/1811.11254.pdf





### Fast-SCNN

https://arxiv.org/pdf/1902.04502.pdf

大感受野，空间细节

Encoder-decoder、多分支

ICNet [36], ContextNet [21], BiSeNet [34] and GUN [17]



DSConv + Inverted ResBlock + PPM 

### CAS

### DF1-Seg-d8



### FasterSeg: Searching for Faster Real-time Semantic Segmentation

https://arxiv.org/pdf/1912.10917.pdf

NAS

 “zoomed conv.”: bilinear downsampling + 3×3 conv. + bilinear upsampling

“zoomed conv. ×2”: bilinear downsampling + 3×3 conv. ×2 + bilinear upsampling

知识蒸馏



### BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation

https://arxiv.org/pdf/1808.00897.pdf



### LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation

https://arxiv.org/pdf/1905.02423.pdf





## 实例分割

### Hybrid Task Cascade for Instance Segmentation

https://arxiv.org/pdf/1901.07518.pdf





三点改进：

Interleaved Execution

Cascade R-CNN 虽然强行在每一个 stage 里面塞下了两个分支，但是这两个分支之间在训练过程中没有任何交互，它们是并行执行的。所以我们提出 Interleaved Execution，也即在每个 stage 里，先执行 box 分支，将回归过的框再交由 mask 分支来预测 mask，如上图（b）所示。这样既增加了每个 stage 内不同分支之间的交互，也消除了训练和测试流程的 gap。

Mask Information Flow

 Cascade Mask R-CNN 中，不同 stage 之间的 mask 分支是没有任何直接的信息流的，Mi+1 只和当前 Bi 通过 RoI Align 有关联而与 Mi 没有任何联系。多个 stage 的 mask 分支更像用不同分布的数据进行训练然后在测试的时候进行 ensemble，而没有起到 stage 间逐渐调整和增强的作用。为了解决这一问题，我们在相邻的 stage 的 mask 分支之间增加一条连接，提供 mask 分支的信息流，让 Mi+1能知道 Mi 的特征。

Semantic Feature Fusion

这一步是我们尝试将语义分割引入到实例分割框架中，以获得更好的 spatial context。因为语义分割需要对全图进行精细的像素级的分类，所以它的特征是具有很强的空间位置信息，同时对前景和背景有很强的辨别能力。通过将这个分支的语义信息再融合到 box 和 mask 分支中，这两个分支的性能可以得到较大提升。





参考：https://baijiahao.baidu.com/s?id=1627161484427243571&wfr=spider&for=pc



### YOLACT



1.两阶段分割网络过于依赖特征定位来生成mask；一阶段如FCIS在定位后有大量后处理，无法实时

2.YOLACT放弃显示的定位，将实例分割分为两个任务：a)生成non-local prototype masks ;b)	为每个实例预测一组linear combination coefficients 



### YOLACT++

https://arxiv.org/pdf/1912.06218.pdf

### PANet

https://arxiv.org/abs/1803.01534

https://github.com/ShuLiu1993/PANet

COCO 2017 实例分割第一名。

信息传递方式很重要(fpn原本自底向上路径太长)；较低层的精准定位通过bottom-up路径增强，提升了整个feature层次结构。



较低层的特征对尺寸大的对象预测也是很重要的

每个边框只通过某一层feature预测不是最优的，其它层丢弃的信息对最后预测可能很重要

mask基于单个视野预测丢弃的多样性信息，增加一个FC来辅助FCN;FC是位置敏感的，且有全局信息







### MS R-CNN

Mask Scoring R-CNN 

https://arxiv.org/pdf/1903.00241.pdf

实例分割中使用分类的分数作为mask的质量得分；mask的得分由IoU量化

smask = scls · siou 



### PolarMask: Single Shot Instance Segmentation with Polar Representation

https://arxiv.org/pdf/1909.13226.pdf



Smooth L1忽略的距离的关系；

IoU loss计算复杂，无法并行





### SOLO

https://arxiv.org/pdf/1912.04488.pdf



### CenterMask:Real-Time Anchor-Free Instance Segmentation

https://arxiv.org/pdf/1911.06667.pdf

https://github.com/youngwanLEE/CenterMask

基于FCOS+VovNet,增加SAM做mask;  SAM参考CBAM；

传统的RoiAlign没有考虑Roi尺寸

VovNet做了改进，增加残差和SE的attention



## 人脸识别

### Docface+

#### DWI-AMSoftmax Loss

$$
w_j= \frac {w_j^*}  {||w_j^*||_2}  \tag 4
$$

$$
w_j^*=(1-\alpha)w_j + \alpha w_j^{batch}  \tag 5
$$

$w_j^{batch}$ 是根据当前mini-batch计算出的目标权重向量; 就是嵌入特征 $f$  的L2归一化值；注意$w_j$ 也只更新当前mini-batch中的权重。

margin m 为5.0,最初使用



Face-ResNet架构

一般的方法到ID-自拍数据集上迁移效果不好：收敛慢，陷入局部极小值；

由于数据集浅，造成欠拟合

**数据采样** 

​           随机采样$B/2$ 各类别；$B$ 是batch-size; 然后从每个类别各采集一个ID图像和自拍图像





### 总结

https://www.jianshu.com/p/1dd8c0364710



## OCR/场景文本检测、识别



### CRNN

类序列识别，长度变化大；CNN不能可变

end-to-end、不需要字符级别标注、不受限于固定词典、序列长度

### CTPN

固定宽度、垂直的anchor

CNN+RNN组合,in-network rnn

end-to-end

#### 简介

之前的检测器不鲁棒、不可靠；使用低级特征，区别单个笔画或字符，没有使用上下文信息。

faster r-cnn很在目标检测上成功;但无法直接用到这里，因为需要更高的定位精度，这是细粒度的识别

定位准、in-network rnn、适用多尺寸、多语言、end-to-end

#### 相关工作

Connected-componets:快速过滤器区分像素文本/非文本; 然后使用低级特征分组为笔画或字符候选

sliding-window:多尺寸密集滑窗检测字符候选; 计算量大

这两种通用方法受限于字符检测性能

#### Connectionist Text Proposal Network

细粒度的提议框、recurrent connectionist text proposals、side-refinement

细粒度的提议框：text line可以较好的区分文本，固定宽度,预测垂直的效果更好；anchor垂直数量10,范围11~273; 同时预测评分和垂直的边框位置;细粒度的的检测提供了更加精细的监督信息，导致更精准的定位

Recurrent Connectionist Text Proposals: 独立的检测proposals，导致类文本的噪声;上下文信息保障可靠

Side-Refinement: 水平距离小于50像素(垂直IoU>0.7)的proposals连起来; 精调两侧的边框水平方向位置







### 总结

https://blog.csdn.net/xwukefr2tnh4/article/details/80589198



## 模型解释

### CAM Learning Deep Features for Discriminative Localization

https://arxiv.org/pdf/1512.04150.pdf

1.本文重新审视全局平均池化(GAP)，阐明了GAP赋能卷积神经网络可观的定位能力，尽管网络是在图像级别的标注上训练的；

2.卷积网络的神经元实际上充当目标检测器的角色；但是分类网络中的全连接层使得网络丧失的空间定位能力

3.简单改变一下分类网络,使用GAP替换FC层，即可使得分类网络有弱监督定位能力。



### Grad-CAM:  Visual Explanations from Deep Networks via Gradient-based Localization 

https://arxiv.org/pdf/1610.02391.pdf

1.本文提出的方法Gradient-weighted Class Activation Mapping (Grad-CAM)通过对任意目标类别梯度反传到最后一个卷积层；在图像中突出对预测类别重要的区域；

2.相对于CAM适用更多网络(image classification、image captioning、VQA)；同时也不需要更改网络结构；

3.通过Grad-CAM可以轻松分别哪个网络更强哪个网络更弱。



### Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks

https://arxiv.org/pdf/1710.11063.pdf



## 模型压缩、量化

### Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM

https://arxiv.org/pdf/1707.09870.pdf

mixed integer programs 

extragradient 

ADMM 

BWN 

These results suggest that we should quantize different parts of the networks with different bit width in practice 







## 其它

视觉会议基本参考

PAMI/IJCV/JMLR 4 分；

TIP/TNNLS/TCSVT 2 分；

CVPR/ICCV/ECCV/NIPS/ICML 1.5 分

PR/CVIU/PRL/neurocomputing/AAAI/IJCAI 1 分