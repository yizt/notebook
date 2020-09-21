## 小目标检测

### An Analysis of Scale Invariance in Object Detection – SNIP

https://arxiv.org/pdf/1711.08189.pdf

小目标检测，非数据增广



### Scale Match for Tiny Person Detection

https://arxiv.org/pdf/1912.10664.pdf



### HRDNet: High-resolution Detection Network for Small Objects

https://arxiv.org/pdf/2006.07607.pdf





## 数据增广/数据合成

### Synthesizing Training Data for Object Detection in Indoor Scenes

https://rss2017.lids.mit.edu/static/papers/67.pdf

去送是机器人最常用功能；本文通过少量真实样本合成，前景放置在合适位置和尺寸；位置和尺寸更加语义分割图和深度图决定；位置放置在桌子、吧台、茶几上；尺寸与深度成反比；



### Modeling Visual Context is Key to Augmenting Object Detection Datasets

https://arxiv.org/pdf/1807.07428.pdf

利用分割标注来增加训练数据的实例数；关键是选择合适的视觉上下文；通过训练来选择合适的上下文；

实验证明随机的复制-粘贴会损害精度；视觉上下文相关的合成才能提升精度



### Applying Domain Randomization to Synthetic Data for Object Category Detection

https://arxiv.org/pdf/1807.09834.pdf

Gazebo 3D合成模拟球、盒子、圆柱体图像，采用纹理随机性，通过真实图像加合成图像效果比CoCo预训练+真实图像fine-tune效果好



### **Augmentation for small object detection**

https://arxiv.org/pdf/1902.07296.pdf

在一个图中将小目标复制多次



### Synthetic Data Generation and Adaption for Object Detection in Smart Vending Machines

https://arxiv.org/pdf/1904.12294.pdf

自动售货机，场景下广角；3D重建前景对象+虚拟环境重建=》对象放置=》相机模拟=》渲染=》风格转换



### SLOT-BASED IMAGE AUGMENTATION SYSTEM FOR OBJECT DETECTION

https://arxiv.org/pdf/1907.12900.pdf

slot-based image augmentation：前景背景组合 

slot 是背景图像的一个矩形区域；原始的slot可以从原始图像和标注中获取；

增加mAP较低的类别作为目标类别；

候选过滤：通过属性匹配（尺寸、长宽比、类别）；



[9] proposed a machine learning based method to search suitable positions to insert the objects. These positions are ranked by the performance evaluation of the applied object detection models. 

Among these works, [10] highlights the importance of scales and [11] conducted similar works on small object detection 



### Effects of different arrangements in visual input data on object detection accuracy(不是论文)

https://www.in.tum.de/fileadmin/w00bws/i06/Teaching/SS19/VFL_Ahmad_Khan.pdf







### How much real data do we actually need: Analyzing object detection performance using synthetic and real data

https://arxiv.org/pdf/1907.07061.pdf



a)减少数据对性能影响；b) 度量不同数据集分布的关系；c)混合大规模合成数据和少量真实数据训练；d)大规模合成数据训练，少量真实样本fine-tune;e)一次训练多个合成数据集，在小量真实样本fine-tune。

采用的是现有的合成模拟数据集(Synscapes (7D) 、Playing for Benchmark (P4B) 、CARLA ),在车和人两个类别上实验

结论：fine-tune优于混合训练；逼真程度不如多样性重要









