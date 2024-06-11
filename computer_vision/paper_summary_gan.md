

[TOC]



Learning to discover cross-domain relations with generative adversarial networks

DualGAN: Unsupervised dual learning for image-to-image translation 

Unsupervised image-to- image translation networks

### DCGAN: UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS

https://arxiv.org/pdf/1511.06434.pdf	

三个关键点使得网络可以稳定训练

1、步长卷积替代池化层，让网络自己学习空间下采样

2、去除全连接层，直接使用Conv；因为全连接要全局平均池化GAP,GAP会使用收敛变慢；判别器的输出层打平后直接Sigmod激活判断

3、使用BN，但是在生成器的输出层和判别器的输入层不使用BN

4、生成器使用ReLU,输出层使用Tanh

5、判别器使用LeakyReLU



### pix2pix: Image-to-Image Translation with Conditional Adversarial Networks

 <https://arxiv.org/pdf/1611.07004.pdf>

欧氏距离会生成模糊照片；

仅仅指定high-level的目标（与真实照片不可分辨）

U-Net结构的生成器

cGAN:  G : {x,z} → y 



PatchGAN判别器：This discriminator tries to classify if each N × N patch in an im- age is real or fake.  



### CycleGAN:Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

<https://arxiv.org/pdf/1703.10593.pdf>

使用未成对的图像，从源领域映射到目标领域G : X → Y ；同时学习一个反转的映射F : Y → X  ；并保证F(G(X)) ≈ X 



### Pix2pixHD:High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs

<https://arxiv.org/pdf/1711.11585.pdf>



### BigGAN:Large Scale GAN Training for High Fidelity Natural Image Synthesis

<https://arxiv.org/pdf/1809.11096.pdf>



### StyleGAN:A Style-Based Generator Architecture for Generative Adversarial Networks

<https://arxiv.org/pdf/1812.04948.pdf>



### VQ-VAE 

https://arxiv.org/pdf/1906.00446.pdf



### A Review on Generative Adversarial Networks: Algorithms, Theory, and Applications

<https://arxiv.org/pdf/2001.06937.pdf>