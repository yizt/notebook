[TOC]

## 扩散模型



https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy



s [90, 215, 220, 225]



denoising diffusion probabilistic models (DDPMs) [90, 215], scorebased generative models (SGMs) [220, 221], and stochastic differential equations (Score SDEs) [113, 219, 225]	

denoising diffusion probabilistic models (DDPMs) [90, 166, 215], score-based generative models (SGMs) [220, 221], and stochastic differential equations (Score SDEs) [219, 225].





[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)

[Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf)

[Generative modeling by estimating gradients of the data distribution](https://arxiv.org/pdf/1907.05600.pdf)



[Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/pdf/2208.11970.pdf)

## 文本生成图像

### clip [Learning Transferable Visual Models From Natural Language Supervision]( )

​        CLIP就是在这些特征上进行对比学习，即缩小正样本间的距离，拉大负样本间的距离。标准的对比学习方法往往是将每一张图像进行两种不同的变换，得到两张新的图像。这两张图像共同构成一个正样本对，而其他图像之间则构成负样本对，即用图像的自身变换作为同一样本的监督信号。

​        CLIP使用自然语言作为监督信号，这里正样本对的定义是图像和其对应的文本，负样本对就是不匹配的图像-文本。样本对之间的距离定义为图像特征和文本特征的点积。对于N个样本，我们可以构建一个N\*N的特征距离矩阵，矩阵**对角线**上的元素就是**正样本对**之间的距离度量，矩阵中**其他元素**则是**负样本对**之间的距离度量。对比学习要做的是让对角线上的元素尽可能地大，让其他元素尽可能地小。为了充分发挥对比学习的作用，OpenAI专门从互联网上收集了**4亿对图像-文本**对来进行模型的预训练。

​        在预训练过程中，我们只对图像编码器和文本编码器进行了训练，并没有训练一个分类器，所以在训练阶段得到的输出只是图像和文本的特征。



**（1）为什么要引入自然语言充当监督信号？**

​         首先，使用自然语言作为监督信号能够更容易地扩展类别信息，而不是从机器学习任务中定义好的N个类别选出一个概率最大的标签。其次，与大多数无监督或自我监督的学习方法相比，从自然语言中学习不仅只是学习一种表征，还可以将这种表征与语言连接起来，从而实现了灵活的zero-shot迁移。如果只使用单模态的自监督学习，无论是单模态的对比学习如Moco，还是单模态的掩码模型如MAE，都只能学到视觉特征，无法与自然语言联系起来，很难通过zero-shot迁移到下游数据集上。



​        之前的工作同样尝试过使用文本作为监督信号，但是它们试图预测每幅图像所伴随的文本的确切单词。其他研究发现，在同样的性能下，生成模型比对比学习的模型需要更多的计算量。作者探究到一个更简单的代理任务来进行预训练，即只预测文本整体与图像配对，而不预测文本的确切单词。如图３，Transformer Language Model是一种按顺序预测的方法来预测文本的确切单词；Bag of Words Prediction代表仅需预测出所有的单词且可以是乱序的。我们可以看出CLIP的预训练方法在效率和准确率上均为达到了最佳。

 

**（3）为什么要利用prompt方法？**

自然语言中一个常见的问题是一词多义。在某些情况下，同一个单词的多个含义可能被包含在同一个数据集中的不同类中。比如，crane本意是鹤，但是也可以表示起重机。当我们直接将label输入给CLIP的文本编码器时，由于缺乏上下文，编码器可能无法区分label所表达的含义。为了帮助弥合这种分布差距，作者发现句子“A photo of A {label}.”是一个很好的帮助描述图像内容的模板。实验表明，仅使用这个提示就可以将ImageNet上的准确率提高1.3%。针对不同类型的数据集也可以设计不同的prompt，例如，当对动物进行分类时，可以使用“A photo of a {label}, a type of pet.”作为文本编码器的输入。



​        无论是对比之前有监督学习的方法(如EfficientNet、ViT、BiT-M、ResNet等)，还是对比自监督学习的方法（如，SimCLRv2、BYOL、Moco等），CLIP在相同的计算量下的性能都达到了最佳。

**总结**

​        CLIP使用自然语言作为对比学习的监督信号来预训练一个迁移性能很好的视觉模型。该模型打破了传统的方法只能预测一个固定的、预先被确定的物体类别的限制，通过从文本里学习监督信号，使用自然语言引导模型进行分类任务，可以扩展到新的分类类别。但是CLIP也具有很多的局限性。比如，CLIP在细分类任务上表现不佳，而且也不能处理特别抽象的概念，比如统计图像中包含有多少个物体。除此之外，虽然CLIP在自然图像上的泛化能力较强，但是对于完全**out-of-distribution**的数据集，如MNIST上预测的泛化结果很差。

​       尽管CLIP打破了传统的方法只能预测一个固定的、预先被确定的物体类别的限制，但是它仍然需要**人为地给定类别标签**，然后**才能计算相似度**。更灵活的方法是让模型直接生成图像的标题作为它的类别。一个简单的想法是将对比学习的目标函数与生成式的目标函数结合起来，以此希望模型可以同时拥有对比学习的高效性和生成模型的灵活性。



A connection between score matching and denoising autoencoders



Estimating the Hessian by back-propagating curvature.



## MCMC



## Autoregressive Models



### **The neural autoregressive distribution estimator** 



## 标准化流



### 依赖知识

变量变换定理





## Score Matching

### Estimation of non-normalized statistical models by score matching

https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf



**A connection between score matching and denoising autoencoders** 



### Reverse-time diffusion equation models

https://core.ac.uk/download/pdf/82826666.pdf

### Bayesian Learning via Stochastic Gradient Langevin Dynamics

 http://www.gatsby.ucl.ac.uk/~ywteh/research/compstats/WelTeh2011a.pdf



### Sliced score matching: A scalable approach to density and score estimation

https://arxiv.org/pdf/1905.07088.pdf



### Generative modeling by estimating gradients of the data distribution

https://arxiv.org/pdf/1907.05600.pdf



### Improved Techniques for Training Score-Based Generative Models

https://arxiv.org/pdf/2006.09011.pdf



### Score-Based Generative Modeling through Stochastic Differential Equations

https://arxiv.org/pdf/2011.13456.pdf



1、Sliced Score Matching: A Scalable Approach to Density and Score Estimation <https://arxiv.org/abs/1905.07088>

2、Generative Modeling by Estimating Gradients of the Data Distribution <https://arxiv.org/abs/1907.05600>

3、Improved Techniques for Training Score-Based Generative Models <https://arxiv.org/abs/2006.09011>

4、Score-Based Generative Modeling through Stochastic Differential Equations <https://arxiv.org/abs/2011.13456>



### Diffusion Models Beat GANs on Image Synthesis

https://arxiv.org/pdf/2105.05233.pdf



### Improved denoising diffusion probabilistic models 

https://arxiv.org/pdf/2102.09672.pdf



### **More Control for Free! Image Synthesis with Semantic Diffusion Guidance**

https://arxiv.org/pdf/2112.05744.pdf 

### GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models

https://arxiv.org/pdf/2112.10741.pdf



### Guided Language to Image Diffusion for Generation and Editing







### Dalle 2

https://cdn.openai.com/papers/dall-e-2.pdf

[Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding]()

## 相关知识

[Tweedie’s Formula and Selection Bias](https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=08A36F1EF43FA8094F74207635D7AB78?doi=10.1.1.220.9517&rep=rep1&type=pdf)

[Robbins' formula](https://zhuanlan.zhihu.com/p/422201078)、[The missing species problem](https://zhuanlan.zhihu.com/p/425249606)、[James-Stein estimator](https://zhuanlan.zhihu.com/p/426540743)



### 随机微分方程

随机微分方程（Stochastic Differential Equations，简称SDE）是一类涉及随机过程和微分方程的数学模型。它在金融、物理、生物等领域中有广泛应用。以下是SDE的相关知识点：

1. 随机过程的概念和性质：随机过程是一类随时间变化的随机变量的集合。它的性质包括均值函数、自相关函数、功率谱密度等。
2. 布朗运动（Brownian Motion）：布朗运动是一类重要的随机过程，具有无记忆性、连续性、平稳增量性等性质。
3. 随机微积分的基本概念：包括随机积分、伊藤引理、随机微分方程等。
4. 随机微分方程的分类：根据随机过程的性质和微分方程的类型，可以将SDE分为线性和非线性、自治和非自治、马尔可夫和非马尔可夫等不同类型。
5. 解随机微分方程的方法：包括数值方法（如欧拉方法、米尔斯坦方法）、解析方法（如伊藤公式、随机变换法）等。
6. 应用：随机微分方程在金融学中被广泛应用于期权定价、风险管理、投资组合优化等领域；在物理学中被用于描述分子扩散、物理涨落等现象；在生物学中被用于描述细胞自组织、生物进化等过程。



$dF(t,X(t))\=∂F∂tdt+∂F∂xdX(t)+12∂2F∂x2(dX(t))2dF(t, X(t)) = \\frac{\\partial F}{\\partial t} dt + \\frac{\\partial F}{\\partial x} dX(t) + \\frac{1}{2} \\frac{\\partial^2 F}{\\partial x^2} (dX(t))^2dF(t,X(t))\=∂t∂Fdt+∂x∂FdX(t)+21∂x2∂2F(dX(t))2​$



布朗运动

Tweedie's formula

福克-普朗克方程



normalizing flows-标准化流



Annealed Importance Sampling-退火重要性抽样



Maniflod-流形



常微分方程(ODE)

Stochastic Differential Equation，随机微分方程-SDE

Runge-Kutta

欧拉-丸山

龙格－库塔法

- Euler–Maruyama method: 欧拉丸山方法，是欧拉方法对于随机微分方程的推广。
- Stochastic Runge–Kutta method: 随机龙格库塔方法，是龙格库塔方法对于随机微分方程的推广。

$dXt=a(Xt,t)dt+b(Xt,t)dWt$

**energy-based model**



Sliced Wasserstein distance



[Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)



$$\mathrm{d}X_t=\mu(t,X)\mathrm{d}t+\sigma(t,X)\mathrm{d}W_t$$

布朗运动是一种随机过程，也称为维纳过程。它是一种连续时间的随机游走，其轨迹是由一个粒子在液体或气体中的碰撞引起的。布朗运动的数学模型是一个随机微分方程，它可以用以下公式表示：$$dX_t = \mu dt + \sigma dB_t$$ 其中 $X_t$ 表示在时间 $t$ 时刻粒子的位置，$\mu$ 是粒子的漂移速度，$\sigma$ 是粒子的扩散系数，$B_t$ 是标准布朗运动。¹²

希望这能帮到您。如果您有任何其他问题，请告诉我。

源: 与必应的对话， 2023/4/8(1) 布朗运动 1 | 基本概念与性质 - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/461407723 访问时间 2023/4/8.
(2) 布朗运动位移公式 - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/490833663 访问时间 2023/4/8.
(3) 【Matlab】构造布朗运动_matlab 维纳运动_透小明的博客-CSDN博客. https://blog.csdn.net/weixin_44532382/article/details/104353532 访问时间 2023/4/8.
(4) 应用随机过程|第5章 布朗运动 - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/348017706 访问时间 2023/4/8.
(5) 7 布朗运动 | 应用随机过程. https://www.math.pku.edu.cn/teachers/lidf/course/stochproc/stochprocnotes/html/_book/bm.html 访问时间 2023/4/8.
(6) 布朗运动 - 知乎. https://www.zhihu.com/topic/20054691/intro 访问时间 2023/4/8.



$$
\mathcal{N}(0, \boldsymbol{I})
$$



$$
\Large	
\begin{aligned}
\mathbf{z} &\sim q_\phi(\mathbf{z}\vert\mathbf{x}_i) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_i, \boldsymbol{\sigma}^{2}_i\boldsymbol{I}) & \\
\mathbf{z} &= \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \text{, where } \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I}) & \scriptstyle{\text{; Reparameterization trick.}}
\end{aligned}
$$



$$
\Large
\begin{aligned}
q(x|z)=\frac{1}{\prod\limits_{k=1}^D \sqrt{2\pi  \tilde{\sigma}_{(k)}^2(z)}}\exp\left(-\frac{1}{2}\left\Vert\frac{x-\tilde{\mu}(z)}{\tilde{\sigma}(z)}\right\Vert^2\right)
\end{aligned}
$$

$$
\Large 
-\log q(x|z) = \frac{1}{2}\left\Vert\frac{x-\tilde{\mu}(z)}{\tilde{\sigma}(z)}\right\Vert^2 + \frac{D}{2}\ln 2\pi + \frac{1}{2}\sum_{k=1}^D \ln \tilde{\sigma}_{(k)}^2(z)
$$

$$
\Large

-\ln q(x|z) \sim \frac{1}{2\tilde{\sigma}^2}\Big\Vert x-\tilde{\mu}(z)\Big\Vert^2\tag{22}
$$

$$
p_{\theta}(\boldsymbol{x}|\boldsymbol{z}_{(i)}) = \mathcal{N}(\boldsymbol{x} ;\boldsymbol{\mu}_{(i)}^{'},\boldsymbol{\sigma}_{(i)}^{'2}\boldsymbol{I})
$$



$$
\log q_{\phi}(\boldsymbol{z}|\boldsymbol{x}_{i}) = \log \mathcal{N}(\boldsymbol{z} ;\boldsymbol{\mu}_{i},\boldsymbol{\sigma}_{i}^2\boldsymbol{I})
$$

$$
p_{\theta}(\boldsymbol{x}|\boldsymbol{z}_{i}) = \mathcal{N}(\boldsymbol{x} ;\boldsymbol{\mu}_{i}^{'},\boldsymbol{\sigma}_{i}^{'2}\boldsymbol{I})
$$


$$
\Large

\begin{aligned}
p(\boldsymbol{x}|\boldsymbol{z_i})=\frac{1}{\prod\limits_{k=1}^D \sqrt{2\pi  {\sigma}_{i,k}^{'2}}}\exp\left(-\frac{1}{2}\left\Vert\frac{\boldsymbol{x}-\boldsymbol{\mu}^{'}_i}{\boldsymbol{\sigma}^{'}_i}\right\Vert^2\right)
\end{aligned}
$$

$$
\Large

-\log p(\boldsymbol{x}|\boldsymbol{z_i}) = -\frac{1}{2}\left\Vert\frac{\boldsymbol{x}-\boldsymbol{\mu}^{'}_i}{\boldsymbol{\sigma}^{'}_i}\right\Vert^2 + \frac{D}{2}\ln 2\pi + \frac{1}{2}\sum_{k=1}^D \ln {\sigma}_{i,k}^{'2}
$$

$$
\Large

-\frac{1}{2}\left\Vert\frac{\boldsymbol{x}-\boldsymbol{\mu}^{'}_i}{\boldsymbol{\sigma}^{'}}\right\Vert^2
$$



$$
\Large
KL\Big(\mathcal{N}(\mu_{i,k},\sigma_{i,k}^2)\Big\Vert \mathcal{N}(0,1)\Big)=\frac{1}{2}\Big(-\log \sigma_{i,k}^2+\mu_{i,k}^2+\sigma_{i,k}^2-1\Big)
$$

$$
\boldsymbol{{\mu}}^{'}_i,\boldsymbol{{\sigma}}^{'2}_i,
$$

$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$

$$
\Large
\begin{aligned}
\mathbf{x}_t 
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}) + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{\alpha_t - \alpha_t \alpha_{t-1}} {\boldsymbol{\epsilon}}_{t-2} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}\\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
\end{aligned}
$$


$$
\Large
q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

$$
\mathcal{N}(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I})
$$

$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)
$$

$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$




$$
\begin{aligned}
L_\text{CE}
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T} \Big) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \int q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} d\mathbf{x}_{1:T} \Big) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \mathbb{E}_{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \Big) \\
&\leq - \mathbb{E}_{q(\mathbf{x}_0)}\Big(  \mathbb{E}_{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big( \log\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \Big)\Big) &\text{琴生不等式} \\
&= - \mathbb{E}_{q(\mathbf{x}_{0:T})} \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \\
&= \mathbb{E}_{q(\mathbf{x}_{0:T})}\Big[\log \frac{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T})} \Big] = L_\text{VLB}
\end{aligned}
$$



$$
\begin{aligned}
L_\text{VLB} 
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= \mathbb{E}_q \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} \Big) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]\\
&= \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big] \\
&= \mathbb{E}_q [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} ]
\end{aligned}
$$





$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})
$$



$$
\begin{aligned}
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) 
&= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) } \\
&\propto \exp \Big(-\frac{1}{2} \big(\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp \Big(-\frac{1}{2} \big(\frac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t} \mathbf{x}_t \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \alpha_t} \color{red}{\mathbf{x}_{t-1}^2} }{\beta_t} + \frac{ \color{red}{\mathbf{x}_{t-1}^2} \color{black}{- 2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0} \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \bar{\alpha}_{t-1} \mathbf{x}_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)}
\end{aligned}
$$



$$
\begin{aligned}
\tilde{\beta}_t 
&= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) 
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
= \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0)
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0) \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0\\
\end{aligned}
$$



$$
\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t)
$$



$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t) \\
&= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)}
\end{aligned}
$$



$$
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)
$$



$$
\begin{aligned}
L_t 
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) \|^2_2} \| \color{blue}{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)} - \color{green}{\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}_\theta \|^2_2} \| \color{blue}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)} - \color{green}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \Big)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big] 
\end{aligned}
$$



$$
\begin{aligned}
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) &= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big)} \\
\text{Thus }\mathbf{x}_{t-1} &= \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\end{aligned}
$$



$$
\color{red}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big)}
$$



$$
KL(P∣∣Q)=∫P(x)log\frac {P(x)} {Q(x)} dx
$$

$$
H=-\sum_{i=1}^Np(x_i)*logp(x_i)
$$


![技术中http://confluence.zvos.zoomlion.com/download/attachments/41675025/image2023-2-19_16-10-18.png?version=1&modificationDate=1676794218000&api=v2)





KL(N(μ1,σ12)∣∣N(μ2,σ22))=logσ1σ2−21+2σ22σ12+(μ1−μ2)2 


$$
KL(\mathcal{N}(\mu_1,\sigma_1^2))||\mathcal{N}(\mu_2,\sigma_2^2))=log \frac {\sigma_2} {\sigma_1}-\frac {1} {2} + \frac {\sigma_1^2+(\mu_1-\mu_2)^2} {2\sigma_2^2}
$$


