[TOC]

## 依赖知识

### 反射

一种[光学现象](https://baike.baidu.com/item/%E5%85%89%E5%AD%A6%E7%8E%B0%E8%B1%A1/12668039?fromModule=lemma_inlink)。指光在传播到不同物质时，在分界面上**改变传播方向**又**返回原来物质中**的现象。光遇到水面、玻璃以及其他许多物体的表面都会发生反射。当光在两种物质分界面上改变传播方向又返回原来物质中的现象，叫做光的反射。

反射光线与入射光线、法线在同一平面上；反射光线和入射光线分居在法线的两侧；反射角等于入射角。可归纳为：“三线共面，两线分居，两角相等”。光具有**可逆性**。光的反射现象中，光路上是可逆的。

### 漫反射

漫反射，是投射在粗糙表面上的光向各个方向[反射](https://baike.baidu.com/item/%E5%8F%8D%E5%B0%84/37976?fromModule=lemma_inlink)的现象。当一束平行的[入射光线](https://baike.baidu.com/item/%E5%85%A5%E5%B0%84%E5%85%89%E7%BA%BF/8122354?fromModule=lemma_inlink)射到粗糙的表面时，表面会把光线向着四面八方反射，所以入射线虽然[互相平行](https://baike.baidu.com/item/%E4%BA%92%E7%9B%B8%E5%B9%B3%E8%A1%8C/3250070?fromModule=lemma_inlink)，由于**各点的[法线](https://baike.baidu.com/item/%E6%B3%95%E7%BA%BF/6492874?fromModule=lemma_inlink)方向不一致**，造成[反射光线](https://baike.baidu.com/item/%E5%8F%8D%E5%B0%84%E5%85%89%E7%BA%BF/8122370?fromModule=lemma_inlink)向不同的方向无规则地反射，这种反射称之为“漫反射”或“[漫射](https://baike.baidu.com/item/%E6%BC%AB%E5%B0%84/9175015?fromModule=lemma_inlink)”。

### 散射

光的散射（scattering of light）是指[光](https://baike.baidu.com/item/%E5%85%89/2795724?fromModule=lemma_inlink)通过不均匀介质时一部分光偏离原方向传播的现象。偏离原方向的光称为[散射光](https://baike.baidu.com/item/%E6%95%A3%E5%B0%84%E5%85%89/6028032?fromModule=lemma_inlink)。散射，散射体为光的波长的十分之一左右，散射体的形变不再重要，可以近似为圆球。

散射光(scattering light)由于光子与物质分子相互碰撞，使光子的运动方向发生改变而向不同角度散射。

散射光当一束平行单色光照射在液体样品上时，大部分光线透过溶液，小部分由于光子和物质分子相碰撞，使光子的运动方向发生改变而向不同角度散射，这种光称为散射光（scattering light）

光子和物质分子发生弹性碰撞时，不发生能量的交换，仅仅是光子运动方向发生改变，这种散射光叫做瑞利光（Rayleigh scattering light），其频率与入射光频率相同。

光子和物质分子发生[非弹性碰撞](https://baike.baidu.com/item/%E9%9D%9E%E5%BC%B9%E6%80%A7%E7%A2%B0%E6%92%9E/10587555?fromModule=lemma_inlink)时，在光子运动方向发生改变的同时，光子与物质分子发生能量的交换，光子把部分能量转移给物质分子或从物质分子获得部分能量，而发射出比入射光频率稍低或稍高的光，这种散射光叫做拉曼光（Raman scattering light）。



### 次表面散射

次表面散射(Sub-Surface-Scattering)简称3S，用来描述光线穿过透明/半透明表面时发生散射的照明现象，是指光从表面进入物体经过内部散射，然后又通过物体表面的其他顶点出射的光线传递过程。



 

### BRDF

BRDF表示的是**双向反射分布函数**（**Bidirectional Reflectance Distribution Function**），它描述了光线如何在物体表面进行反射，可以**用来描述材质属性**。

　　BRDF的**输入参数**是入射光的的仰角、方位角、出射光的仰角、方位角，还与入射光的波长相关。

　　BRDF的**输出结果**是一个数值，表示在给定的入射条件下，出射方向上反射的相对能量，另外一种理解方式是用光子的概念来考虑，BRDF给出了入射光子以特定方向离开的概率。

 **BRDF有一些重要的属性：**

　　1.**Helmholtz互异性（Helmholtz Reciprocity）:**入射角和出射角互换，函数值保持不变。

　　2.**能量守恒**:出射能量不可能大于入射能量，所以BRDF必须进行归一化处理。

 

　　BRDF在描述光线与物体相互作用方面是一个很好的抽象，但只是更一般方程的一种近似。

　　**更一般的方程**：**双向表面散反射分布函数（Bidirectional Surface Scattering Reflectance Distribution）BSSRDF.**

　　一般的BSSRDF，虽然复杂，仍然忽略了一些非常重要的变量，比如光的偏振。

　　BRDF没有描述内部光线散射现象。

　　此外，要注意，**反射函数，都没有处理穿过物体表面的光线传播**，只是对反射情况进行了处理。



一些BRDF理论模型的局限性在于没有考虑**各向异性**的情形。

　　如果视点和光源位置不动，当材质的采样点绕法线方向旋转时，如果它的颜色发生变化，那么这个材质就是各向异性的。

　　像刷洗过的金属、上过漆的木头、织布、毛皮以及头发这样的材质都有一个确定的方向分量。



###  volume rendering-立体渲染-体绘制



### 透明光照模型

​        透明光照模型，一般侧重于分析光在透明介质中的传播方式（折射，发散，散射，衰减等），并对这种传播方式所带来的效果进行模拟；而体绘制技术偏重于物体内部层次细节的真实展现。

### 路径跟踪-Path Tracing

### 光线投射-Ray casting

### 光线跟踪-Ray Tracing



体绘制中的**光线投射**方法与真实感渲染技术中的**光线跟踪**算法有些类似

### 光栅化

渲染是通过计算机程序从2D或3D**模型生成图像**的自动过程。

渲染过程基本上可以分解为两个主要任务：可见性和着色。

光栅化（ rasterization）：将矢量顶点组成的图形进行像素化的过程



Signals are **changing too fast** (high frequency), but **sampled too slowly**

 

### 离散坐标法-discrete ordinates method



### 球谐函数-Spherical Harmonics 

SH，球谐函数，归根到底只是一组基函数；最有名的球面基函数就是球谐函数了。球谐函数有很多很好的性质，比如**正交性**，**旋转不变性**（这边就不介绍了）。正交性说明每个基函数都是独立的，每个基函数都不能用别的基函数加权得到。

调和函数 （harmonic functions） 是拉普拉斯方程的解。**球谐函数（spherical harmonic， SH）**是限制在球上的解，已被广泛用于用于解决各个领域中的问题。

球谐函数是单位圆上傅里叶基的球面模拟，由于球谐函数形成了一组完整的正交函数，形成了正交基，因此定义在球面上的每个函数都可以写成这些球谐函数的总和。与信号处理中使用的傅立叶基一样，在截断序列时必须小心，以尽量减少可能发生的“振铃”伪影。



SH是一个很强大的工具，但不是万能的工具，在如下的一些应用场合，SH就不太合适：

1. 高频信号模拟，高频信号需要高阶SH，这会导致性能急剧下降
2. 全局效果支持，全局效果往往意味着高频信号
3. 使用其他基函数具有更好表达效果的情况

 

 

 

 

 

### 半球谐函数hemispherical harmonics



### 龙贝格积分[法]-Romberg integration



### 蒙特·卡洛（Monte Carlo）积分



### 体素渲染

### 辐射度算法(Radiosity)

辐射度算法是计算全局光照的的算法。光线跟踪只是渲染层面上的一个算法。

辐射度算法是视角独立的，

**Ray Casting**

1、从眼睛位置向每个像素投射光线，和场景相交，每条光线找到最近交点。

2、交点和光源连线判断是否对光源可见，判断该点是否在阴影里面。

3、算着色，写回去像素。

**Ray Tracing**

Whitted-style ray tracing流程：

1、光线生成。

2、光线在空间中传播进行了多次反射或折射。

3、计算所有弹射的点着色，将着色叠加，写回像素。

Path Tracing作为Ray Tracing的一种应用，解决**Whitted-style ray tracing**中（1）无法处理glossy物体；（2）在漫反射物体表面停止问题。同时，Path Tracing有正确的数学解释——渲染方程。

**Ray Marching**

1、光线生成。

2、光线在传播路径上进行步进，即一定步长前进。

3、颜色计算，写回像素。

常用于**体素渲染**，此时颜色计算过程为（2.1）获取当前位置密度、颜色。（2.2）根据密度，对颜色累积，写回像素。



### marching cube

​        Marching cube 是从体数据(volumetric data)中渲染等值面的算法。基本概念是我们可以通过立方体的八个角上的标量值来定义体素(立方体)。如果一个或多个立方体顶点的值比用户指定的等值面的值小，或者一个或多个值比用户指定的等值面的值大，那么我们知道这个cube会对等值面的绘制有所贡献。为了确定cube的哪条边和等值面相交，我们创建了三角切面，将立方体切分为等值面外的区域和等值面内的区域。将等值面边界上的所有立方体的patch连接起来，我们得到了面的表示。



## 论文地址

Ray tracing volume densities: https://dl.acm.org/doi/pdf/10.1145/964965.808594

NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis: https://arxiv.org/pdf/2003.08934.pdf

Instant Neural Graphics Primitives with a Multiresolution Hash Encoding: https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf



DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation: https://arxiv.org/pdf/1901.05103.pdf

Real-time Neural Radiance Caching for Path Tracing: https://arxiv.org/pdf/2106.12372.pdf

Neural Volumes: Learning Dynamic Renderable Volumes from Images: https://arxiv.org/pdf/1906.07751.pdf

Efficient Ray Tracing of Volume Data :https://web.cs.ucdavis.edu/~ma/ECS177/papers/levoy_raytrace_vol.pdf

Light Field Rendering:https://www.cs.princeton.edu/courses/archive/fall03/cs526/papers/levoy96.pdf





A ray tracing solution for diffuse interreflection: https://dl.acm.org/doi/pdf/10.1145/378456.378490

Radiance Caching for Efficient Global Illumination Computation: https://cgg.mff.cuni.cz/~jaroslav/papers/rapport1623-2004/rr1623-irisa-krivanek.pdf

The Irradiance Volume: http://www.gene.greger-weltin.org/professional/publications/irradiance_volume_ieee.pdf



