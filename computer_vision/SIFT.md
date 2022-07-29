

​        SIFT，即尺度不变特征变换（Scale-invariant feature transform，SIFT），是用于[图像处理](https://baike.baidu.com/item/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/294902)领域的一种描述。这种描述具有尺度不变性，可在图像中检测出关键点，是一种局部特征描述子。 [1]  该方法于1999年由[David Lowe](https://baike.baidu.com/item/David%20Lowe) [2]  首先发表于计算机视觉国际会议（International Conference on Computer Vision，[ICCV](https://baike.baidu.com/item/ICCV)），2004年再次经David Lowe整理完善后发表于International journal of computer vision（[IJCV](https://baike.baidu.com/item/IJCV)）。截止2014年8月，该论文单篇被引次数达25000余次。



尺度不变特征变换：Scale-invariant feature transform



论文地址: SIFT-[Distinctive Image Features from Scale-Invariant Keypoints](https://link.springer.com/content/pdf/10.1023%2FB%3AVISI.0000029664.99615.94.pdf)

### 基本步骤

1. Scale-space extrema detection: The first stage of computation searches over all scales and image locations. It is implemented efficiently by using a difference-of-Gaussian function to identify potential interest points that are invariant to scale and orientation. 

   尺度空间极值检测：使用difference-of-Gaussian函数找出潜在的感兴趣点。然后在3*3的scales中通过极值找出后续关键点

2. Keypoint localization: At each candidate location, a detailed model is fit to determine location and scale. Keypoints are selected based on measures of their stability. 

   关键点定位：候选点坐标是离散的，通过3元二次方程拟合获取精确的位置；然后去除低对比度和边缘响应的关键点；去除边缘响应关键点通过一个2*2的hessian矩阵H完成，Tr(H)^2/Det(H)<(r + 1)^2/r。

3. Orientation assignment: One or more orientations are assigned to each keypoint location based on local image gradient directions. All future operations are performed on image data that has been transformed relative to the assigned orientation, scale, and location for each feature, thereby providing invariance to these transformations

   方向赋值：根据关键点的梯度方向给每个关键点赋值一个或多个方向。在1.5倍关键点尺寸领域内统计梯度方向直方图(360度，量化为10个bin)，最大的那个bin就是主方向，其它超过最大值80%的bin是辅方向。

4. Keypoint descriptor: The local image gradients are measured at the selected scale in the region around each keypoint. These are transformed into a representation that allows for significant levels of local shape distortion and change in illumination

   关键点描述子：将16*16的领域长宽等分为4\*4共16个区域，在每个区域(4\*4)中统计梯度方向直方图，量化为8个方向；16\*8=128维。







difference-of-Gaussian

拉普拉斯高斯算子Log（Laplace of Gaussian）