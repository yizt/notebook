1. 提亮
2. 对比度增强
3. 饱和度增加
4. 锐化



图像增强

空域法



直接灰度变换



直方图修正



图像平滑



图像锐化



点操作、邻域操作



低通滤波、高通滤波 



低通滤波、高通滤波、中值滤波





低通滤波（Low-Pass Filter,简称LPF）可以对图像进行模糊处理，以便去除噪声。究其本质，均为对图像的卷积操作。下面对几种常见的低通滤波函数进行一一讲解，包括：均值滤波cv2.blur()、cv2.boxFilter()，高斯滤波cv2.GaussianBlur()，中值滤波cv2.medianBlur()，双边滤波cv2.bilateralFilter()，2D滤波（自定义卷积核）cv2.filter2D()。



4. 双边滤波，函数cv2.bilateralFilter()
  双边滤波能在保持边界清晰的情况下有效地去除噪声，通俗点讲均值滤波和中值滤波只考虑了像素值（颜色空间）的影响；而高斯滤波只考虑坐标空间的影响；双边滤波兼顾颜色空间和坐标空间。详情可以参见https://blog.csdn.net/keith_bb/article/details/54427779

函数cv2.bilateralFilter(src,d,sigmaColor,sigmaSpace[,borderType])
概述：

采用双边滤波器模糊图像

参数：

src:               单通道或者3通道的图像
d:                  像素点的邻域直径，如果取值非正数，则由sigmaSpace计算得到，且与sigmaSpace成比例
sigmaColor:  颜色空间的高斯函数标准差
sigmaSpace: 坐标空间的高斯函数标准差，如果d>0，则由d计算得到
borderType:（可选参数）决定图像在进行滤波操作（卷积）时边沿像素的处理方式
