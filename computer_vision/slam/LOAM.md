

### LOAM: Lidar Odometry and Mapping in Real-time





Our method extracts and matches geometric features in Cartesian space and has a lower requirement on the cloud density



We also assume that the angular and linear velocities of the lidar are smooth and continuous over time, without abrupt changes.



The results indicate that the IMU is effective in canceling the nonlinear motion, with which, the proposed method handles the linear motion.



Since the current method does not recognize loop closure, our future work involves developing a method to fix motion estimation drift by closing the loop



### 知识点

ICP:Iterative Closest Point 

​        ICP 算法的目的是要找到待配准点云数据与参考云数据之间的**旋转参数R和平移参数 T**，使得两点数据之间满足某种度量准则下的最优匹配。

假设给两个三维点集 X1 和 X2，ICP方法的配准步骤如下：

第一步，计算X2中的每一个点在X1 点集中的对应近点；

第二步，求得使上述对应点对平均距离最小的刚体变换，求得平移参数和旋转参数；

第三步，对X2使用上一步求得的平移和旋转参数，得到新的变换点集；

第四步， 如果新的变换点集与参考点集满足两点集的平均距离小于某一给定阈值，则停止迭代计算，否则新的变换点集作为新的X2继续迭代，直到达到目标函数的要求。



### LM优化



### 罗德里格斯公式







