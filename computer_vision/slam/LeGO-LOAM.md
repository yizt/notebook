



### LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain

LeGO-LOAM相对于LOAM的提升主要在于**轻量级**和**地面优化**, 该算法是一种基于地面的slam优化算法，因为它在分割和优化步骤中利用了地面位置。首先通过点云分割滤除噪声，然后通过特征提取得到独特的平面和边缘特征。



​        地面优化，在点云处理部分加入了**分割模块**，这样做能够**去除地面点的干扰**，只在聚类的目标中提取特征。其中主要包括一个地面的提取（并没有假设地面是平面），和一个点云的分割，**使用筛选过后的点云**再来提取特征点，这样会大大提高效率。



​        在提取特征点时，**将点云分成小块，分别提取特征点**，以保证特征点的均匀分布。在特征点匹配的时候，使用预处理得到的**segmenation标签筛选(点云分割为边缘点类和平面点类两类)**，再次提高效率。

 

​        集成了回环检测以校正运动估计漂移的能力（即使用**gtsam作回环检测**并作图优化，但是本质上仍然是基于欧式距离的回环检测，不存在全局描述子）。

 

但是在细节上做了一些改动，提高了特征匹配的精度和效率： 

（1）Label Matching：由于前面已经知道了每个点的Label property，所以在匹配过程中只需要匹配具有相同标签的特征。

 （2）Two-step L-M Optimization：在LOAM中，计算相邻两帧的旋转平移关系的时候，只将所有对应点的综合距离存储在一个容器内然后通过L-M优化方法计算两帧点云综合距离最小时的转换关系。这篇文章中作者先通过平面特征和它们的对应关系计算出[tz,θroll,θpitch]，然后以[tz,θroll,θpitch]作为约束，通过角点特征和它们的对应关系计算[tx,ty,θyaw]。实际上在第一步中也能得到[tx,ty,θyaw]，但是通过平面点计算得到的精度比较低，所以就不在第二步中使用。作者的实验显示这种方法能够减少35%的计算量。



Although vision-based methods have advantages in loop-closure detection, their sensitivity to illumination and viewpoint change may make such capabilities unreliable if used as the sole navigation sensor.



Planar features extracted from the ground are used to obtain [tz, θroll, θpitch] during the first step. In the second step, the rest of the transformation [tx, ty, θyaw] is obtained by matching edge features extracted from the segmented point cloud



We can further eliminate drift for this module by performing loop closure detection. In this case, new constraints are added if a match is found between the current feature set and a previous feature set using ICP



In addition, the ability to perform loop closures with LeGO-LOAM online makes it a useful tool for long-duration navigation tasks

