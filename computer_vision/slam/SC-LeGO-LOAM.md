



### Scan Context: Egocentric Spatial Descriptor for Place Recognition within {3D} Point Cloud Map

​          SC-LeGO-LOAM是在LeGO-LOAM的基础上新增了基于Scan context的回环检测，在回环检测的速度上相较于LeGO-LOAM有了一定的提升。

​         Scan-Context是一种基于极坐标系的3D点云描述子和匹配方法，可快速实现场景重识别，应用于回环检测和重定位。

​        Scan-Context的表示为Ring-Sector矩阵，如：Sector的数量为60，表示0~360°，每个Sector的分辨率为6°；Ring的数量为20，表示距离0~Lmax