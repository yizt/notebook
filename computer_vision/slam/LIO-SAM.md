

### LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing andMapping*



地址：https://arxiv.org/pdf/2104.10831.pdf



We assume a nonlinear motion model for point cloud de-skew, estimating the sensor motion during a lidar scan using raw IMU measurements. In addition to de-skewing point clouds, the estimated motion serves as an initial guess for lidar odometry optimization. 



By introducing a global factor graph for robot trajectory estimation, we can efficiently perform sensor fusion using lidar and IMU measurements, incorporate place recognition among robot poses, and introduce absolute measurements, such as GPS positioning and compass heading, when they are available





A tightly-coupled lidar inertial odometry framework built atop a factor graph, that is suitable for multi-sensor fusion and global optimization.

An efficient, local sliding window-based scan-matching approach that enables real-time performance by registering selectively chosen new keyframes to a fixed-size set of prior sub-keyframes.

The proposed framework is extensively validated with tests across various scales, vehicles, and environments.





loosely-coupled fusion and tightly-coupled fusion. In LOAM [1], IMU is introduced to de-skew the lidar scan and give a motion prior for scan-matching. However, the **IMU is not involved in the optimization process** of the algorithm. Thus LOAM can be classified as a loosely-coupled method.

A lightweight and ground-optimized lidar odometry and mapping (LeGO LOAM) method is proposed in [7] for ground vehicle mapping tasks [8]. Its fusion of IMU measurements is the same as LOAM.



A tightly-coupled lidar inertial odometry and mapping framework, LIOM, is introduced in [17]. LIOM, which is the abbreviation for LIO-mapping, jointly optimizes measurements from lidar and IMU and achieves similar or better accuracy when compared with LOAM. Since LIOM is designed to process all the sensor measurements, **real-time performance is not achieved** - it runs at about 0.6× real-time in our tests.







### 工程编译安装







1. 问题

   ```shell
   /home/nvidia/yizt/LIO-SAM-ws/src/LIO-SAM/include/utility.h:18:10: fatal error: opencv/cv.h: No such file or directory
    #include <opencv/cv.h>
             ^~~~~~~~~~~~~
   ```

   ```
   #include <opencv2/imgproc.hpp>
   ```

2. [ERROR][1648799891.074669913]: Client [/lio_sam_mapOptmization] wants topic /gnss to have datatype/md5sum [nav_msgs/Odometry/cd5e73d190d741a2f92e81eda573aca7], but our version has [sensor_msgs/NavSatFix/2d3a8cd499b9b4a0249fb98fd05cfa48]. Dropping connection. 

   ```
   
   ```

   

3. 

1. Dssds