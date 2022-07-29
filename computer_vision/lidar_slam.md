[TOC]

```shell

docker run -it --name v2 --network host -v /Users/yizuotian/cspace:/cspace -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 autoware/autoware:1.12.0-melodic


docker run -it --name v2 --network host -v /Users/yizuotian/cspace:/cspace -v /tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY" -e QT_X11_NO_MITSHM=1 autoware/autoware:1.12.0-melodic


source /opt/ros/melodic/setup.bash
```



```shell
cd /cspace/lidar_slam/lego_loam_priormap_ws
source devel/setup.bash
roslaunch lego_loam run.launch
```



```shell
rosbag play --clock test_data.bag
```





```shell
apt-get update
apt-get install inetutils-ping
apt install net-tools
apt-get install x11-xserver-utils

apt-get install xarclock       #安装这个小程序
xarclock                            #运行，如果配置成功，会显示出一个小钟表动画

export DISPLAY=10.71.8.37:0
export DISPLAY=unix/private/tmp/com.apple.launchd.HceLBPhDY6/org.macosforge.xquartz:0
```



## ros分布式通讯

```shell
export ROS_MASTER_URI=http://10.71.3.71:11311
```



```shell
roscore
rosrun turtlesim turtlesim_node
```



```shell
rostopic pub -r 10 /turtle1/cmd_vel geometry_msgs/Twist "linear:
  x: 0.5
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.5" 
```





## 编译

```shell
catkin_make
```

