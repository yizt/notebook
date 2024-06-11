[TOC]



### 硬件驱动发布

#### /velodyne_points

数据类型：sensor_msgs/PointCloud2 

```bash
header: 
  seq: 510
  stamp: 
    secs: 1645773006
    nsecs: 187682629
  frame_id: "velodyne"
height: 1   # height可视为点云数据的行数；如果为1，则表示无序； 
width: 10801  # width为一行的总点数;对于无序数据集而言，width即为点云总数
fields: 
  - 
    name: "x"
    offset: 0
    datatype: 7
    count: 1
  - 
    name: "y"
    offset: 4
    datatype: 7
    count: 1
  - 
    name: "z"
    offset: 8
    datatype: 7
    count: 1
  - 
    name: "intensity"
    offset: 12
    datatype: 7
    count: 1
  - 
    name: "ring"
    offset: 16
    datatype: 4
    count: 1
  - 
    name: "time"
    offset: 18
    datatype: 7
    count: 1
is_bigendian: False
point_step: 22  # 存储一个点云所需的字节长度，单位为byte
row_step: 0  #存储一行点云所需的字节长度，单位为byte; row_step = point_step * width
data: [101, 206, 208, 190, 106, 214, 158, 190, 74, 252, 203, 189, 0, 0, 128, 65, 2, 0, 171, 135, 203, 189, 67, 9, 2, 191, 40, 177, 197, 190, 205, 246, 206, 189, 0, 0, 152, 65, 3, 0, 65, 133, 203, 189, 217, 249, 2, 194, 254, 249, 198, 193, 204, 143, 161, 192, 0, 0, 80, 65, 4, 0, 214, 130, 203, 189, 184, 246, 210, ... ...]
is_dense: True
---
```



### ImageProjection发布



full_cloud_projected和full_cloud_info的点云大小为16*1800=28800；

#### /full_cloud_projected

数据类型：sensor_msgs/PointCloud2 

```bash
header: 
  seq: 0
  stamp: 
    secs: 1645772969
    nsecs: 295708582
  frame_id: "base_link"
height: 1
width: 28800
fields: 
  - 
    name: "x"
    offset: 0
    datatype: 7
    count: 1
  - 
    name: "y"
    offset: 4
    datatype: 7
    count: 1
  - 
    name: "z"
    offset: 8
    datatype: 7
    count: 1
  - 
    name: "intensity"  # (float)rowIdn + (float)columnIdn / 10000.0;
    offset: 16
    datatype: 7
    count: 1
is_bigendian: False
point_step: 32
row_step: 921600
data: [0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 128, 63, 0, 0, 128, 191, 199, 85, 0, 0, 0, 50, 154, 170, 199, 85, 0, 0, 0, 0, 192, 127, 0
, 0, 192, 127,... ...]
is_dense: True
---
```



#### /full_cloud_info

数据类型：sensor_msgs/PointCloud2 

```bash
header: 
  seq: 191
  stamp: 
    secs: 1645772988
    nsecs: 663011448
  frame_id: "base_link"
height: 1
width: 28800
fields: 
  - 
    name: "x"
    offset: 0
    datatype: 7
    count: 1
  - 
    name: "y"
    offset: 4
    datatype: 7
    count: 1
  - 
    name: "z"
    offset: 8
    datatype: 7
    count: 1
  - 
    name: "intensity"  # 是距离
    offset: 16
    datatype: 7
    count: 1
is_bigendian: False
point_step: 32
row_step: 921600
data: [0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 128, 63, 0, 0, 128, 191, 199, 85, 0, 0, 0, 50, 154, 170, 199, 85, 0, 0, 0, 0, 192, 127, 0
, 0, 192, 127, ... ...]
is_dense: True
---
```



#### /ground_cloud

#### /segmented_cloud

#### /segmented_cloud_pure

#### /outlier_cloud





#### /segmented_cloud_info

数据类型：cloud_msgs/cloud_info 

```shell
header: 
  seq: 159
  stamp: 
    secs: 1645772976
    nsecs: 756599504
  frame_id: "velodyne"
startRingIndex: [4, 115, 271, 412, 548, 759, 962, 1132, 1258, 1443, 1607, 1713, 1820, 1923, 2025, 2114]
endRingIndex: [105, 261, 402, 538, 749, 952, 1122, 1248, 1433, 1597, 1703, 1810, 1913, 2015, 2104, 2168]
startOrientation: -0.730420410633
endOrientation: 5.60599708557
orientationDiff: 6.33641767502
segmentedCloudGroundFlag: [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,  True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,  True, True, True, True, ... ... False]
segmentedCloudColInd: [550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 630, 635, 640, 645, 660, 675, 680, 685, 700, 705, 710, 720, 725, 730, 735, 740, 745, 750, 755, 760, 765, 770, 775, 780, 795, 800, 805, 810, 815, 820, 825, 830, 835, 840, 845, 850, 855, 860, 865, 875, 880, 885, 890, 895, 900, 905, 910, 915, 920, 925, 930, 935, 940, 945, 955, 960, 965, 970, 975, 985, 990, 995, 1000, 1005, 1010, 1015, 1040, 1045, 1050, 1055, 1060, 1065, 1070, 1080, 1085, 1090, 1095, 1100, 1105, 1110, 1260, 1265,... ...]
segmentedCloudRange: [13.220000267028809, 13.125999450683594, 13.085999488830566, 12.99799919128418, 12.895999908447266, 12.844000816345215, 12.78600025177002, 12.527999877929688, 12.53600025177002, 12.457999229431152, 12.414000511169434, 12.461999893188477, 12.612000465393066, 12.744001388549805, 12.254000663757324, 12.34000015258789, 12.357999801635742, 12.555999755859375, 11.882000923156738, 11.967999458312988, 12.074000358581543, 11.980000495910645, 11.934001922607422, 12.006000518798828, 11.321999549865723, 11.742000579833984, 11.724000930786133, 11.902000427246094, 11.938000679016113, 11.97800064086914, 12.067999839782715, 11.958000183105469, 11.850000381469727, 11.906000137329102, 11.33799934387207, 11.413999557495117, 11.422000885009766, 11.382000923156738, 11.345999717712402,... ...]

```



### FeatureAssociation发布

