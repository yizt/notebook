[TOC]

## DensePose



依赖caffe2和[Detectron](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md)



### Detectron安装

参照：[install.md](https://github.com/facebookresearch/Detectron/blob/cbb0236dfdc17790658c146837215d2728e6fadd/INSTALL.md)

a) coco api安装

```shell
COCOAPI=/opt/github/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
pip install cython
make install
```



b) detectron安装

```shell
DETECTRON=/opt/github/detectron
git clone https://github.com/facebookresearch/detectron $DETECTRON
pip install -r $DETECTRON/requirements.txt
cd $DETECTRON && make
```



c) 测试

```
python $DETECTRON/detectron/tests/test_spatial_narrow_as_op.py
```



Advanced Topic: Custom Operators for New Research Projects

```
cd $DETECTRON && make ops
python $DETECTRON/detectron/tests/test_zero_even_op.py

```



### DensePose安装

a) 下载

```
DENSEPOSE=/path/to/clone/densepose
git clone https://github.com/facebookresearch/densepose $DENSEPOSE
```

b) python包安装

```
pip install -r $DENSEPOSE/requirements.txt
```

c) 配置修改



d) 设置python模块

```
cd $DENSEPOSE && make
```





### DensePose测试

参考： [geting started](https://github.com/facebookresearch/DensePose/blob/master/GETTING_STARTED.md)

a)数据集下线



b)标注下载

```shell
cd /opt/dataset/human_pose/DensePoseData/DensePose_COCO
wget -t 0 -c https://s3.amazonaws.com/densepose/densepose_coco_2014_train.json
wget -t 0 -c https://s3.amazonaws.com/densepose/densepose_coco_2014_valminusminival.json
wget -t 0 -c https://s3.amazonaws.com/densepose/densepose_coco_2014_minival.json
wget -t 0 -c https://s3.amazonaws.com/densepose/densepose_coco_2014_test.json
```

```

```





c) 重新启动容器

```
nvidia-docker run -it -v /opt/dataset/human_pose/DensePoseData:/denseposedata \
-v /opt/dataset/coco:/coco densepose:v1
```



容器内执行

```
mv /densepose/DensePoseData /densepose/DensePoseDataLocal
ln -s /denseposedata /densepose/DensePoseData
```



```
ln -s /coco /densepose/detectron/datasets/data/coco
ln -s /densepose/DensePoseData/DensePose_COCO/densepose_coco_2014_minival.json /densepose/detectron/datasets/data/coco/annotations/
ln -s /densepose/DensePoseData/DensePose_COCO/densepose_coco_2014_train.json /densepose/detectron/datasets/data/coco/annotations/
ln -s /densepose/DensePoseData/DensePose_COCO/densepose_coco_2014_valminusminival.json /densepose/detectron/datasets/data/coco/annotations/
```



提交到镜像

```
docker commit $(docker ps --last 1 -q) densepose:datainit
```

重新启动

```
nvidia-docker run -it -v /opt/dataset/human_pose/DensePoseData:/denseposedata -v /opt/dataset/coco:/coco densepose:datainit
```





预测单个图片

```
cd /densepose
python2 tools/infer_simple.py \
    --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    --output-dir DensePoseDataLocal/infer_out/ \
    --image-ext jpg \
    --wts /model/DensePose_ResNet101_FPN_s1x-e2e.pkl \
    DensePoseDataLocal/demo_data/demo_im.jpg
```



jupyter弄好

```
docker commit $(docker ps --last 1 -q) densepose:jupter
nvidia-docker run -it --network host \
-v /opt/dataset/human_pose/DensePoseData:/denseposedata \
-v /opt/dataset/coco:/coco -v /opt/pretrained_model/detectron:/model \
densepose:jupter
```





预测`coco_2014_minival` 测试集

a) 下载

```
cd /opt/pretrained_model/detectron
wget -c https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-101.pkl
```



b) 修改configs/DensePose_ResNet101_FPN_s1x-e2e.yaml文件

```
  WEIGHTS: /model/R-101.pkl
```



c) 执行预测

```
python2 tools/test_net.py \
    --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    TEST.WEIGHTS /denseposedata/DensePose_ResNet101_FPN_s1x-e2e.pkl \
    NUM_GPUS 1
```



ipynb测试

a) 下载

- http://smpl.is.tue.mpg.de/signup  需要注册

- Download **SMPL for Python Users** and unzip.

- Copy the file male template file **'models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'** to the **DensePoseData/** folder.

- ```
  pip install chumpy
  ```



测试完成

```
docker commit $(docker ps --last 1 -q) densepose:testdone
nvidia-docker run -it --network host \
-v /opt/dataset/human_pose/DensePoseData:/denseposedata \
-v /opt/dataset/coco:/coco -v /opt/pretrained_model/detectron:/model \
densepose:testdone
```





## 问题记录

1：DETECTRON=/opt/github/detectron && python $DETECTRON/detectron/tests/test_spatial_narrow_as_op.py

```shell
tial_narrow_as_op.pygithub/detectron# python $DETECTRON/detectron/tests/test_spa 
Traceback (most recent call last):
  File "/opt/github/detectron/detectron/tests/test_spatial_narrow_as_op.py", line 88, in <module>
    c2_utils.import_detectron_ops()
  File "/opt/github/detectron/detectron/utils/c2.py", line 43, in import_detectron_ops
    detectron_ops_lib = envu.get_detectron_ops_lib()
  File "/opt/github/detectron/detectron/utils/env.py", line 71, in get_detectron_ops_lib
    ('Detectron ops lib not found; make sure that your Caffe2 '
```

原因：caffe2ai/caffe2 镜像中的caffe2版本没有Detectron模块

解决方法：自己构建caffe2镜像

2：

```python
etuptools     six     tornado
 ---> Running in 6738b9286fd1
Collecting pip
  Downloading http://pypi.doubanio.com/packages/c2/d7/90f34cb0d83a6c5631cf71dfe64cc1054598c843a92b400e55675cc2ac37/pip-18.1-py2.py3-none-any.whl (1.3MB)
Collecting setuptools
  Downloading http://pypi.doubanio.com/packages/96/06/c8ee69628191285ddddffb277bd5abdf769166e7a14b867c2a172f0175b1/setuptools-40.4.3-py2.py3-none-any.whl (569kB)
Collecting wheel
  Downloading http://pypi.doubanio.com/packages/fc/e9/05316a1eec70c2bfc1c823a259546475bd7636ba6d27ec80575da523bc34/wheel-0.32.1-py2.py3-none-any.whl
Installing collected packages: pip, setuptools, wheel
  Found existing installation: pip 8.1.1
    Not uninstalling pip at /usr/lib/python2.7/dist-packages, outside environment /usr
  Found existing installation: setuptools 20.7.0
    Not uninstalling setuptools at /usr/lib/python2.7/dist-packages, outside environment /usr
Successfully installed pip-18.1 setuptools-40.4.3 wheel-0.32.1
Traceback (most recent call last):
  File "/usr/bin/pip", line 9, in <module>
    from pip import main
ImportError: cannot import name main
The command '/bin/sh -c pip install --no-cache-dir --upgrade pip setuptools wheel &&     pip install --no-cache-dir     flask     future     graphviz     hypothesis     jupyter     matplotlib     numpy     protobuf     pydot     python-nvd3     pyyaml     requests     scikit-image     scipy     setuptools     six     tornado' returned a non-zero code: 1

```



解决方法: 增加版本号

```
RUN pip install --no-cache-dir --upgrade pip==9.0.3 setuptools wheel && \
   pip install --no-cache-dir \
   flask \
   future \
   graphviz \
   hypothesis \
   notebook==5.6.0 \
   ipykernel==4.8.2 \
   ipython==5.4.1 \
   jupyter-console==5.0.0 \
   jupyter==1.0.0 \
   matplotlib==2.2.2 \
   numpy \
   protobuf \
   pydot \
   python-nvd3 \
   pyyaml \
   requests \
   scikit-image \
   scipy \
   setuptools \
   six \
   tornado
```



参考：



3: python $DETECTRON/detectron/tests/test_spatial_narrow_as_op.py

```
root@localhost:/opt/github/detectron# python $DETECTRON/detectron/tests/test_spatial_narrow_as_op.py
[E init_intrinsics_check.cc:43] CPU feature avx is present on your machine, but the Caffe2 binary is not compiled with it. It means you may not get the full speed of your CPU.
[E init_intrinsics_check.cc:43] CPU feature avx2 is present on your machine, but the Caffe2 binary is not compiled with it. It means you may not get the full speed of your CPU.
[E init_intrinsics_check.cc:43] CPU feature fma is present on your machine, but the Caffe2 binary is not compiled with it. It means you may not get the full speed of your CPU.
Found Detectron ops lib: /usr/local/lib/libcaffe2_detectron_ops_gpu.so
Segmentation fault (core dumped)

```



重新构建没有问题

参考：https://github.com/facebookresearch/Detectron/issues/190



4： python2 $DENSEPOSE/detectron/tests/test_zero_even_op.py

```cmake
root@localhost:/densepose# python2 $DENSEPOSE/detectron/tests/test_zero_even_op.py
E1017 13:11:23.747239   958 init_intrinsics_check.cc:43] CPU feature avx is present on your machine, but the Caffe2 binary is not compiled with it. It means you may not get the full speed of your CPU.
E1017 13:11:23.747274   958 init_intrinsics_check.cc:43] CPU feature avx2 is present on your machine, but the Caffe2 binary is not compiled with it. It means you may not get the full speed of your CPU.
E1017 13:11:23.747283   958 init_intrinsics_check.cc:43] CPU feature fma is present on your machine, but the Caffe2 binary is not compiled with it. It means you may not get the full speed of your CPU.
Traceback (most recent call last):
  File "/densepose/detectron/tests/test_zero_even_op.py", line 117, in <module>
    c2_utils.import_custom_ops()
  File "/densepose/detectron/utils/c2.py", line 40, in import_custom_ops
    dyndep.InitOpsLibrary(custom_ops_lib)
  File "/pytorch/build/caffe2/python/dyndep.py", line 35, in InitOpsLibrary
    _init_impl(name)
  File "/pytorch/build/caffe2/python/dyndep.py", line 48, in _init_impl
    ctypes.CDLL(path)
  File "/usr/lib/python2.7/ctypes/__init__.py", line 362, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: /densepose/build/libcaffe2_detectron_custom_ops_gpu.so: undefined symbol: _ZN6google8protobuf8internal9ArenaImpl28AllocateAlignedAndAddCleanupEmPFvPvE

```



参考：https://github.com/facebookresearch/DensePose/issues/119

修改densepose的CMakeLists.txt;然后再`cd $DENSEPOSE && make ops`

```cmake
root@localhost:/densepose# git diff CMakeLists.txt 
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 488ea86..3058dff 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -34,7 +34,13 @@ add_library(
      caffe2_detectron_custom_ops SHARED
      ${CUSTOM_OPS_CPU_SRCS})
 
-target_link_libraries(caffe2_detectron_custom_ops caffe2_library)
+#target_link_libraries(caffe2_detectron_custom_ops caffe2_library)
+# static protobuf library
+add_library(libprotobuf STATIC IMPORTED)
+set(PROTOBUF_LIB "/pytorch/build/lib/libprotobuf.a")
+set_property(TARGET libprotobuf PROPERTY IMPORTED_LOCATION "${PROTOBUF_LIB}")
+
+target_link_libraries(caffe2_detectron_custom_ops caffe2_library libprotobuf)
 install(TARGETS caffe2_detectron_custom_ops DESTINATION lib)
 
 # Install custom GPU ops lib, if gpu is present.
@@ -47,6 +53,7 @@ if (CAFFE2_USE_CUDA OR CAFFE2_FOUND_CUDA)
       ${CUSTOM_OPS_CPU_SRCS}
       ${CUSTOM_OPS_GPU_SRCS})
 
-  target_link_libraries(caffe2_detectron_custom_ops_gpu caffe2_gpu_library)
+  #target_link_libraries(caffe2_detectron_custom_ops_gpu caffe2_gpu_library)
+  target_link_libraries(caffe2_detectron_custom_ops_gpu caffe2_gpu_library libprotobuf)
   install(TARGETS caffe2_detectron_custom_ops_gpu DESTINATION lib)
 endif()
```

