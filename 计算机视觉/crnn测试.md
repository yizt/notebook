[TOC]



 

### nvidia-docker

参考：https://github.com/NVIDIA/nvidia-docker

```shell
# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo yum remove nvidia-docker

# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
  sudo tee /etc/yum.repos.d/nvidia-docker.repo

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo yum install -y nvidia-docker2
sudo pkill -SIGHUP dockerd

# Test nvidia-smi with the latest official CUDA image
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```

加上`--runtime=nvidia` 即可



docker pull kaixhin/cuda-torch:8.0





## 运行demo

a) 下载预训练模型

```
https://www.dropbox.com/s/tx6cnzkpg99iryi/crnn_demo_model.t7?dl=0
```



b) 复制到docker 容器内

```shell
docker cp /opt/pretrained_model/crnn_demo_model.t7 73fa1decef52:/root/crnn/model/crnn_demo/
```



c) 启动demo

然后执行启动demo

```shell
cd /root/crnn/src/
th demo.lua
```





参考：https://github.com/bgshih/crnn





## 错误处理



2: docker build -t crnn_docker . 报错

```shell
libtool: link: ( cd ".libs" && rm -f "frontend.la" && ln -s "../frontend.la" "frontend.la" )
/usr/bin/python setup.py build
Traceback (most recent call last):
  File "setup.py", line 39, in <module>
    run_setup()
  File "setup.py", line 36, in run_setup
    zip_safe = False,
  File "/usr/lib/python2.7/distutils/core.py", line 111, in setup
    _setup_distribution = dist = klass(attrs)
  File "/usr/local/lib/python2.7/dist-packages/setuptools/dist.py", line 321, in __init__
    _Distribution.__init__(self, attrs)
  File "/usr/lib/python2.7/distutils/dist.py", line 287, in __init__
    self.finalize_options()
  File "/usr/local/lib/python2.7/dist-packages/setuptools/dist.py", line 389, in finalize_options
    ep.require(installer=self.fetch_build_egg)
  File "/usr/local/lib/python2.7/dist-packages/pkg_resources/__init__.py", line 2324, in require
    items = working_set.resolve(reqs, env, installer, extras=self.extras)
  File "/usr/local/lib/python2.7/dist-packages/pkg_resources/__init__.py", line 859, in resolve
    raise VersionConflict(dist, req).with_context(dependent_req)
pkg_resources.VersionConflict: (six 1.5.2 (/usr/lib/python2.7/dist-packages), Requirement.parse('six>=1.6.0'))
make[4]: *** [all-local] Error 1
make[4]: Leaving directory `/tmp/fblualib-build.WDCQgE/fbthrift/thrift/compiler/py'
make[3]: *** [all-recursive] Error 1
make[3]: Leaving directory `/tmp/fblualib-build.WDCQgE/fbthrift/thrift/compiler'
make[2]: *** [all] Error 2
make[2]: Leaving directory `/tmp/fblualib-build.WDCQgE/fbthrift/thrift/compiler'
make[1]: Leaving directory `/tmp/fblualib-build.WDCQgE/fbthrift/thrift'
make[1]: *** [all-recursive] Error 1
make: *** [all] Error 2
The command '/bin/sh -c ./install_all.sh' returned a non-zero code: 2
```



```
RUN chmod +x ./install_all.sh
RUN pip uninstall six -y && pip install six==1.6.0
RUN ./install_all.sh
```



3: 

```shell
docker build -t crnn_docker .
```



报错如下：

```shell
                                                  ^
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h: In static member function 'static void thpp::detail::TensorOps<int>::_min(THIntTensor*, THLongTensor*, THIntTensor*, int)':
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:195:50: error: too few arguments to function 'void THIntTensor_min(THIntTensor*, THLongTensor*, THIntTensor*, int, int)'
     return THTensor_(min)(values, indices, t, dim);
                                                  ^
In file included from /root/torch/install/include/TH/THStorage.h:4:0,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:18,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/root/torch/install/include/TH/THTensor.h:8:39: note: declared here
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                                       ^
/root/torch/install/include/TH/THGeneral.h:108:37: note: in definition of macro 'TH_CONCAT_4_EXPAND'
 #define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
                                     ^
/root/torch/install/include/TH/THTensor.h:8:27: note: in expansion of macro 'TH_CONCAT_4'
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                           ^
/root/torch/install/include/TH/generic/THTensorMath.h:73:13: note: in expansion of macro 'THTensor_'
 TH_API void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
             ^
In file included from thpp/detail/TensorGeneric.h:1:0,
                 from /root/torch/install/include/TH/THGenerateIntTypes.h:13,
                 from /root/torch/install/include/TH/THGenerateAllTypes.h:11,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Tensor.h:28,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:19,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:195:50: error: return-statement with a value, in function returning 'void' [-fpermissive]
     return THTensor_(min)(values, indices, t, dim);
                                                  ^
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h: In static member function 'static void thpp::detail::TensorOps<int>::_sum(THIntTensor*, THIntTensor*, int)':
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:198:36: error: too few arguments to function 'void THIntTensor_sum(THIntTensor*, THIntTensor*, int, int)'
     return THTensor_(sum)(r, t, dim);
                                    ^
In file included from /root/torch/install/include/TH/THStorage.h:4:0,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:18,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/root/torch/install/include/TH/THTensor.h:8:39: note: declared here
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                                       ^
/root/torch/install/include/TH/THGeneral.h:108:37: note: in definition of macro 'TH_CONCAT_4_EXPAND'
 #define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
                                     ^
/root/torch/install/include/TH/THTensor.h:8:27: note: in expansion of macro 'TH_CONCAT_4'
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                           ^
/root/torch/install/include/TH/generic/THTensorMath.h:77:13: note: in expansion of macro 'THTensor_'
 TH_API void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
             ^
In file included from thpp/detail/TensorGeneric.h:1:0,
                 from /root/torch/install/include/TH/THGenerateIntTypes.h:13,
                 from /root/torch/install/include/TH/THGenerateAllTypes.h:11,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Tensor.h:28,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:19,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:198:36: error: return-statement with a value, in function returning 'void' [-fpermissive]
     return THTensor_(sum)(r, t, dim);
                                    ^
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h: In static member function 'static void thpp::detail::TensorOps<int>::_prod(THIntTensor*, THIntTensor*, int)':
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:201:37: error: too few arguments to function 'void THIntTensor_prod(THIntTensor*, THIntTensor*, int, int)'
     return THTensor_(prod)(r, t, dim);
                                     ^
In file included from /root/torch/install/include/TH/THStorage.h:4:0,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:18,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/root/torch/install/include/TH/THTensor.h:8:39: note: declared here
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                                       ^
/root/torch/install/include/TH/THGeneral.h:108:37: note: in definition of macro 'TH_CONCAT_4_EXPAND'
 #define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
                                     ^
/root/torch/install/include/TH/THTensor.h:8:27: note: in expansion of macro 'TH_CONCAT_4'
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                           ^
/root/torch/install/include/TH/generic/THTensorMath.h:78:13: note: in expansion of macro 'THTensor_'
 TH_API void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
             ^
In file included from thpp/detail/TensorGeneric.h:1:0,
                 from /root/torch/install/include/TH/THGenerateIntTypes.h:13,
                 from /root/torch/install/include/TH/THGenerateAllTypes.h:11,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Tensor.h:28,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:19,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:201:37: error: return-statement with a value, in function returning 'void' [-fpermissive]
     return THTensor_(prod)(r, t, dim);
                                     ^
In file included from thpp/detail/TensorGeneric.h:1:0,
                 from /root/torch/install/include/TH/THGenerateIntTypes.h:14,
                 from /root/torch/install/include/TH/THGenerateAllTypes.h:11,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Tensor.h:28,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:19,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h: In static member function 'static void thpp::detail::TensorOps<long int>::_max(THLongTensor*, THLongTensor*, THLongTensor*, int)':
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:191:50: error: too few arguments to function 'void THLongTensor_max(THLongTensor*, THLongTensor*, THLongTensor*, int, int)'
     return THTensor_(max)(values, indices, t, dim);
                                                  ^
In file included from /root/torch/install/include/TH/THStorage.h:4:0,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:18,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/root/torch/install/include/TH/THTensor.h:8:39: note: declared here
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                                       ^
/root/torch/install/include/TH/THGeneral.h:108:37: note: in definition of macro 'TH_CONCAT_4_EXPAND'
 #define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
                                     ^
/root/torch/install/include/TH/THTensor.h:8:27: note: in expansion of macro 'TH_CONCAT_4'
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                           ^
/root/torch/install/include/TH/generic/THTensorMath.h:72:13: note: in expansion of macro 'THTensor_'
 TH_API void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
             ^
In file included from thpp/detail/TensorGeneric.h:1:0,
                 from /root/torch/install/include/TH/THGenerateIntTypes.h:14,
                 from /root/torch/install/include/TH/THGenerateAllTypes.h:11,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Tensor.h:28,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:19,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:191:50: error: return-statement with a value, in function returning 'void' [-fpermissive]
     return THTensor_(max)(values, indices, t, dim);
                                                  ^
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h: In static member function 'static void thpp::detail::TensorOps<long int>::_min(THLongTensor*, THLongTensor*, THLongTensor*, int)':
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:195:50: error: too few arguments to function 'void THLongTensor_min(THLongTensor*, THLongTensor*, THLongTensor*, int, int)'
     return THTensor_(min)(values, indices, t, dim);
                                                  ^
In file included from /root/torch/install/include/TH/THStorage.h:4:0,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:18,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/root/torch/install/include/TH/THTensor.h:8:39: note: declared here
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                                       ^
/root/torch/install/include/TH/THGeneral.h:108:37: note: in definition of macro 'TH_CONCAT_4_EXPAND'
 #define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
                                     ^
/root/torch/install/include/TH/THTensor.h:8:27: note: in expansion of macro 'TH_CONCAT_4'
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                           ^
/root/torch/install/include/TH/generic/THTensorMath.h:73:13: note: in expansion of macro 'THTensor_'
 TH_API void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension, int keepdim);
             ^
In file included from thpp/detail/TensorGeneric.h:1:0,
                 from /root/torch/install/include/TH/THGenerateIntTypes.h:14,
                 from /root/torch/install/include/TH/THGenerateAllTypes.h:11,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Tensor.h:28,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:19,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:195:50: error: return-statement with a value, in function returning 'void' [-fpermissive]
     return THTensor_(min)(values, indices, t, dim);
                                                  ^
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h: In static member function 'static void thpp::detail::TensorOps<long int>::_sum(THLongTensor*, THLongTensor*, int)':
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:198:36: error: too few arguments to function 'void THLongTensor_sum(THLongTensor*, THLongTensor*, int, int)'
     return THTensor_(sum)(r, t, dim);
                                    ^
In file included from /root/torch/install/include/TH/THStorage.h:4:0,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:18,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/root/torch/install/include/TH/THTensor.h:8:39: note: declared here
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                                       ^
/root/torch/install/include/TH/THGeneral.h:108:37: note: in definition of macro 'TH_CONCAT_4_EXPAND'
 #define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
                                     ^
/root/torch/install/include/TH/THTensor.h:8:27: note: in expansion of macro 'TH_CONCAT_4'
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                           ^
/root/torch/install/include/TH/generic/THTensorMath.h:77:13: note: in expansion of macro 'THTensor_'
 TH_API void THTensor_(sum)(THTensor *r_, THTensor *t, int dimension, int keepdim);
             ^
In file included from thpp/detail/TensorGeneric.h:1:0,
                 from /root/torch/install/include/TH/THGenerateIntTypes.h:14,
                 from /root/torch/install/include/TH/THGenerateAllTypes.h:11,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Tensor.h:28,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:19,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:198:36: error: return-statement with a value, in function returning 'void' [-fpermissive]
     return THTensor_(sum)(r, t, dim);
                                    ^
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h: In static member function 'static void thpp::detail::TensorOps<long int>::_prod(THLongTensor*, THLongTensor*, int)':
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:201:37: error: too few arguments to function 'void THLongTensor_prod(THLongTensor*, THLongTensor*, int, int)'
     return THTensor_(prod)(r, t, dim);
                                     ^
In file included from /root/torch/install/include/TH/THStorage.h:4:0,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Storage.h:14,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:18,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/root/torch/install/include/TH/THTensor.h:8:39: note: declared here
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                                       ^
/root/torch/install/include/TH/THGeneral.h:108:37: note: in definition of macro 'TH_CONCAT_4_EXPAND'
 #define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
                                     ^
/root/torch/install/include/TH/THTensor.h:8:27: note: in expansion of macro 'TH_CONCAT_4'
 #define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
                           ^
/root/torch/install/include/TH/generic/THTensorMath.h:78:13: note: in expansion of macro 'THTensor_'
 TH_API void THTensor_(prod)(THTensor *r_, THTensor *t, int dimension, int keepdim);
             ^
In file included from thpp/detail/TensorGeneric.h:1:0,
                 from /root/torch/install/include/TH/THGenerateIntTypes.h:14,
                 from /root/torch/install/include/TH/THGenerateAllTypes.h:11,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/Tensor.h:28,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/Tensor.h:19,
                 from /tmp/fblualib-build.7Iqzhq/thpp/thpp/TensorSerialization.cpp:11:
/tmp/fblualib-build.7Iqzhq/thpp/thpp/../thpp/detail/TensorGeneric.h:201:37: error: return-statement with a value, in function returning 'void' [-fpermissive]
     return THTensor_(prod)(r, t, dim);
                                     ^
make[2]: *** [CMakeFiles/thpp.dir/TensorSerialization.cpp.o] Error 1
make[1]: *** [CMakeFiles/thpp.dir/all] Error 2
make: *** [all] Error 2
The command '/bin/sh -c ./install_all.sh' returned a non-zero code: 2
```





参考：https://github.com/michael-chiang-mc5/slabStreetview-repo/blob/fb0082460702a104031a12af487ebfa6835abeb9/textDetector/work3.txt



a) 修改/opt/github/thpp/thpp/detail/TensorGeneric.h

```c
  static void _max(THTensor* values, THLongTensor* indices,
                   THTensor* t, int dim) {
    return THTensor_(max)(values, indices, t, dim);
  }
  static void _min(THTensor* values, THLongTensor* indices,
                   THTensor* t, int dim) {
    return THTensor_(min)(values, indices, t, dim);
  }
  static void _sum(THTensor* r, THTensor* t, int dim) {
    return THTensor_(sum)(r, t, dim);
  }
  static void _prod(THTensor* r, THTensor* t, int dim) {
    return THTensor_(prod)(r, t, dim);
  }

```

改为

```c
  static void _max(THTensor* values, THLongTensor* indices,
                   THTensor* t, int dim) {
    return THTensor_(max)(values, indices, t, dim,1);
  }
  static void _min(THTensor* values, THLongTensor* indices,
                   THTensor* t, int dim) {
    return THTensor_(min)(values, indices, t, dim,1);
  }
  static void _sum(THTensor* r, THTensor* t, int dim) {
    return THTensor_(sum)(r, t, dim,1);
  }
  static void _prod(THTensor* r, THTensor* t, int dim) {
    return THTensor_(prod)(r, t, dim,1);
  }

```



b) 注释install_all.sh中的git clone,增加mv

```shell
if [ $current -eq 1 ]; then
    git clone --depth 1 https://github.com/facebook/folly
    git clone --depth 1 https://github.com/facebook/fbthrift
    #git clone https://github.com/facebook/thpp
    git clone https://github.com/facebook/fblualib
    git clone https://github.com/facebook/wangle
else
    git clone -b v0.35.0  --depth 1 https://github.com/facebook/folly
    git clone -b v0.24.0  --depth 1 https://github.com/facebook/fbthrift
    #git clone -b v1.0 https://github.com/facebook/thpp
    git clone -b v1.0 https://github.com/facebook/fblualib
fi
mv /root/thpp $dir/
```



c) Dockerfile中增加 `ADD /opt/github/thpp /root/thpp`

```
# Install fblualib and its dependencies :
ADD install_all.sh /root/install_all.sh
ADD thpp_build.sh /root/thpp_build.sh
ADD /opt/github/thpp /root/thpp
```



4:



报错如下：

```shell
Status: Downloaded newer image for kaixhin/cuda-torch:8.0
 ---> 5607e0f9231e
Step 2/13 : ADD install_all.sh /root/install_all.sh
 ---> 6e93f465af1a
Step 3/13 : ADD thpp_build.sh /root/thpp_build.sh
 ---> e2ae42030aab
Step 4/13 : ADD /opt/github/thpp /root/thpp
ADD failed: stat /var/lib/docker/tmp/docker-builder229086589/opt/github/thpp: no such file or directory

```



原因 ADD只能加入当前目录的内容；把/opt/github/thpp 复制到/opt/github/crnn目录下





5：容器内th demo.lua启动demo报错

```shell
root@73fa1decef52:~/crnn/src# th demo.lua
/root/torch/install/bin/luajit: /root/torch/install/share/lua/5.1/trepl/init.lua:389: /root/torch/install/share/lua/5.1/trepl/init.lua:389: /root/torch/install/share/lua/5.1/cudnn/ffi.lua:1603: 'libcudnn (R5) not found in library path.
Please install CuDNN from https://developer.nvidia.com/cuDNN
Then make sure files named as libcudnn.so.5 or libcudnn.5.dylib are placed in
your library load path (for example /usr/local/lib , or manually add a path to LD_LIBRARY_PATH)

Alternatively, set the path to libcudnn.so.5 or libcudnn.5.dylib
to the environment variable CUDNN_PATH and rerun torch.
For example: export CUDNN_PATH = "/usr/local/cuda/lib64/libcudnn.so.5"

stack traceback:
	[C]: in function 'error'
	/root/torch/install/share/lua/5.1/trepl/init.lua:389: in function 'require'
	demo.lua:4: in main chunk
	[C]: in function 'dofile'
	/root/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:150: in main chunk
	[C]: at 0x00406670

```



解决方法：安装cudnn



a) 

```shell
docker cp cudnn-8.0-linux-x64-v5.1.tgz 73fa1decef52:/root/
```





```shell

mkdir /usr/local/cudnn
tar -xvf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local/cudnn
cd /usr/local/cudnn/cuda/include
cp *.h /usr/local/cuda/include/
cd /usr/local/cudnn/cuda/lib64
cp libcudnn* /usr/local/cuda/lib64/
chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn

export CUDNN_PATH=/usr/local/cuda/lib64/libcudnn.so.5

```



然后执行启动demo

```shell
cd /root/crnn/src/
th demo.lua
```



结果如下：

```shell
root@73fa1decef52:~/crnn/src# th demo.lua
Found Environment variable CUDNN_PATH = /usr/local/cuda/lib64/libcudnn.so.5	
Loading model...	
Model loaded from /root/crnn/model/crnn_demo/crnn_demo_model.t7	
Recognized text: available (raw: a-----v--a-i-l-a-bb-l-e---)	

```

