[TOC]

## 环境准备

### boost安装

a) 依赖安装

```sh
yum -y install gcc gcc-c++ python python-devel libicu libicu-devel zlib zlib-devel bzip2 bzip2-devel
yum -y install python36-devel.x86_64
```



b) 下载解压

```shell
wget https://jaist.dl.sourceforge.net/project/boost/boost/1.51.0/boost_1_51_0.7z

7za x boost_1_51_0.7z
```



c) 编译安装

```shell
cd boost_1_51_0

export CPLUS_INCLUDE_PATH=/root/anaconda3/include/python3.6m

./bootstrap.sh --prefix=/usr/local/boost --with-python=/root/anaconda3/bin/python --with-python-root=/root/anaconda3/lib/python3.6

./b2 

./b2 install

```

```
./b2 –-with-python include=”你pyconfig.h的路径”←可用locate去寻找pyconfig.h的路径
```

### 依赖包安装

```shell
yum install protobuf-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel -y
yum install gflags-devel glog-devel lmdb-devel -y
yum install atlas-devel -y
```



### python环境创建

```shell
conda create -n caffe python=3.6
source activate caffe
```





## caffe 安装

a)下载

```shell
git clone https://github.com/BVLC/caffe.git
cd caffe
```



b)修改Makefile.config 如下属性

```makefile
USE_CUDNN := 1
CUDA_ARCH := #-gencode arch=compute_20,code=sm_20 \
		#-gencode arch=compute_20,code=sm_21 \
		-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

BLAS_LIB := /usr/lib64/atlas

# Uncomment to use Python 3 (default is Python 2)
PYTHON_LIBRARIES := boost_python3 python3.6m
PYTHON_INCLUDE := /root/anaconda3/envs/caffe/include/python3.6m \
    /root/anaconda3/envs/caffe/lib/python3.6/site-packages/numpy/core/include
PYTHON_LIB := /root/anaconda3/envs/caffe/lib

# Uncomment to support layers written in Python (will link against Python libs)
WITH_PYTHON_LAYER := 1

# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/local/boost/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/local/boost/lib
```



set(python_version "3" CACHE STRING "Specify which Python version to use")



c) 确定atlas软链接

​    确认atlas路径下是否包含libcblas.so和libatlas.so如果没有是因为 ATLAS现在的名称变了，要新建一下软连

```shell
cd /usr/lib64/atlas
ln -sv libsatlas.so.3.10 libcblas.so
ln -sv libsatlas.so.3.10 libatlas.so
```



d) python依赖安装

​    i) 首先修改python/requirements.txt文件中的protobuf版本为2.6.1，其它不变

```shell
protobuf==2.6.1
```

​    ii) 安装依赖

```shell
for req in $(cat python/requirements.txt); do pip install $req; done
```

​    

e) 编译安装

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/boost/lib
make -j8
make test
make runtest
make pycaffe
```



## 案例测试

1：mnist训练测试

export CPLUS_INCLUDE_PATH=/usr/local/protobuf2.6.1/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/boost/lib

```
sh data/mnist/get_mnist.sh
sh examples/mnist/create_mnist.sh
time sh examples/mnist/train_lenet.sh
```



## 错误记录

1：make -j8 报错如下

```
collect2: 错误：ld 返回 1

make: *** [.build_release/tools/convert_imageset.bin] 错误 1

/usr/bin/ld: warning: libboost_chrono.so.1.51.0, needed by /usr/local/boost/lib/libboost_thread.so, not found (try using -rpath or -rpath-link)

.build_release/lib/libcaffe.so：对‘boost::thread::start_thread_noexcept()’未定义的引用

.build_release/lib/libcaffe.so：对‘boost::thread::join_noexcept()’未定义的引用

collect2: 错误：ld 返回 1

make: *** [.build_release/tools/upgrade_net_proto_binary.bin] 错误 1

/usr/bin/ld: warning: libboost_chrono.so.1.51.0, needed by /usr/local/boost/lib/libboost_thread.so, not found (try using -rpath or -rpath-link)

.build_release/lib/libcaffe.so：对‘boost::thread::start_thread_noexcept()’未定义的引用

.build_release/lib/libcaffe.so：对‘boost::thread::join_noexcept()’未定义的引用

collect2: 错误：ld 返回 1

make: *** [.build_release/tools/caffe.bin] 错误 1

/usr/bin/ld: warning: libboost_chrono.so.1.51.0, needed by /usr/local/boost/lib/libboost_thread.so, not found (try using -rpath or -rpath-link)

.build_release/lib/libcaffe.so：对‘boost::thread::start_thread_noexcept()’未定义的引用

.build_release/lib/libcaffe.so：对‘boost::thread::join_noexcept()’未定义的引用

collect2: 错误：ld 返回 1

make: *** [.build_release/tools/upgrade_net_proto_text.bin] 错误 1

```



解决方法：在Makefile.config文件中修改如下属性

```
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/local/boost/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/local/boost/lib
```







2: make runtest -j16 报错

```
(caffe) [root@localhost caffe]# sh examples/mnist/create_mnist.sh

Creating lmdb...

build/examples/mnist/convert_mnist_data.bin: error while loading shared libraries: libboost_system.so.1.51.0: cannot open shared object file: No such file or directory

(caffe) [root@localhost caffe]# make runtest -j16

.build_release/tools/caffe

.build_release/tools/caffe: error while loading shared libraries: libboost_system.so.1.51.0: cannot open shared object file: No such file or directory

make: *** [runtest] 错误 127

```



解决方法：

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/boost/lib
make runtest -j16
```



3：import caffe报错

```python
import sys
caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe
```



```shell
b/python3.6', '/root/anaconda3/envs/caffe/lib/python3.6/lib-dynload', '/root/anaconda3/envs/caffe/lib/python3.6/site-packages', '/root/anaconda3/envs/caffe/lib/python3.6/site-packages/IPython/extensions', '/root/.ipython']
Failed to include caffe_pb2, things might go wrong!
Traceback (most recent call last):

  File "/root/anaconda3/envs/caffe/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 2963, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)

  File "<ipython-input-4-1b20e07ad883>", line 8, in <module>
    import caffe

  File "/opt/github/caffe/python/caffe/__init__.py", line 4, in <module>
    from .proto.caffe_pb2 import TRAIN, TEST

  File "/opt/github/caffe/python/caffe/proto/caffe_pb2.py", line 7, in <module>
    from google.protobuf import reflection as _reflection

  File "/root/anaconda3/envs/caffe/lib/python3.6/site-packages/google/protobuf/reflection.py", line 68, in <module>
    from google.protobuf.internal import python_message

  File "/root/anaconda3/envs/caffe/lib/python3.6/site-packages/google/protobuf/internal/python_message.py", line 848
    except struct.error, e:
                       ^
SyntaxError: invalid syntax
```



解决方法：

```shell
pip install protobuf-py3
```

   这是因为 protobuf 不支持 python3，解决方案是安装 pip install protobuf-py3, 一个 python3.x 版本的替代包。





4: import caffe 报错

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/boost/lib
ipython
```



```python
import sys
sys.path.insert(0,'/opt/github/caffe/python')
import caffe

```





```python
In [9]: import caffe
Failed to include caffe_pb2, things might go wrong!
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-9-6e7bb19bc708> in <module>()
----> 1 import caffe

/opt/github/caffe/python/caffe/__init__.py in <module>()
      2 from ._caffe import init_log, log, set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver, layer_type_list, set_random_seed, solver_count, set_solver_count, solver_rank, set_solver_rank, set_multiprocess, has_nccl
      3 from ._caffe import __version__
----> 4 from .proto.caffe_pb2 import TRAIN, TEST
      5 from .classifier import Classifier
      6 from .detector import Detector

/opt/github/caffe/python/caffe/proto/caffe_pb2.py in <module>()
   1004       name='type', full_name='caffe.FillerParameter.type', index=0,
   1005       number=1, type=9, cpp_type=9, label=1,
-> 1006       has_default_value=True, default_value=unicode("constant", "utf-8"),
   1007       message_type=None, enum_type=None, containing_type=None,
   1008       is_extension=False, extension_scope=None,

NameError: name 'unicode' is not defined

```



参考：https://stackoverflow.com/questions/33423207/nameerror-name-unicode-is-not-defined-when-compile-with-python3



参考：https://blog.csdn.net/aBlueMouse/article/details/77744023





5: 重新编译

```shell
export PATH=/usr/local/protobuf/bin:$PATH
export CPLUS_INCLUDE_PATH=/usr/local/protobuf/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/boost/lib

export LD_LIBRARY_PATH=/usr/local/protobuf/lib:$LD_LIBRARY_PATH

make clean
make -j16
```





```shell
NVCC src/caffe/layers/cudnn_pooling_layer.cu
In file included from /usr/include/c++/4.8.2/mutex:35:0,
                 from /usr/local/protobuf/include/google/protobuf/stubs/mutex.h:33,
                 from /usr/local/protobuf/include/google/protobuf/stubs/common.h:52,
                 from .build_release/src/caffe/proto/caffe.pb.h:9,
                 from ./include/caffe/util/cudnn.hpp:8,
                 from ./include/caffe/util/device_alternate.hpp:40,
                 from ./include/caffe/common.hpp:19,
                 from ./include/caffe/blob.hpp:8,
                 from ./include/caffe/layers/batch_reindex_layer.hpp:7,
                 from src/caffe/layers/batch_reindex_layer.cu:5:
/usr/include/c++/4.8.2/bits/c++0x_warning.h:32:2: 错误：#error This file requires compiler and library support for the ISO C++ 2011 standard. This support is currently experimental, and must be enabled with the -std=c++11 or -std=gnu++11 compiler options.
 #error This file requires compiler and library support for the \
  ^
In file included from /usr/include/c++/4.8.2/mutex:35:0,
                 from /usr/local/protobuf/include/google/protobuf/stubs/mutex.h:33,
                 from /usr/local/protobuf/include/google/protobuf/stubs/common.h:52,
                 from .build_release/src/caffe/proto/caffe.pb.h:9,
                 from ./include/caffe/util/cudnn.hpp:8,
                 from ./include/caffe/util/device_alternate.hpp:40,
                 from ./include/caffe/common.hpp:19,
                 from ./include/caffe/blob.hpp:8,
                 from ./include/caffe/layers/batch_norm_layer.hpp:6,
                 from src/caffe/layers/batch_norm_layer.cu:4:
/usr/include/c++/4.8.2/bits/c++0x_warning.h:32:2: 错误：#error This file requires compiler and library support for the ISO C++ 2011 standard. This support is currently experimental, and must be enabled with the -std=c++11 or -std=gnu++11 compiler options.
 #error This file requires compiler and library support for the \

```



解决方法：尚未解决

参考：https://github.com/Tencent/ncnn/issues/336

https://stackoverflow.com/questions/16886591/how-do-i-enable-c11-in-gcc

https://blog.csdn.net/jay463261929/article/details/59591104



6: protobuf  (3.3,3.5,3.0-beta)  make check 报错

```
./autogen.sh
./configure --prefix=/usr/local/protobuf3.3.0
make
make check
```





```
depbase=`echo src/gmock-all.lo | sed 's|[^/]*$|.deps/&|;s|\.lo$||'`;\
/bin/sh ./libtool  --tag=CXX   --mode=compile g++ -DHAVE_CONFIG_H -I. -I./build-aux  -I./../googletest/include -I./include  -pthread -DGTEST_HAS_PTHREAD=1 -g -DNDEBUG -MT src/gmock-all.lo -MD -MP -MF $depbase.Tpo -c -o src/gmock-all.lo src/gmock-all.cc &&\
mv -f $depbase.Tpo $depbase.Plo
libtool: Version mismatch error.  This is libtool 2.4.6, but the
libtool: definition of this LT_INIT comes from libtool 2.4.2.
libtool: You should recreate aclocal.m4 with macros from libtool 2.4.6
libtool: and run autoconf again.
make[3]: *** [src/gmock-all.lo] 错误 63
make[3]: 离开目录“/opt/github/protobuf/third_party/googletest/googlemock”
make[2]: *** [check-local] 错误 2
make[2]: 离开目录“/opt/github/protobuf”
make[1]: *** [check-am] 错误 2
make[1]: 离开目录“/opt/github/protobuf”
make: *** [check-recursive] 错误 1

```



尚未解决：

参考：https://stackoverflow.com/questions/3096989/libtool-version-mismatch-error

http://www.howtodoityourself.org/how-to-fix-libtool-version-mismatch-error.html





7: protobuf 2.6.1   python setup.py test 报错

```shell
source activate caffe
export PATH=/usr/local/protobuf2.6.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/protobuf2.6.1/lib:$LD_LIBRARY_PATH
protoc --version
cd python/
python setup.py build
python setup.py test

```



```shell
running build_ext
Traceback (most recent call last):
  File "setup.py", line 200, in <module>
    "Protocol Buffers are Google's data interchange format.",
  File "/root/anaconda3/envs/caffe/lib/python3.6/site-packages/setuptools/__init__.py", line 129, in setup
    return distutils.core.setup(**attrs)
  File "/root/anaconda3/envs/caffe/lib/python3.6/distutils/core.py", line 148, in setup
    dist.run_commands()
  File "/root/anaconda3/envs/caffe/lib/python3.6/distutils/dist.py", line 955, in run_commands
    self.run_command(cmd)
  File "/root/anaconda3/envs/caffe/lib/python3.6/distutils/dist.py", line 974, in run_command
    cmd_obj.run()
  File "/root/anaconda3/envs/caffe/lib/python3.6/site-packages/setuptools/command/test.py", line 226, in run
    self.run_tests()
  File "/root/anaconda3/envs/caffe/lib/python3.6/site-packages/setuptools/command/test.py", line 248, in run_tests
    exit=False,
  File "/root/anaconda3/envs/caffe/lib/python3.6/unittest/main.py", line 94, in __init__
    self.parseArgs(argv)
  File "/root/anaconda3/envs/caffe/lib/python3.6/unittest/main.py", line 141, in parseArgs
    self.createTests()
  File "/root/anaconda3/envs/caffe/lib/python3.6/unittest/main.py", line 148, in createTests
    self.module)
  File "/root/anaconda3/envs/caffe/lib/python3.6/unittest/loader.py", line 219, in loadTestsFromNames
    suites = [self.loadTestsFromName(name, module) for name in names]
  File "/root/anaconda3/envs/caffe/lib/python3.6/unittest/loader.py", line 219, in <listcomp>
    suites = [self.loadTestsFromName(name, module) for name in names]
  File "/root/anaconda3/envs/caffe/lib/python3.6/unittest/loader.py", line 204, in loadTestsFromName
    test = obj()
  File "/home/soft/protobuf-2.6.1/python/setup.py", line 89, in MakeTestSuite
    import google.protobuf.pyext.descriptor_cpp2_test as descriptor_cpp2_test
  File "/home/soft/protobuf-2.6.1/python/google/protobuf/pyext/descriptor_cpp2_test.py", line 43, in <module>
    from google.apputils import basetest
  File "/home/soft/protobuf-2.6.1/python/.eggs/google_apputils-0.4.2-py3.6.egg/google/apputils/basetest.py", line 1063
    0600)
       ^
SyntaxError: invalid token

```



protobuf2.6.1不支持python3

参考：https://stackoverflow.com/questions/33715923/failed-to-install-protobuf-in-python3





8: python2.7环境   

```shell
source activate faster-rcnn
export PATH=/usr/local/protobuf2.6.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/protobuf2.6.1/lib:$LD_LIBRARY_PATH
protoc --version
cd python/
python setup.py build
python setup.py test
```



```shell
running build_ext
Traceback (most recent call last):
  File "setup.py", line 200, in <module>
    "Protocol Buffers are Google's data interchange format.",
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/distutils/core.py", line 151, in setup
    dist.run_commands()
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/distutils/dist.py", line 953, in run_commands
    self.run_command(cmd)
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/distutils/dist.py", line 972, in run_command
    cmd_obj.run()
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/site-packages/setuptools-20.7.0-py2.7.egg/setuptools/command/test.py", line 159, in run
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/site-packages/setuptools-20.7.0-py2.7.egg/setuptools/command/test.py", line 140, in with_project_on_sys_path
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/site-packages/setuptools-20.7.0-py2.7.egg/setuptools/command/test.py", line 180, in run_tests
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/unittest/main.py", line 94, in __init__
    self.parseArgs(argv)
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/unittest/main.py", line 149, in parseArgs
    self.createTests()
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/unittest/main.py", line 158, in createTests
    self.module)
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/unittest/loader.py", line 130, in loadTestsFromNames
    suites = [self.loadTestsFromName(name, module) for name in names]
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/unittest/loader.py", line 115, in loadTestsFromName
    test = obj()
  File "/home/soft/protobuf-2.6.1/python/setup.py", line 89, in MakeTestSuite
    import google.protobuf.pyext.descriptor_cpp2_test as descriptor_cpp2_test
  File "/home/soft/protobuf-2.6.1/python/google/protobuf/pyext/descriptor_cpp2_test.py", line 47, in <module>
    from google.protobuf.internal.descriptor_test import *
  File "/home/soft/protobuf-2.6.1/python/google/protobuf/internal/descriptor_test.py", line 38, in <module>
    from google.protobuf import unittest_custom_options_pb2
  File "/home/soft/protobuf-2.6.1/python/google/protobuf/unittest_custom_options_pb2.py", line 7, in <module>
    from google.protobuf import descriptor as _descriptor
  File "/home/soft/protobuf-2.6.1/python/google/protobuf/descriptor.py", line 50, in <module>
    from google.protobuf.pyext import _message
ImportError: cannot import name _message
```



解决方法

```
python setup.py test --cpp_implementation
```



新的错误

```
copying build/lib.linux-x86_64-2.7/google/protobuf/pyext/_message.so -> google/protobuf/pyext
Traceback (most recent call last):
  File "setup.py", line 200, in <module>
    "Protocol Buffers are Google's data interchange format.",
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/distutils/core.py", line 151, in setup
    dist.run_commands()
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/distutils/dist.py", line 953, in run_commands
    self.run_command(cmd)
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/distutils/dist.py", line 972, in run_command
    cmd_obj.run()
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/site-packages/setuptools-20.7.0-py2.7.egg/setuptools/command/test.py", line 159, in run
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/site-packages/setuptools-20.7.0-py2.7.egg/setuptools/command/test.py", line 140, in with_project_on_sys_path
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/site-packages/setuptools-20.7.0-py2.7.egg/setuptools/command/test.py", line 180, in run_tests
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/unittest/main.py", line 94, in __init__
    self.parseArgs(argv)
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/unittest/main.py", line 149, in parseArgs
    self.createTests()
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/unittest/main.py", line 158, in createTests
    self.module)
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/unittest/loader.py", line 130, in loadTestsFromNames
    suites = [self.loadTestsFromName(name, module) for name in names]
  File "/root/anaconda3/envs/faster-rcnn/lib/python2.7/unittest/loader.py", line 115, in loadTestsFromName
    test = obj()
  File "/home/soft/protobuf-2.6.1/python/setup.py", line 89, in MakeTestSuite
    import google.protobuf.pyext.descriptor_cpp2_test as descriptor_cpp2_test
  File "/home/soft/protobuf-2.6.1/python/google/protobuf/pyext/descriptor_cpp2_test.py", line 47, in <module>
    from google.protobuf.internal.descriptor_test import *
  File "/home/soft/protobuf-2.6.1/python/google/protobuf/internal/descriptor_test.py", line 38, in <module>
    from google.protobuf import unittest_custom_options_pb2
  File "/home/soft/protobuf-2.6.1/python/google/protobuf/unittest_custom_options_pb2.py", line 7, in <module>
    from google.protobuf import descriptor as _descriptor
  File "/home/soft/protobuf-2.6.1/python/google/protobuf/descriptor.py", line 50, in <module>
    from google.protobuf.pyext import _message
ImportError: /home/soft/protobuf-2.6.1/python/google/protobuf/pyext/_message.so: undefined symbol: _ZNK6google8protobuf10TextFormat17FieldValuePrinter9PrintBoolEb

```

无法解决：



解决方法：修改Makefile.config



```makefile
INCLUDE_DIRS := /usr/local/protobuf2.6.1/include $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := /usr/local/protobuf2.6.1/lib $(PYTHON_LIB) /usr/local/lib /usr/lib
```



然后



10:   py-faster-rcnn caffe安装

```shell
make -j8 runtest
```



```shell
CXX src/caffe/test/test_reshape_layer.cpp
CXX src/caffe/test/test_roi_pooling_layer.cpp
CXX src/caffe/test/test_scale_layer.cpp
CXX src/caffe/test/test_sigmoid_cross_entropy_loss_layer.cpp
CXX src/caffe/test/test_slice_layer.cpp
CXX src/caffe/test/test_smooth_L1_loss_layer.cpp
src/caffe/test/test_smooth_L1_loss_layer.cpp:11:35: 致命错误：caffe/vision_layers.hpp：没有那个文件或目录
 #include "caffe/vision_layers.hpp"
                                   ^
编译中断。
make: *** [.build_release/src/caffe/test/test_smooth_L1_loss_layer.o] 错误 1
make: *** 正在等待未完成的任务....

```

解决方法：直接删除#include "caffe/vision_layers.hpp"这行





11 py-faster-rcnn caffe安装



```shell
make -j16 runtest
```



```shell
.build_release/tools/caffe
.build_release/tools/caffe: error while loading shared libraries: libhdf5_hl.so.10: cannot open shared object file: No such file or directory
make: *** [runtest] 错误 127

```



增加后

```
export LD_LIBRARY_PATH=/root/anaconda3/envs/faster-rcnn/lib:$LD_LIBRARY_PATH
```

报错如下：

```
.build_release/tools/caffe
.build_release/tools/caffe: /root/anaconda3/envs/faster-rcnn/lib/libtiff.so.5: no version information available (required by /lib64/libopencv_highgui.so.2.4)
[libprotobuf FATAL google/protobuf/stubs/common.cc:61] This program requires version 3.5.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1.  Please update your library.  If you compiled the program yourself, make sure that your headers are from the same version of Protocol Buffers as your link-time library.  (Version verification failed in "google/protobuf/any.pb.cc".)
terminate called after throwing an instance of 'google::protobuf::FatalException'
  what():  This program requires version 3.5.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1.  Please update your library.  If you compiled the program yourself, make sure that your headers are from the same version of Protocol Buffers as your link-time library.  (Version verification failed in "google/protobuf/any.pb.cc".)
make: *** [runtest] 已放弃

```

