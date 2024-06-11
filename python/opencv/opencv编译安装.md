



```shell
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git


cd opencv
mkdir build


```





```shell
brew search gcc
brew install --build-from-source gcc@9
```



```shell
alias gcc='/usr/local/Cellar/gcc@8/8.4.0/bin/gcc-8'
alias cc='/usr/local/Cellar/gcc@8/8.4.0/bin/gcc-8'
alias g++='/usr/local/Cellar/gcc@8/8.4.0/bin/g++-8'
alias c++='/usr/local/Cellar/gcc@8/8.4.0/bin/c++-8'

alias gcc='/usr/local/Cellar/gcc/9.1.0/bin/gcc-9'
alias cc='/usr/local/Cellar/gcc/9.1.0/bin/gcc-9'
alias g++='/usr/local/Cellar/gcc/9.1.0/bin/g++-9'
alias c++='/usr/local/Cellar/gcc/9.1.0/bin/c++-9'


alias gcc='/usr/local/Cellar/gcc@9/9.3.0_1/bin/gcc-9'
alias cc='/usr/local/Cellar/gcc@9/9.3.0_1/bin/gcc-9'
alias g++='/usr/local/Cellar/gcc@9/9.3.0_1/bin/g++-9'
alias c++='/usr/local/Cellar/gcc@9/9.3.0_1/bin/c++-9'
```





```shell
## 安装
cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-DCMAKE_INSTALL_PREFIX=/usr/local \
-DCMAKE_CXX_FLAGS=-std=c++11 \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_python3=OFF \
-DINSTALL_C_EXAMPLES=OFF \
-DOPENCV_ENABLE_NONFREE=ON \
-DBUILD_EXAMPLES=OFF ..

git checkout 3.4
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-DCMAKE_INSTALL_PREFIX=/Users/yizuotian/OpenCV3.4 \
-DCMAKE_CXX_FLAGS=-std=c++11 \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_python3=OFF \
-DINSTALL_C_EXAMPLES=OFF \
-DOPENCV_ENABLE_NONFREE=ON \
-DBUILD_EXAMPLES=OFF ..

# 带python3
git checkout 3.4

cd build-3.4
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-DCMAKE_INSTALL_PREFIX=/Users/yizuotian/OpenCV3.4.x \
-DCMAKE_CXX_FLAGS=-std=c++11 \
-DCMAKE_CXX_COMPILER=/usr/local/Cellar/gcc@9/9.3.0_1/bin/g++-9 \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_python3=ON \
-D PYTHON3_EXECUTABLE=/Users/yizuotian/miniconda2/envs/weibo/bin/python3.6m \
-D PYTHON3_INCLUDE_DIR=/Users/yizuotian/miniconda2/envs/weibo/include/python3.6m \
-D PYTHON3_LIBRARY=/Users/yizuotian/miniconda2/envs/weibo/lib/libpython3.6m.dylib \
-D PYTHON3_NUMPY_INCLUDE_DIRS=/Users/yizuotian/miniconda2/envs/weibo/lib/python3.6/site-packages/numpy/core/include \
-D PYTHON3_PACKAGES_PATH=/Users/yizuotian/miniconda2/envs/weibo/lib/python3.6/site-packages \
-D PYTHON_DEFAULT_EXECUTABLE=/Users/yizuotian/miniconda2/envs/weibo/bin/python3.6m \
-DINSTALL_C_EXAMPLES=OFF \
-DOPENCV_ENABLE_NONFREE=ON \
-DBUILD_EXAMPLES=OFF ..


cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-DCMAKE_INSTALL_PREFIX=/Users/yizuotian/OpenCV3.4.x \
-DCMAKE_CXX_FLAGS=-std=c++11 \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_python3=OFF \
-DINSTALL_C_EXAMPLES=OFF \
-DOPENCV_ENABLE_NONFREE=ON \
-DBUILD_EXAMPLES=OFF ..



make -j5
sudo make install
```





```shell
mkdir /Users/yizuotian/OpenCV3.3.1
git checkout 3.3.1

mkdir build-3.3.1
cd build-3.3.1

cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=/Users/yizuotian/OpenCV3.3.1 \
-DCMAKE_CXX_FLAGS=-std=c++11 \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_python3=ON \
-D PYTHON3_EXECUTABLE=/Users/yizuotian/miniconda2/envs/weibo/bin/python3.6m \
-D PYTHON3_INCLUDE_DIR=/Users/yizuotian/miniconda2/envs/weibo/include/python3.6m \
-D PYTHON3_LIBRARY=/Users/yizuotian/miniconda2/envs/weibo/lib/libpython3.6m.dylib \
-D PYTHON3_NUMPY_INCLUDE_DIRS=/Users/yizuotian/miniconda2/envs/weibo/lib/python3.6/site-packages/numpy/core/include \
-D PYTHON3_PACKAGES_PATH=/Users/yizuotian/miniconda2/envs/weibo/lib/python3.6/site-packages \
-D PYTHON_DEFAULT_EXECUTABLE=/Users/yizuotian/miniconda2/envs/weibo/bin/python3.6m \
-DINSTALL_C_EXAMPLES=OFF \
-DOPENCV_ENABLE_NONFREE=ON \
-DWITH_V4AL=OFF \
-DWITH_V4L=OFF \
-DWITH_LIBV4L=OFF \
-DBUILD_EXAMPLES=OFF ..

cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=/Users/yizuotian/OpenCV3.3.1 \
-DCMAKE_CXX_FLAGS=-std=c++11 \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_python3=ON \
-D PYTHON3_EXECUTABLE=/Users/yizuotian/miniconda2/envs/weibo/bin/python3.6m \
-D PYTHON3_INCLUDE_DIR=/Users/yizuotian/miniconda2/envs/weibo/include/python3.6m \
-D PYTHON3_LIBRARY=/Users/yizuotian/miniconda2/envs/weibo/lib/libpython3.6m.dylib \
-D PYTHON3_NUMPY_INCLUDE_DIRS=/Users/yizuotian/miniconda2/envs/weibo/lib/python3.6/site-packages/numpy/core/include \
-D PYTHON3_PACKAGES_PATH=/Users/yizuotian/miniconda2/envs/weibo/lib/python3.6/site-packages \
-D PYTHON_DEFAULT_EXECUTABLE=/Users/yizuotian/miniconda2/envs/weibo/bin/python3.6m \
-DINSTALL_C_EXAMPLES=OFF \
-DOPENCV_ENABLE_NONFREE=ON \
-DBUILD_EXAMPLES=OFF ..
```





## 错误记录

1. error: "__POPCNT__ is not defined by compiler"

   ```shell
   brew install gcc@9.1
   ```

   

   

2. /Users/yizuotian/soft/opencv/cmake/checks/cxx11.cpp:4:2: error: "C++11 is not supported"

​    -DCMAKE_CXX_FLAGS=-std=c++11 \



3.

https://raw.githubusercontent.com/opencv/opencv_3rdparty/a56b6ac6f030c312b2dce17430eef13aed9af274/ippicv/ippicv_2020_mac_intel64_20191018_general.tgz



4. *** No rule to make target `/usr/local/lib/libpng.dylib', needed by `lib/libopencv_viz.3.4.13.dylib'.  Stop. 

   ```shell
   brew install libpng
   ```

   

5. fatal error: 'malloc.h' file not found
   #include <malloc.h>

      ```bash
grep -R 'include <malloc.h>' *

#include <malloc.h>
修改为
#include <sys/malloc.h>

sed -i "" "s/<malloc.h>/<sys\/malloc.h>/g" 3rdparty/ippicv/ippicv_mac/iw/src/iw_core.c
sed -i "" "s/<malloc.h>/<sys\/malloc.h>/g" 3rdparty/ippicv/ippicv_mac/iw/src/iw_own.c
sed -i "" "s/<malloc.h>/<sys\/malloc.h>/g" 3rdparty/tinydnn/tiny-dnn-1.0.0a3/tiny_dnn/util/aligned_allocator.h
sed -i "" "s/<malloc.h>/<sys\/malloc.h>/g" 3rdparty/tinydnn/tiny-dnn-1.0.0a3/third_party/gemmlowp/internal/allocator.h
      ```

 

6. fatal error: 'linux/videodev.h' file not found

   ```shell
   mkdir /usr/include/linux
   ln -s /usr/local/arm/3.4.1/arm-linux/include/linux/videodev.h  /usr/include/linux/videodev.h
   ln -s /usr/local/arm/3.4.1/arm-linux/include/linux/videodev2.h  /usr/include/linux/videodev2.h
   
   sudo ln -s /usr/local/arm/3.4.1/arm-linux/include/linux /usr/include/linux
   sudo ln -s /usr/local/arm/3.4.1/arm-linux/include/asm /usr/include/asm
   ```

   

7. 'linux/types.h' file not found

   ```shell
   sudo ln -s /usr/local/arm/3.4.1/arm-linux/include/linux/types.h /usr/include/linux/types.h
   ```

8. fatal error: 'sys/videoio.h' file not found

   ```
   -DWITH_V4AL=OFF and -DWITH_LIBV4L=ON
   ```

   

 