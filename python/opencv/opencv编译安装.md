



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


make -j5
sudo make install
```





```shell
mkdir /Users/yizuotian/OpenCV3.3.1
git checkout 3.3.1

mkdir build-3.3.1
cd build-3.3.1

cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
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

      ```
grep -R 'include <malloc.h>' *
      ```

 

​     