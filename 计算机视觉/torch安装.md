



## centos7 下安装torch



docker run -it --name torch7 centos /bin/bash



```shell
wget -O /etc/yum.repos.d/epel-7.repo http://mirrors.aliyun.com/repo/epel-7.repo
```



### gpu环境



```shell
./NVIDIA-Linux-x86_64-390.48.run --kernel-source-path=/usr/src/kernels/3.10.0-862.9.1.el7.x86_64/ -k 3.10.0-862.9.1.el7.x86_64 --dkms
```





cudn

```
./cuda_8.0.61_375.26_linux.run --kernel-source-path='/usr/src/kernels/3.10.0-862.9.1.el7.x86_64'
```



cudnn

```
mkdir /usr/local/cudnn
tar -xvf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local/cudnn
```





```shell
yum install -y git cmake
yum -y install make gcc+ gcc-c++
yum install -y which wget
yum install pciutils lsof -y
yum install -y epel-release
```



### torch安装

```shell
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | bash
```



yum -y install gcc automake autoconf libtool make

yum -y install make gcc+ gcc-c++





## 错误记录

1: 

```shell
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | bash
```

报错如下：

```shell
CMake Error: CMake was unable to find a build program corresponding to "Unix Makefiles".  CMAKE_MAKE_PROGRAM is not set.  You probably need to select a different build tool.
CMake Error: Error required internal CMake variable not set, cmake may be not be built correctly.
Missing variable is:
CMAKE_C_COMPILER_ENV_VAR
CMake Error: Error required internal CMake variable not set, cmake may be not be built correctly.
Missing variable is:
CMAKE_C_COMPILER
CMake Error: Could not find cmake module file: /tmp/luajit-rocks/build/CMakeFiles/2.8.12.2/CMakeCCompiler.cmake
CMake Error: Error required internal CMake variable not set, cmake may be not be built correctly.
Missing variable is:
CMAKE_CXX_COMPILER_ENV_VAR
CMake Error: Error required internal CMake variable not set, cmake may be not be built correctly.
Missing variable is:
CMAKE_CXX_COMPILER
CMake Error: Could not find cmake module file: /tmp/luajit-rocks/build/CMakeFiles/2.8.12.2/CMakeCXXCompiler.cmake
CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
-- Configuring incomplete, errors occurred!
Error. Exiting.

```



解决方法：

```shell
yum -y install make gcc+ gcc-c++
```



2:

```shell
./NVIDIA-Linux-x86_64-390.48.run --kernel-source-path=/usr/src/kernels/3.10.0-862.9.1.el7.x86_64/ -k $(uname -r) --dkms
```



```shell
ERROR: An NVIDIA kernel module 'nvidia-uvm' appears to already be loaded in your kernel.  This may be because it is in use (for example, by  
         an X server, a CUDA program, or the NVIDIA Persistence Daemon), but this may also happen if your kernel was configured without        
         support for module unloading.  Please be sure to exit any programs that may be using the GPU(s) before attempting to upgrade your     
         driver.  If no GPU-based programs are running, you know that your kernel supports module unloading, and you still receive this        
         message, then an error may have occured that has corrupted an NVIDIA kernel module's usage count, for which the simplest remedy is to 
         reboot your computer.   
```



rmmod nvidia-uvm





3: yum -y install gcc kernel-devel "kernel-devel-uname-r == $(uname -r)" dkms

```
No package kernel-devel-uname-r == 3.10.0-693.21.1.el7.x86_64 available
```



```shell
yum -y install gcc kernel-devel "kernel-devel-uname-r == 3.10.0-862.9.1.el7.x86_64" dkms
```

