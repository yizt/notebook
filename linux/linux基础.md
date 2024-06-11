a)操作系统版本查看

```shell
lsb_release -a 
cat /etc/issue
```



b) 

```shell

===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-11.7/

Please make sure that
 -   PATH includes /usr/local/cuda-11.7/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-11.7/lib64, or, add /usr/local/cuda-11.7/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.7/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 515.00 is required for CUDA 11.7 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log

```



参考：https://towardsdatascience.com/installing-multiple-cuda-cudnn-versions-in-ubuntu-fcb6aa5194e2



c) cudnn安装

```shell
tar -xzvf cudnn-x.x-linux-x64-v8.x.x.x.tgz
tar -xzvf cudnn-x.x-linux-aarch64sbsa-v8.x.x.x.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda-M.m/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-M.m/lib64

sudo chmod a+r /usr/local/cuda-M.m/include/cudnn*.h /usr/local/cuda-M.m/lib64/libcudnn*

# 实际执行
sudo cp include/cudnn*.h /usr/local/cuda-11.7/include/
sudo cp lib/libcudnn* /usr/local/cuda-11.7/lib64/

sudo chmod a+r /usr/local/cuda-11.7/include/cudnn*.h /usr/local/cuda-11.7/lib64/libcudnn*

PATH includes /usr/local/cuda-11.7/bin
LD_LIBRARY_PATH includes /usr/local/cuda-11.7/lib64
```

