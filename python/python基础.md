[TOC]



## python环境配置



### pip

镜像配置, vi ~/.pip/pip.conf,不存在，就创建此文件，内容如下

```wiki
[global]
timeout = 6000
index-url = http://pypi.douban.com/simple/
[install]
trusted-host = pypi.douban.com
```



### conda

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```





## 基础编程



### 压缩文件读写

### xml读写



### 图像文件IO

```
skimage.io
plt
opencv
pillow
keras.preprocessing.image
```





## 综合

### 参数传递

形参*args, **kwargs

### 导入

相对路径导入

绝对路径导入



### py2to3

30/4