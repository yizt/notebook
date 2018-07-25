[TOC]



# python环境配置



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





### xml读写

