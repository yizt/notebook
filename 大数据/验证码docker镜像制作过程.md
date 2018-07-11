[TOC]



yum -y install docker-io



docker pull centos



```
docker run -t -i centos /bin/bash
```

运行

docker run -it --name verifycode centos /bin/bash



docker run -it --name verifycode --network host centos:v1 /bin/bash



删除container

​	docker rm 0a07b3bcf249

删除image

docker rmi centos:v1





制作image

docker commit -a 'yizt' -m 'test' fb09f8e9973c centos:v1



docker commit -a 'yizt' -m 'conda install' 645e3a3831e4 centos:conda



docker commit -a 'yizt' -m 'caffe install' 645e3a3831e4 centos:caffe



docker commit -a 'yizt' -m 'init' 645e3a3831e4 centos:init



docker commit -a 'yizt' -m 'init' cee4118557c8 centos:12306



进入容器

docker attach 645e3a3831e4



导出

docker save -o soft/test.tar centos:v1



docker save -o docker.tar centos:12306

docker save centos:12306 | gzip > docker.tgz



gzip docker.tar

分割、合并

split -b 500m test.tar.gz test.tar.gz.

cat test.tar.gz.aa test.tar.gz.ab test.tar.gz.ac test.tar.gz.ad test.tar.gz.ae test.tar.gz.af test.tar.gz.ag > test.tar.gz





split -b 500m docker.tar.gz docker.tar.gz.

导入

docker load --input soft/test.tar



```shell
docker load < docker.tgz
```



复制文件

docker cp caffe verifycode:/opt/github/caffe



## 安装软件

yum install -y wget

yum install -y bzip2

yum install -y net-tools

yum -y install gcc automake autoconf libtool make

yum -y install gcc+ gcc-c++

wget --user=root --password=123456 ftp://192.168.1.165/ambari/soft/Anaconda3-5.1.0-Linux-x86_64.sh



conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes



### caffe虚拟环境

conda create -n caffe python=2.7

source activate caffe



conda install tornado

conda install configparser

conda install flask



### tensorflow虚拟环境

conda remove -n tf --all

conda create -n tf python=3.6

source activate tf

conda install tensorflow=1.4.1

conda install tornado

conda install configparser

conda install flask

conda install pillow





基础环境

conda install jupyter

jupyter-notebook password 

设置为123456

jupyter notebook --no-browser --port=9000 --ip=192.168.1.219 --allow-root



### 复制代码和模型

docker cp /opt/code verifycode:/opt/code

docker cp /opt/models verifycode:/opt/models







caffe启动

python /opt/code/verify_code/caffe_head_predict_rest.py





tf启动

python /opt/code/verify_code/verify_code_tf_api/tf_predict_rest.py









## 问题记录

echo 'export LANG=en_US.UTF-8' >> ~/.bashrc