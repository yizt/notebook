[TOC]





## 安装配置

### 安装

```shell
yum -y install docker-io
```



**docker-io与docker-ce区别**

​         docker-io 是以前早期的版本，版本号是 1.\*，最新版是 1.13，而 docker-ce 是新的版本，分为社区版 docker-ce 和企业版 docker-ee，版本号是 17.\*



### 国内镜像配置

​        修改/etc/sysconfig/docker文件，增加docker加速地址--registry-mirror=[http://aad0405c.m.daocloud.io](http://aad0405c.m.daocloud.io/);修改OPTIONS属性值

```shell
OPTIONS='--selinux-enabled=false --log-driver=journald --signature-verification=false --registry-mirror=http://aad0405c.m.daocloud.io'
```



### 启动

```
service docker start
chkconfig docker on
```



## 镜像操作

a) 搜索

```shell
docker search centos
```

结果

```
[root@chinese ~]# docker search centos
INDEX       NAME                                         DESCRIPTION                                     STARS     OFFICIAL   AUTOMATED
docker.io   docker.io/centos                             The official build of CentOS.                   4419      [OK]       
docker.io   docker.io/ansible/centos7-ansible            Ansible on Centos7                              114                  [OK]
docker.io   docker.io/jdeathe/centos-ssh                 CentOS-6 6.9 x86_64 / CentOS-7 7.4.1708 x8...   97                   [OK]
docker.io   docker.io/consol/centos-xfce-vnc             Centos container with "headless" VNC sessi...   56                   [OK]
docker.io   docker.io/tutum/centos                       Simple CentOS docker image with SSH access      43                   
docker.io   docker.io/imagine10255/centos6-lnmp-php56    centos6-lnmp-php56                              42                   [OK]
docker.io   docker.io/centos/mysql-57-centos7            MySQL 5.7 SQL database server                   31                   

```





b) 拉取

```shell
docker pull centos
```



c) 列出本地镜像

```shell
docker images
```

或

```shell
docker image ls
```



d) 删除本地镜像

```shell
docker rmi centos:latest
```



## 容器操作

a) 启动容器

```shell
docker run -it --name v1 --network host centos:v1 /bin/bash
```

​        `-t` 选项让Docker分配一个伪终端（pseudo-tty）并绑定到容器的标准输入上， `-i` 则让容器的标准输入保持打开。



b) 查看容器

```shell
docker ps -l
```

或

```shell
docker container ls -a
```



c) 启动终止的容器

```shell
docker container start 15021ef90316
```



d) 









## 参考资料

[Docker-从入门到实战](https://yeasy.gitbooks.io/docker_practice/content/basic_concept/image.html)









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