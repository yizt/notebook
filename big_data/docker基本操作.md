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

或

```shell
docker image rm centos:latest
```



e) commit创建镜像

```
docker commit -a 'yizt' -m 'conda install' 645e3a3831e4 centos:conda
```

​          ` -a` 是指定修改的作者，而 `-m` 则是记录本次修改的内容；提交后`docker image ls` 可以看到刚才提交的 centos:conda镜像



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



d) 终止运行的容器

```shell
docker container stop 15021ef90316
```



e) 进入容器

```shell
docker exec -it 15021ef90316 /bin/bash
```



f) 容器删除

```shell
docker rm 411be4f25cb8
```

或

```shell
docker container rm 411be4f25cb8
```

​         如果要删除一个运行中的容器，可以添加 `-f` 参数。Docker 会发送 `SIGKILL` 信号给容器。



## 导入导出

a) 镜像导出

```shell
docker save -o docker.tar centos:v1
```

导出加压缩

```shell
docker save centos:v1 | gzip > docker.tgz
```



b) 镜像导入

```shell
docker load -i docker.tar
```

或

```shell
docker load < docker.tgz
```



c) 容器导出

```shell
docker export 15021ef90316 > centos.tar
```



d) 容器导入

```
docker import centos.tar centos:v1
```



### docker save和docker export的区别

docker save和docker export的区别：

1. docker save保存的是镜像（image），docker export保存的是容器（container）；
2. docker load用来载入镜像包，docker import用来载入容器包，但两者都会恢复为镜像；
3. docker load不能对载入的镜像重命名，而docker import可以为镜像指定新名称。



## 容器与主机通信

a) 文件复制

​     i)从容器中复制文件或目录到执行命令所在机器的指定路径
docker cp [OPTIONS] CONTAINER:SRC_PATH DEST_PATH

​     ii)从执行命令所在的机器复制文件或目录到容器内部的指定路径
docker cp [OPTIONS] SRC_PATH CONTAINER:DEST_PATH ;

```shell
docker cp /www/runoob 96f7f14e99ab:/www/
```

​        将主机/www/runoob目录拷贝到容器96f7f14e99ab的/www目录下。



## Dockerfile定制镜像

在Dockerfile目录下执行docker build . 即可; -t指定镜像名称

```
docker build -t caffe2:v1 .
```





## Nvidia-docker安装

​             可以在docker容器中执行gpu；参照：<https://github.com/NVIDIA/nvidia-docker> 执行如下命令即可

```
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
docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```



## mac主机安装

```shell
wget https://download.docker.com/mac/stable/Docker.dmg
```



```shell
yizt-mac:~ admin$ more ~/.docker/daemon.json
{
  "debug" : true,
  "experimental" : false,
  "insecure-registries" : [

  ],
  "registry-mirrors" : [
    "http://aad0405c.m.daocloud.io"
  ]
}
http://aad0405c.m.daocloud.io
```





## 参考资料

[Docker-从入门到实战](https://yeasy.gitbooks.io/docker_practice/content/basic_concept/image.html)
