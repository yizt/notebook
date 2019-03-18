[TOC]



### 部署



a) 参考

https://hortonworks.com/tutorial/sandbox-deployment-and-install-guide/section/3/

b) 部署

```shell
sh docker-deploy-{HDPversion}.sh
```



c) 启停

```shell
docker start sandbox-hdp
docker start sandbox-proxy
```



```shell
docker stop sandbox-hdp
docker stop sandbox-proxy
```



d) 删除

```
docker rm sandbox-hdp
docker rm sandbox-proxy
```



e) 手动启动

```
hostname="sandbox-hdp.hortonworks.com"
docker run --privileged --name sandbox-hdp --network host -h $hostname -v /docker/share:/share -d hortonworks/sandbox-hdp:2.6.5

docker exec -t "sandbox-hdp" sh -c "rm -rf /var/run/postgresql/*; systemctl restart postgresql;"
```



```
docker run --privileged --name sandbox-hdp --network host -h sandbox-hdp.hortonworks.com -d sandbox-hdp:backup
sandbox/proxy/proxy-deploy.sh 
```





## 非代理启动

```shell
docker run -d --privileged --network host --hostname sandbox-hdp sandbox-hdp:backup
```





### 使用

a) 访问 http://master:4200/

输入用户和密码，分别为root/hadoop

b) 修改ambari admin账号密码

在宿主机中`docker ps` 查看`hortonworks/sandbox-hdp:2.6.5` 镜像运行的container; 进入容器

```shell
docker exec -it 121a30af762f /bin/bash
```

如下命令修改密码为admin

```shell
[root@sandbox-hdp /]# ambari-admin-password-reset
Please set the password for admin: 
Please retype the password for admin: 

The admin password has been set.
```

c) 修改本机hosts文件增加

```shell
192.168.1.211 sandbox-hdp.hortonworks.com sandbox-hdp
```

192.168.1.211是docker宿主机ip



d) 访问http://192.168.1.211:8080 





## 问题

1: 启动docker时，报：WARNING: IPv4 forwarding is disabled. Networking will not work

```
[root@master docker]# sh docker-deploy-hdp265.sh 
+ registry=hortonworks
+ name=sandbox-hdp
+ version=2.6.5
+ proxyName=sandbox-proxy
+ proxyVersion=1.0
+ flavor=hdp
+ echo hdp
+ mkdir -p sandbox/proxy/conf.d
+ mkdir -p sandbox/proxy/conf.stream.d
+ docker pull hortonworks/sandbox-hdp:2.6.5
2.6.5: Pulling from hortonworks/sandbox-hdp
Digest: sha256:0b34fa5cb197717828d6ffe547c23ad9b1c09f3b953e570e37f6f09809fbf3ba
Status: Image is up to date for hortonworks/sandbox-hdp:2.6.5
+ docker pull hortonworks/sandbox-proxy:1.0
1.0: Pulling from hortonworks/sandbox-proxy
Digest: sha256:42e4cfbcbb76af07e5d8f47a183a0d4105e65a1e7ef39fe37ab746e8b2523e9e
Status: Image is up to date for hortonworks/sandbox-proxy:1.0
+ '[' hdp == hdf ']'
+ '[' hdp == hdp ']'
+ hostname=sandbox-hdp.hortonworks.com
++ docker images
++ grep hortonworks/sandbox-hdp
++ awk '{print $2}'
+ version=2.6.5
+ docker network create cda
+ docker run --privileged --name sandbox-hdp -h sandbox-hdp.hortonworks.com --network=cda --network-alias=sandbox-hdp.hortonworks.com -d hortonworks/sandbox-hdp:2.6.5
WARNING: IPv4 forwarding is disabled. Networking will not work.
3a76127c744f6c259a9ad0c3339cb858b8db116c1dbfc237ccabfacd65f96ef0
+ echo ' Remove existing postgres run files. Please wait'
 Remove existing postgres run files. Please wait
+ sleep 2
+ docker exec -t sandbox-hdp sh -c 'rm -rf /var/run/postgresql/*; systemctl restart postgresql;'
```



参考：https://blog.csdn.net/yjk13703623757/article/details/68939183

```
vim  /usr/lib/sysctl.d/00-system.conf
添加如下代码：

net.ipv4.ip_forward=1

重启network服务

# systemctl restart network
```





2：kafka端口外部无法访问

```
[c:\~]$ telnet master 6667


Host 'master' resolved to 192.168.1.211.
Connecting to 192.168.1.211:6667...
Could not connect to 'master' (port 6667): Connection failed.

Type `help' to learn how to use Xshell prompt.

```





修改sandbox/proxy/conf.stream.d/tcp-hdp.conf ，增加如下：

```
server {
  listen 6667;
  proxy_pass sandbox-hdp:6667;
}

```



重启

```
docker stop sandbox-proxy
sandbox/proxy/proxy-deploy.sh
```



参考：https://community.hortonworks.com/idea/189480/about-hdp-standbox-expose-kafka-port-6667-outside.html

