1. 磁盘类型查看,结构为0是SSD,1是机械盘

```
cat /sys/block/sda/queue/rotational
```



2、查看待挂载磁盘

```shell
lsblk -lp
```





3、磁盘挂载

```shell
sudo mount /dev/sda /mnt
```

