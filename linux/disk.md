1. 磁盘类型查看,结构为0是SSD,1是机械盘

```
cat /sys/block/sda/queue/rotational
```

