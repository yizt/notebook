
```shell
docker pull splendor90/detectron2

```


```
docker run -it --runtime=nvidia \
-v /sdb/tmp:/sdb/tmp --network=host \
--shm-size="1g" --ipc=host \
--name=yizt_abc splendor90/detectron2 /bin/bash

```