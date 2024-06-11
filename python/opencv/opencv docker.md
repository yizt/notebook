

安装启动

```shell
docker pull spmallick/opencv-docker:opencv

docker run --device=/dev/video0:/dev/video0 --name opencv -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -p 5000:5000 -p 8888:8888 -it spmallick/opencv-docker:opencv /bin/bash

docker run -v /Users/yizuotian/cspace:/cspace --name opencv -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -p 5000:5000 -p 8888:8888 -it spmallick/opencv-docker:opencv /bin/bash


```



查看版本

```shell
workon OpenCV-3.4.4-py3
ipython
import cv2
cv2.__version__ 
exit()
```

