```python
yum update
```



```
uname -a 
```



```bash
yum install docker-ce
yum install -y nvidia-container-toolkit
systemctl restart docker
```



```shell
#### Test nvidia-smi with the latest official CUDA image
docker run --gpus all nvidia/cuda:10.0-base nvidia-smi

# Start a GPU enabled container on two GPUs
docker run --gpus 2 nvidia/cuda:10.0-base nvidia-smi

# Starting a GPU enabled container on specific GPUs
docker run --gpus '"device=1,2"' nvidia/cuda:10.0-base nvidia-smi
docker run --gpus '"device=UUID-ABCDEF,1"' nvidia/cuda:10.0-base nvidia-smi

# Specifying a capability (graphics, compute, ...) for my container
# Note this is rarely if ever used this way
docker run --gpus all,capabilities=utility nvidia/cuda:10.0-base nvidia-smi

docker run --gpus all pytorch/pytorch nvidia-smi
```



搜索镜像

https://hub.docker.com/

<https://gitlab.com/nvidia/container-images/cuda/-/tree/master/dist>



```
docker pull nvidia/cuda:10.2-devel-ubuntu18.04
```





样例

```shell
docker run -it --gpus all --net host -v /home/mydir:/home/mydir --name yizt pytorch:adelaidet /bin/bash

docker run -it --gpus all --net host -v /home/mydir:/home/mydir --name torch1.4 pytorch/pytorch:1.4-cuda10.1-cudnn7-devel /bin/bash

docker run -it --gpus all --net host -v /home/mydir:/home/mydir --name torch1.4.adelaidet torch1.4:adelaidet /bin/bash

docker run -it --gpus all --net host -v /home/mydir:/home/mydir --name torch1.5 pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel /bin/bash



docker run -it --gpus all --net host -v /home/mydir:/home/mydir --name tf tensorflow/tensorflow /bin/bash
#Adelaidet fatal error: cusparse.h: No such file or directory

docker run -it --gpus all --net host -v /home/mydir:/home/mydir --name cuda nvidia/cuda:10.2-devel-ubuntu18.04 /bin/bash
# pytorch system has unsupported display driver / cuda driver combination 


docker run -it --runtime=nvidia --net host -v /home/mydir:/home/mydir --name yizt pytorch:adelaidet /bin/bash
```

