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



样例

```
docker run -it --gpus all --net host -v /home/mydir:/home/mydir pytorch/pytorch /bin/bash
```

