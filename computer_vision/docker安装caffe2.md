

### 构建caffe2镜像

a) 下载

```shell
cd /opt/github
git clone https://github.com/pytorch/pytorch.git
```



b) 宿主机执行如下命令，将.pip复制到Dockerfile所在目录

```shell
cd /opt/github/pytorch/docker/caffe2/ubuntu-16.04-cuda8-cudnn7-all-options
cp -rp /root/.pip ./
```



c) 在中增加Dockerfile中增加(copy的文件一定要在Dockerfile所在目录)

```shell
COPY .pip /root/.pip
```



d) 修改Dockerfile中pip安装的版本，以及环境变量等



最终的Dockerfile文件如下：

```dockerfile
FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
LABEL maintainer="aaronmarkham@fb.com"

# caffe2 install with gpu support

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libprotobuf-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pip \
    python-pydot \
    python-setuptools \
    python-scipy \
    wget \
    && rm -rf /var/lib/apt/lists/*
COPY .pip /root/.pip
RUN pip install --no-cache-dir --upgrade pip==9.0.3 setuptools wheel && \
    pip install --no-cache-dir \
    flask \
    future \
    graphviz \
    hypothesis \
    ipykernel==4.8.2 \
    ipython==5.4.1 \
    jupyter-console==5.0.0 \
    jupyter==1.0.0 \
    matplotlib==2.2.2 \
    numpy \
    protobuf \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    six \
    tornado

########## INSTALLATION STEPS ###################
RUN git clone --branch master --recursive https://github.com/pytorch/pytorch.git
RUN pip install typing
RUN cd pytorch && mkdir build && cd build \
    && cmake .. \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="35 52 60 61" \
    -DCUDA_ARCH_PTX="61" \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    && make -j"$(nproc)" install \
    && ldconfig \
    && cd .. \

ENV PYTHONPATH /pytorch/build
```



e) 构建docker 镜像，命名为caffe2:v1

```shell
cd /opt/github/pytorch/docker/caffe2/ubuntu-16.04-cuda8-cudnn7-all-options
docker build -t caffe2:v1 .
```





### 测试docker 镜像

a) 运行容器(用nvidia-docker)

```shell
nvidia-docker run -it caffe2:v1 /bin/bash
```

 b) 测试

```shell
cd /pytorch/build/
python2 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```



结果如下(构建成功)：

```
root@localhost:/pytorch# cd /pytorch/build/
root@localhost:/pytorch/build# python2 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
Success
root@localhost:/pytorch/build# python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
1

```



