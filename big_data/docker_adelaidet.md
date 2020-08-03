docker run -it -v /sdb/tmp:/sdb/tmp --network=host --shm-size="1g" --ipc=host \
--name=yizt_crnn crnn:init bash



## 24上制作容器记录
docker pull pytorch/pytorch

docker run -it -v /sdb/tmp/users/yizt:/pyspace --network=host --shm-size="1g" --ipc=host \
--name=yizt_crnn pytorch/pytorch /bin/bash

docker exec -it yizt_crnn /bin/bash

# 系统包更新
apt-get update

apt-get install vim  

apt-get install python3

apt-get install python3-pip

apt-get install libglib2.0-dev -y
apt-get install apt-file -y
apt-file update 
apt-file search libSM.so.6
apt-get install libsm6 -y
apt-get install libxrender1 -y

apt-get install libxext-dev -y


# python 依赖包
pip install opencv-python
pip install tornado
pip install flask
pip install flask_cors
pip install cython
pip install pytest-runner
pip install mmcv==0.5.9
pip install python-Levenshtein




# 启动rest
python rest.py --net crnn --weight-path crnn.054.pth.bak0629 --device cpu --port 5001


# 保存为image
docker commit -a 'yizt' -m 'crnn init' 1ea6b4041521 crnn:init
# 导出容器
cd /sdb/tmp/users/yizt/soft
docker save -o crnn.init.tar crnn:init

####### Adelaidet工程安装###
# 启动新容器
docker stop 1ea6b4041521
docker rm 1ea6b4041521

docker run -it -v /sdb/tmp/users/yizt:/pyspace --network=host --shm-size="1g" --ipc=host \
--name=yizt_crnn crnn:init /bin/bash

docker run -it --runtime=nvidia -v /sdb/tmp/users/yizt:/pyspace --network=host --shm-size="1g" --ipc=host \
--name=yizt_crnn_cuda crnn:init /bin/bash


# python包安装

pip install pycocotools>=2.0.1
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

pip install Shapely
pip install Polygon3

git clone https://github.com/aim-uofa/AdelaiDet.git
cd AdelaiDet

python setup.py build develop

# 保存为image
docker commit -a 'yizt' -m 'crnn adelaidet' 9ca4455f4d0b crnn:adelaidet
# 导出容器
cd /sdb/tmp/users/yizt/soft
docker save -o crnn.adelaidet.tar crnn:adelaidet

# 启动新容器
docker stop 9ca4455f4d0b
docker rm 9ca4455f4d0b

docker run -it -v /sdb/tmp/users/yizt:/pyspace -v /sdb/tmp:/sdb/tmp --network=host --shm-size="1g" --ipc=host \
--name=yizt_crnn crnn:adelaidet /bin/bash



## EAST工程

apt-get install unzip

pip install tensorflow==1.14.0
pip install scipy

## PAN工程
pip install sklearn

## DifferentiableBinarinzation工程测试
pip install Keras==2.2.5
pip install keras-resnet==0.2.0
pip install pyclipper
pip install tensorflow==1.14.0
pip install scikit-image

python inference.py ../0705/


# 保存为image
docker commit -a 'yizt' -m 'crnn db' e7ff9f3bde4f crnn:db
# 导出容器
cd /sdb/tmp/users/yizt/soft
docker save -o crnn.db.tar crnn:db

# 启动新容器
docker stop e7ff9f3bde4f
docker rm e7ff9f3bde4f

docker run -it -v /sdb/tmp/users/yizt:/pyspace -v /sdb/tmp:/sdb/tmp --network=host --shm-size="1g" --ipc=host \
--name=yizt_crnn crnn:db /bin/bash

nvidia-docker run -it -v /sdb/tmp/users/yizt:/pyspace -v /sdb/tmp:/sdb/tmp --network=host --shm-size="1g" --ipc=host \
--name=yizt_crnn crnn:db /bin/bash

## download list
http://pypi.doubanio.com/packages/d7/ee/753ea56fda5bc2a5516a1becb631bf5ada593a2dd44f21971a13a762d4db/scikit_image-0.17.2-cp37-cp37m-manylinux1_x86_64.whl
