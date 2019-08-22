## 训练

```
export MXNET_CPU_WORKER_NTHREADS=6
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py \
    --network r100 \
    --loss-type 4 \
    --margin-m 0.5 \
    --per-batch-size 32 \
    --ckpt 1 \
    --lr 0.01 \
    --data-dir /home/dataset/face_recognize/faces_ms1m_112x112 \
    --prefix ../model/model-r100 >/opt/github/insightface/src/train.log.txt 2>&1
```

输出如下:

```
gpu num: 1
num_layers 100
image_size [112, 112]
num_classes 85164
Called with argument: Namespace(batch_size=32, beta=1000.0, beta_freeze=0, beta_min=5.0, bn_mom=0.9, ckpt=1, ctx_num=1, cutoff=0, data_dir='/home/dataset/face_recognize/faces_ms1m_112x112', easy_margin=0, emb_size=512, end_epoch=100000, fc7_wd_mult=1.0, gamma=0.12, image_channel=3, image_h=112, image_w=112, loss_type=4, lr=0.01, lr_steps='', margin=4, margin_a=1.0, margin_b=0.0, margin_m=0.5, margin_s=64.0, max_steps=0, mom=0.9, network='r100', num_classes=85164, num_layers=100, per_batch_size=32, power=1.0, prefix='../model/model-r100', pretrained='', rand_mirror=1, rescale_threshold=0, scale=0.9993, target='lfw,cfp_fp,agedb_30', use_deformable=0, verbose=2000, version_act='prelu', version_input=1, version_output='E', version_se=0, version_unit=3, wd=0.0005)
init resnet 100
0 1 E 3 prelu
INFO:root:loading recordio /home/dataset/face_recognize/faces_ms1m_112x112/train.rec...
header0 label [3804847. 3890011.]

```



batch-size、per-batch-size区别：per-batch-size每个GPU的batch-size

注：作者在(4 或者8块) Tesla P40 GPU上训练，默认10w个epoch; 初始学习率0.1, batch-size为512(每个GPU128),训练20w个step；样本量:1343992; 

作者训练的结果: training dataset: ms1m
LFW: 99.50, CFP_FP: 88.94, AgeDB30: 95.91

在一块1080Ti上训练一个epoch用了16个小时。











## 演示demo

```shell
yum install wget unzip git libgomp -y
yum install libXext libSM libXrender -y
yum install libtool -y
yum install libtool-ltdl -y
yum install libtool-ltdl-devel -y
yum install python-devel -y
yum install libevent-devel
# pip安装

pip install mxnet==1.2.0
pip install easydict pandas tornado flask sklearn matplotlib==2.2.2 opencv-python scikit-image
```



```
#python升级
mv /usr/bin/python /usr/bin/python2.7.5
ln -s /usr/local/bin/python2.7 /usr/bin/python
#docker pull mxnet/python

vi /usr/bin/yum、vi /usr/libexec/urlgrabber-ext-down
#!/usr/bin/python to #!/usr/bin/python2.6.6
```





```shell
python /opt/code/face/arcface/rest_1vn/arcface_1vn_search_rest.py
python /opt/code/face/arcface/rest_feature/arcface_feature_rest.py
```



```
docker cp /home/dataset/face_recognize/lfw 291d682b1cc6:/home/dataset/face_recognize/lfw
docker cp /opt/github/insightface/model/ 291d682b1cc6:/opt/github/insightface/model
docker cp /opt/github/insightface/src/eval/kdtree_20w.pkl 291d682b1cc6:/opt/github/insightface/src/eval/kdtree_20w.pkl

docker cp /opt/github/insightface/src/eval/kdtree_3000.pkl 291d682b1cc6:/opt/github/insightface/src/eval/kdtree_3000.pkl
```





```
docker commit -m 'insightface 初始化' f4699dfd1476 arcface-cpu:init
docker commit -m '1vn测试完成' 291d682b1cc6 arcface-cpu:ok

docker save arcface-cpu:ok | gzip > arcface-cpu.tgz
```



