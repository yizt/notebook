
```shell
docker pull splendor90/detectron2

```


```
docker run -it --runtime=nvidia \
-v /sdb/tmp:/sdb/tmp --network=host \
--shm-size="1g" --ipc=host \
--name=yizt_abc splendor90/detectron2 /bin/bash

```



## mac 下测试

安装
```shell
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ pip install -e .

```

依赖安装
```shell
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py install
```

demo 测试
 
```shell
export KMP_DUPLICATE_LIB_OK=TRUE
python demo/demo.py --config-file configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml \
  --input /Users/yizuotian/pyspace/notebook/data/detection/009573.jpg \
  --opts MODEL.WEIGHTS /Users/yizuotian/pretrained_model/model_final_b1acc2.pkl MODEL.DEVICE cpu
```


## ubuntu下测试

```shell
cd /sdb/tmp/users/yizt/detectron2
cd 
ln -s /sdb/tmp/open_dataset/COCO coco
```

```shell

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
python tools/train_net.py --num-gpus 4 \
	--config-file configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml

```