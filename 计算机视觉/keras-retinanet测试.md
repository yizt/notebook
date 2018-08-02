[TOC]



## 安装



环境

keras 2.2.0,python 3.6



下载

```shell
cd /opt/github
git clone https://github.com/fizyr/keras-retinanet
```





下载预训练模型

```shell
cd /opt/pretrained_model
wget -t 0 -c https://github.com/fizyr/keras-retinanet/releases/download/0.4.1/resnet50_coco_best_v2.1.0.h5
```



```shell
cd /opt/github/keras-retinanet/snapshots
ln -s /opt/pretrained_model/resnet50_coco_best_v2.1.0.h5 resnet50_coco_best_v2.1.0.h5
```



```python
import sys
sys.path.append('/root/.local/bin')
```





## 问题记录

1: conda环境下pip和jupyter一定要装，确保是本环境的



2:

```python
!python train.py --imagenet-weights resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
--backbone resnet50 \
--epochs 3 \
```



```python
Using TensorFlow backend.
Traceback (most recent call last):
  File "train.py", line 35, in <module>
    from .. import layers  # noqa: F401
  File "../../keras_retinanet/layers/__init__.py", line 1, in <module>
    from ._misc import RegressBoxes, UpsampleLike, Anchors, ClipBoxes  # noqa: F401
  File "../../keras_retinanet/layers/_misc.py", line 19, in <module>
    from ..utils import anchors as utils_anchors
  File "../../keras_retinanet/utils/anchors.py", line 20, in <module>
    from ..utils.compute_overlap import compute_overlap
ModuleNotFoundError: No module named 'keras_retinanet.utils.compute_overlap'
```

 

解决方法：anchors.py导入前增加两行

```python
import pyximport
pyximport.install()
from ..utils.compute_overlap import compute_overlap
```

