


```shell
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1
python tools/train.py configs/pascal_voc/ssd300_voc.py
```

多卡训练

```shell
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PORT=6666
./tools/dist_train.sh configs/pascal_voc/ssd300_voc.py 8

```

测试
```shell
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1
python tools/test.py configs/pascal_voc/ssd300_voc.py \
work_dirs/ssd300_voc/epoch_24.pth \
--out result.pkl
```

评估
```shell
python tools/voc_eval.py result.pkl configs/pascal_voc/ssd300_voc.py
```
评估结果
```text
+-------------+------+--------+--------+-------+
| class       | gts  | dets   | recall | ap    |
+-------------+------+--------+--------+-------+
| aeroplane   | 285  | 9343   | 0.898  | 0.704 |
| bicycle     | 337  | 6031   | 0.938  | 0.725 |
| bird        | 459  | 63769  | 0.843  | 0.467 |
| boat        | 263  | 41460  | 0.916  | 0.532 |
| bottle      | 469  | 66717  | 0.691  | 0.228 |
| bus         | 213  | 4781   | 0.915  | 0.711 |
| car         | 1201 | 46635  | 0.946  | 0.793 |
| cat         | 358  | 8527   | 0.936  | 0.707 |
| chair       | 756  | 141257 | 0.894  | 0.384 |
| cow         | 244  | 4513   | 0.955  | 0.566 |
| diningtable | 206  | 9221   | 0.898  | 0.618 |
| dog         | 489  | 8790   | 0.943  | 0.637 |
| horse       | 348  | 3649   | 0.940  | 0.762 |
| motorbike   | 325  | 3403   | 0.926  | 0.746 |
| person      | 4528 | 261760 | 0.936  | 0.690 |
| pottedplant | 480  | 86428  | 0.810  | 0.292 |
| sheep       | 242  | 10271  | 0.909  | 0.613 |
| sofa        | 239  | 3710   | 0.941  | 0.642 |
| train       | 282  | 8875   | 0.954  | 0.766 |
| tvmonitor   | 308  | 21373  | 0.909  | 0.604 |
+-------------+------+--------+--------+-------+
| mAP         |      |        |        | 0.609 |
+-------------+------+--------+--------+-------+
```

```shell
+-------------+------+--------+--------+-------+
| class       | gts  | dets   | recall | ap    |
+-------------+------+--------+--------+-------+
| aeroplane   | 285  | 5904   | 0.902  | 0.767 |
| bicycle     | 337  | 4542   | 0.935  | 0.800 |
| bird        | 459  | 19630  | 0.917  | 0.716 |
| boat        | 263  | 21207  | 0.920  | 0.651 |
| bottle      | 469  | 29705  | 0.791  | 0.390 |
| bus         | 213  | 4349   | 0.981  | 0.794 |
| car         | 1201 | 26649  | 0.963  | 0.831 |
| cat         | 358  | 3359   | 0.958  | 0.865 |
| chair       | 756  | 80524  | 0.940  | 0.531 |
| cow         | 244  | 2997   | 0.971  | 0.773 |
| diningtable | 206  | 6629   | 0.927  | 0.699 |
| dog         | 489  | 4538   | 0.969  | 0.837 |
| horse       | 348  | 2432   | 0.925  | 0.835 |
| motorbike   | 325  | 3062   | 0.935  | 0.796 |
| person      | 4528 | 110623 | 0.938  | 0.745 |
| pottedplant | 480  | 43449  | 0.865  | 0.424 |
| sheep       | 242  | 4853   | 0.930  | 0.721 |
| sofa        | 239  | 3340   | 0.958  | 0.751 |
| train       | 282  | 6367   | 0.947  | 0.838 |
| tvmonitor   | 308  | 14809  | 0.945  | 0.726 |
+-------------+------+--------+--------+-------+
| mAP         |      |        |        | 0.724 |
+-------------+------+--------+--------+-------+
```

## 问题

1. 随机出现AssertionError: classification scores become infinite or NaN!