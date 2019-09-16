[TOC]



## val数据集处理



在jupyter中执行如下如下代码块

```python
import os
import shutil
import tempfile
import torch
```



```python
def parse_devkit(root):
    idx_to_wnid, wnid_to_classes = parse_meta(root)
    val_idcs = parse_val_groundtruth(root)
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
    return wnid_to_classes, val_wnids


def parse_meta(devkit_root, path='data', filename='meta.mat'):
    import scipy.io as sio

    metafile = os.path.join(devkit_root, path, filename)
    meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
            if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(', ')) for clss in classes]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
    return idx_to_wnid, wnid_to_classes


def parse_val_groundtruth(devkit_root, path='data',
                          filename='ILSVRC2012_validation_ground_truth.txt'):
    with open(os.path.join(devkit_root, path, filename), 'r') as txtfh:
        val_idcs = txtfh.readlines()
    return [int(val_idx) for val_idx in val_idcs]


def prepare_train_folder(folder):
    for archive in [os.path.join(folder, archive) for archive in os.listdir(folder)]:
        extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)


def prepare_val_folder(folder, wnids):
    img_files = sorted([os.path.join(folder, file) for file in os.listdir(folder)])

    for wnid in set(wnids):
        os.mkdir(os.path.join(folder, wnid))

    for wnid, img_file in zip(wnids, img_files):
        shutil.move(img_file, os.path.join(folder, wnid, os.path.basename(img_file)))


def _splitexts(root):
    exts = []
    ext = '.'
    while ext:
        root, ext = os.path.splitext(root)
        exts.append(ext)
    return root, ''.join(reversed(exts))
```



```python
ARCHIVE_DICT = {
    'train': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
        'md5': '1d675b47d978889d74fa0da5fadfb00e',
    },
    'val': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
        'md5': '29b22e2961454d5413ddabcf34fc5622',
    },
    'devkit': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz',
        'md5': 'fa75699e90414af021442c21a62c3abf',
    }
}
```





```python
archive_dict = ARCHIVE_DICT['devkit']
devkit_folder = _splitexts(os.path.basename(archive_dict['url']))[0]
meta = parse_devkit(os.path.join('/tmp_train_data/imagenet', devkit_folder))
```





```python
prepare_val_folder('/tmp_train_data/imagenet/val',meta[1])
```



完成后val下建立了相应的目录

```
drwxr-xr-x. 2 root root 4096 8月  19 16:47 n12985857
drwxr-xr-x. 2 root root 4096 8月  19 16:47 n12998815
drwxr-xr-x. 2 root root 4096 8月  19 16:47 n13037406
drwxr-xr-x. 2 root root 4096 8月  19 16:47 n13040303
drwxr-xr-x. 2 root root 4096 8月  19 16:47 n13044778
drwxr-xr-x. 2 root root 4096 8月  19 16:47 n13052670
drwxr-xr-x. 2 root root 4096 8月  19 16:47 n13054560
drwxr-xr-x. 2 root root 4096 8月  19 16:47 n13133613
drwxr-xr-x. 2 root root 4096 8月  19 16:47 n15075141
```







## 预测

```shell
ln -s /dataset/pretrained_model/torch checkpoints
```



```shell
(pytorch) [root@localhost classification]# python train.py --model mobilenet_v2 --pretrained --data-path /tmp_train_data/imagenet/ --test-only 
```



结果如下

```shell
(pytorch) [root@localhost classification]# python train.py --model mobilenet_v2 --pretrained --data-path /tmp_train_data/imagenet/ --test-only 
Not using distributed mode
Namespace(apex=False, apex_opt_level='O1', batch_size=32, cache_dataset=False, data_path='/tmp_train_data/imagenet/', device='cuda', dist_url='env://', distributed=False, epochs=90, lr=0.1, lr_gamma=0.1, lr_step_size=30, model='mobilenet_v2', momentum=0.9, output_dir='.', pretrained=True, print_freq=10, resume='', start_epoch=0, sync_bn=False, test_only=True, weight_decay=0.0001, workers=16, world_size=1)
Loading data
Loading training data
Took 2.7062411308288574
Loading validation data
Creating data loaders
Creating model
Test:  [   0/1563]  eta: 1:04:54  loss: 0.6066 (0.6066)  acc1: 87.5000 (87.5000)  acc5: 96.8750 (96.8750)  time: 2.4919  data: 1.3408  max mem: 966
Test:  [ 100/1563]  eta: 0:01:49  loss: 1.1036 (0.8832)  acc1: 65.6250 (77.2896)  acc5: 90.6250 (93.0384)  time: 0.0459  data: 0.0055  max mem: 966
Test:  [ 200/1563]  eta: 0:01:37  loss: 0.6557 (0.8494)  acc1: 78.1250 (78.6536)  acc5: 93.7500 (93.4391)  time: 0.1060  data: 0.0673  max mem: 966
Test:  [ 300/1563]  eta: 0:01:31  loss: 0.9172 (0.8425)  acc1: 71.8750 (78.2911)  acc5: 93.7500 (93.7189)  time: 0.0750  data: 0.0352  max mem: 966
Test:  [ 400/1563]  eta: 0:01:23  loss: 0.5586 (0.8491)  acc1: 84.3750 (77.6886)  acc5: 96.8750 (93.8825)  time: 0.0383  data: 0.0024  max mem: 966
Test:  [ 500/1563]  eta: 0:01:18  loss: 0.8398 (0.8547)  acc1: 75.0000 (77.6509)  acc5: 93.7500 (94.0182)  time: 0.0682  data: 0.0342  max mem: 966
Test:  [ 600/1563]  eta: 0:01:12  loss: 0.9446 (0.8461)  acc1: 71.8750 (77.9742)  acc5: 90.6250 (94.1452)  time: 0.0867  data: 0.0475  max mem: 966
Test:  [ 700/1563]  eta: 0:01:04  loss: 1.4117 (0.8990)  acc1: 62.5000 (76.7475)  acc5: 87.5000 (93.5494)  time: 0.0602  data: 0.0221  max mem: 966
Test:  [ 800/1563]  eta: 0:00:56  loss: 1.4240 (0.9746)  acc1: 65.6250 (75.2536)  acc5: 84.3750 (92.5718)  time: 0.0646  data: 0.0250  max mem: 966
Test:  [ 900/1563]  eta: 0:00:49  loss: 0.8852 (0.9994)  acc1: 78.1250 (74.8543)  acc5: 93.7500 (92.1823)  time: 0.0621  data: 0.0244  max mem: 966
Test:  [1000/1563]  eta: 0:00:41  loss: 1.3385 (1.0445)  acc1: 59.3750 (73.9979)  acc5: 87.5000 (91.5616)  time: 0.0711  data: 0.0333  max mem: 966
Test:  [1100/1563]  eta: 0:00:34  loss: 1.1425 (1.0766)  acc1: 65.6250 (73.3197)  acc5: 90.6250 (91.2210)  time: 0.0713  data: 0.0308  max mem: 966
Test:  [1200/1563]  eta: 0:00:26  loss: 1.2658 (1.1043)  acc1: 68.7500 (72.8351)  acc5: 87.5000 (90.8436)  time: 0.0693  data: 0.0323  max mem: 966
Test:  [1300/1563]  eta: 0:00:19  loss: 1.0486 (1.1274)  acc1: 71.8750 (72.4803)  acc5: 90.6250 (90.5289)  time: 0.0781  data: 0.0377  max mem: 966
Test:  [1400/1563]  eta: 0:00:12  loss: 1.1676 (1.1499)  acc1: 68.7500 (71.8951)  acc5: 87.5000 (90.1990)  time: 0.0711  data: 0.0328  max mem: 966
Test:  [1500/1563]  eta: 0:00:04  loss: 0.6743 (1.1482)  acc1: 81.2500 (71.9000)  acc5: 96.8750 (90.2461)  time: 0.0580  data: 0.0203  max mem: 966
Test: Total time: 0:01:54
 * Acc@1 71.878 Acc@5 90.286

```



shufflenet_v2





```shell
export CUDA_VISIBLE_DEVICES=1
(pytorch) [root@localhost classification]# python train.py --model shufflenet_v2_x1_0 --pretrained --data-path /tmp_train_data/imagenet/ --test-only 
```



结果如下:

```
(pytorch) [root@localhost classification]# python train.py --model shufflenet_v2_x1_0 --pretrained --data-path /tmp_train_data/imagenet/ --test-only 
Not using distributed mode
Namespace(apex=False, apex_opt_level='O1', batch_size=32, cache_dataset=False, data_path='/tmp_train_data/imagenet/', device='cuda', dist_url='env://', distributed=False, epochs=90, lr=0.1, lr_gamma=0.1, lr_step_size=30, model='shufflenet_v2_x1_0', momentum=0.9, output_dir='.', pretrained=True, print_freq=10, resume='', start_epoch=0, sync_bn=False, test_only=True, weight_decay=0.0001, workers=16, world_size=1)
Loading data
Loading training data
Took 2.630046844482422
Loading validation data
Creating data loaders
Creating model
Test:  [   0/1563]  eta: 0:58:20  loss: 0.7547 (0.7547)  acc1: 90.6250 (90.6250)  acc5: 93.7500 (93.7500)  time: 2.2394  data: 1.3672  max mem: 1115
Test:  [ 100/1563]  eta: 0:01:41  loss: 1.4606 (1.1060)  acc1: 65.6250 (75.8663)  acc5: 87.5000 (91.4913)  time: 0.0417  data: 0.0154  max mem: 1115
Test:  [ 200/1563]  eta: 0:01:24  loss: 1.0528 (1.0372)  acc1: 75.0000 (76.9590)  acc5: 90.6250 (92.0243)  time: 0.0849  data: 0.0597  max mem: 1115
Test:  [ 300/1563]  eta: 0:01:11  loss: 1.1897 (1.0090)  acc1: 68.7500 (76.8999)  acc5: 90.6250 (92.3484)  time: 0.0405  data: 0.0119  max mem: 1115
Test:  [ 400/1563]  eta: 0:01:03  loss: 0.7395 (1.0097)  acc1: 81.2500 (76.1767)  acc5: 96.8750 (92.7759)  time: 0.0509  data: 0.0236  max mem: 1115
Test:  [ 500/1563]  eta: 0:00:57  loss: 1.3585 (1.0274)  acc1: 71.8750 (75.8483)  acc5: 90.6250 (92.8518)  time: 0.0409  data: 0.0106  max mem: 1115
Test:  [ 600/1563]  eta: 0:00:52  loss: 1.0460 (1.0181)  acc1: 75.0000 (76.0919)  acc5: 90.6250 (93.0012)  time: 0.0851  data: 0.0583  max mem: 1115
Test:  [ 700/1563]  eta: 0:00:45  loss: 1.8500 (1.0852)  acc1: 59.3750 (74.7325)  acc5: 84.3750 (92.1229)  time: 0.0347  data: 0.0044  max mem: 1115
Test:  [ 800/1563]  eta: 0:00:40  loss: 1.7843 (1.1678)  acc1: 56.2500 (73.1664)  acc5: 84.3750 (91.0385)  time: 0.0450  data: 0.0178  max mem: 1115
Test:  [ 900/1563]  eta: 0:00:34  loss: 1.0359 (1.1959)  acc1: 75.0000 (72.6658)  acc5: 90.6250 (90.5834)  time: 0.0567  data: 0.0299  max mem: 1115
Test:  [1000/1563]  eta: 0:00:29  loss: 1.5820 (1.2437)  acc1: 59.3750 (71.7251)  acc5: 87.5000 (89.8664)  time: 0.0447  data: 0.0173  max mem: 1115
Test:  [1100/1563]  eta: 0:00:24  loss: 1.3063 (1.2777)  acc1: 68.7500 (71.0859)  acc5: 87.5000 (89.4556)  time: 0.0355  data: 0.0087  max mem: 1115
Test:  [1200/1563]  eta: 0:00:18  loss: 1.6019 (1.3102)  acc1: 68.7500 (70.4803)  acc5: 84.3750 (88.9935)  time: 0.0439  data: 0.0159  max mem: 1115
Test:  [1300/1563]  eta: 0:00:13  loss: 1.4133 (1.3379)  acc1: 65.6250 (69.9438)  acc5: 84.3750 (88.5905)  time: 0.0431  data: 0.0165  max mem: 1115
Test:  [1400/1563]  eta: 0:00:08  loss: 1.6146 (1.3616)  acc1: 62.5000 (69.3656)  acc5: 84.3750 (88.2718)  time: 0.0532  data: 0.0276  max mem: 1115
Test:  [1500/1563]  eta: 0:00:03  loss: 0.7648 (1.3606)  acc1: 81.2500 (69.3683)  acc5: 93.7500 (88.3057)  time: 0.0645  data: 0.0375  max mem: 1115
Test: Total time: 0:01:19
 * Acc@1 69.362 Acc@5 88.316

```





## 训练

a) 单卡训练

```shell
cd /home/github/vision/references/classification
export CUDA_VISIBLE_DEVICES=1
nohup python train.py --model shufflenet_v2_x1_0 --data-path /tmp_train_data/imagenet/ --batch-size 256 &
```



```python
nohup python train.py --model shufflenet_v2_x1_0 --data-path /tmp_train_data/imagenet/ --batch-size 256 --args.resume &
```



b)多卡训练

   修改train.py增加`parser.add_argument('--local_rank', type=int, default=0)`

```bash
python -m torch.distributed.launch --nproc_per_node 2 train.py --model shufflenet_v2_x1_0 --data-path /tmp_train_data/imagenet/
```

   日志如下：

```bash
(pytorch) [root@localhost classification]# python -m torch.distributed.launch --nproc_per_node 2 train.py --model shufflenet_v2_x1_0 --data-path /tmp_train_data/imagenet/
| distributed init (rank 1): env://
| distributed init (rank 0): env://
Namespace(apex=False, apex_opt_level='O1', batch_size=32, cache_dataset=False, data_path='/tmp_train_data/imagenet/', device='cuda', dist_backend='nccl', dist_url='env://', distributed=True, epochs=90, gpu=0, local_rank=0, lr=0.1, lr_gamma=0.1, lr_step_size=30, model='shufflenet_v2_x1_0', momentum=0.9, output_dir='.', pretrained=False, print_freq=10, rank=0, resume='', start_epoch=0, sync_bn=False, test_only=False, weight_decay=0.0001, workers=16, world_size=2)
Loading data
Loading training data
Took 2.669806718826294
Loading validation data
Creating data loaders
Creating model
Start training
```

