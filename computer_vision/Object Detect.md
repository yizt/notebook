[TOC]

## 知识点

a) 通用padding到固定尺寸,删除padding

b) 图像加载

c) 图像resize、图像元数据信息、GT boxes调整

d) 图像还原

1：anchor生成

2：IoU、非极大抑制(nms)

3：rpn分类和回归目标

4：应用回归目标生成proposal

5：RoIAlign

6：rpn 分类和回归损失(类别无关、anchors映射)

7：rcnn分类和回归目标(类别相关)

8：应用rcnn回归目标生成最终检测框



### IoU计算

#### numpy

```python
def iou_np(boxes_a, boxes_b):
    """
    numpy 计算IoU
    :param boxes_a: (N,4)
    :param boxes_b: (M,4)
    :return:  IoU (N,M)
    """
    # 扩维
    boxes_a = np.expand_dims(boxes_a, axis=1)  # (N,1,4)
    boxes_b = np.expand_dims(boxes_b, axis=0)  # (1,M,4)

    # 分别计算高度和宽度的交集
    min_y2 = np.where(boxes_a[..., 2] < boxes_b[..., 2], boxes_a[..., 2], boxes_b[..., 2])  # (N,M)
    max_y1 = np.where(boxes_a[..., 0] > boxes_b[..., 0], boxes_a[..., 0], boxes_b[..., 0])
    overlap_h = np.where(min_y2 < max_y1, 0, min_y2 - max_y1 + 1)

    min_x2 = np.where(boxes_a[..., 3] < boxes_b[..., 3], boxes_a[..., 3], boxes_b[..., 3])
    max_x1 = np.where(boxes_a[..., 1] > boxes_b[..., 1], boxes_a[..., 1], boxes_b[..., 1])
    overlap_w = np.where(min_x2 < max_x1, 0, min_x2 - max_x1 + 1)

    # 交集
    overlap = overlap_w * overlap_h

    # 计算面积
    area_a = (boxes_a[..., 2] - boxes_a[..., 0] + 1) * (boxes_a[..., 3] - boxes_a[..., 1] + 1)
    area_b = (boxes_b[..., 2] - boxes_b[..., 0] + 1) * (boxes_b[..., 3] - boxes_b[..., 1] + 1)

    # 交并比
    iou = overlap / (area_a + area_b - overlap)
    return iou

```



#### tf

```python
def iou_tf(gt_boxes, anchors):
    """
    tf 计算iou
    :param gt_boxes: [N,(y1,x1,y2,x2)]
    :param anchors: [M,(y1,x1,y2,x2)]
    :return: IoU [N,M]
    """
    gt_boxes = tf.expand_dims(gt_boxes, axis=1)  # [N,1,4]
    anchors = tf.expand_dims(anchors, axis=0)  # [1,M,4]
    # 交集
    intersect_w = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 3], anchors[:, :, 3]) -
                             tf.maximum(gt_boxes[:, :, 1], anchors[:, :, 1]))
    intersect_h = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 2], anchors[:, :, 2]) -
                             tf.maximum(gt_boxes[:, :, 0], anchors[:, :, 0]))
    intersect = intersect_h * intersect_w

    # 计算面积
    area_gt = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1]) * \
              (gt_boxes[:, :, 2] - gt_boxes[:, :, 0])
    area_anchor = (anchors[:, :, 3] - anchors[:, :, 1]) * \
                  (anchors[:, :, 2] - anchors[:, :, 0])

    # 计算并集
    union = area_gt + area_anchor - intersect
    # 交并比
    iou = tf.divide(intersect, union, name='regress_target_iou')
    return iou

```





### 边框可视化

```python
def random_colors(num, bright=True):
    """
    随机生成颜色
    :param num:
    :param bright:
    :return:  [(r,g,b),...]  列表，长度num
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / num, 1, brightness) for i in range(num)]  # 色调（H），饱和度（S），明度（V）
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
  
def display_boxes(image, boxes, class_ids, class_names,
                  scores=None, title="",
                  figsize=(16, 16), ax=None,
                  show_bbox=True,
                  colors=None, captions=None):
    """
    # 边框可视化
    :param image: 图像的numpy数组
    :param boxes: 边框[N,(y1,x1,y2,x2)]
    :param class_ids: 
    :param class_names:
    :param scores:
    :param title:
    :param figsize:
    :param ax:
    :param show_bbox:
    :param colors: 颜色列表  [(r,g,b)...]
    :param captions: 边框标题列表
    :return:
    """
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # 生成随机颜色
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()

```

画边框用`patches.Rectangle`; 增加标题用`ax.text`





## 工程测试



### ssds.pytorch



```
git clone https://github.com/ShuangXieIrene/ssds.pytorch
```



```
python train.py --cfg=./experiments/cfgs/ssd_lite_mobilenetv2_train_voc.yml
```



遇到问题：错误太多了

```python
  File "/root/anaconda3/envs/pytorch/lib/python3.6/multiprocessing/connection.py", line 493, in Client
    answer_challenge(c, authkey)
  File "/root/anaconda3/envs/pytorch/lib/python3.6/multiprocessing/connection.py", line 737, in answer_challenge
    response = connection.recv_bytes(256)        # reject large message
  File "/root/anaconda3/envs/pytorch/lib/python3.6/multiprocessing/connection.py", line 216, in recv_bytes
    buf = self._recv_bytes(maxlength)
  File "/root/anaconda3/envs/pytorch/lib/python3.6/multiprocessing/connection.py", line 407, in _recv_bytes
    buf = self._recv(4)
  File "/root/anaconda3/envs/pytorch/lib/python3.6/multiprocessing/connection.py", line 379, in _recv
    chunk = read(handle, remaining)
ConnectionResetError: [Errno 104] Connection reset by peer
```



### mmdection

```python
python tools/train.py configs/retinanet_r50_fpn_1x.py
```



```
  File "/home/github/mmdetection/mmdet/ops/dcn/functions/deform_conv.py", line 5, in <module>
    from .. import deform_conv_cuda
ImportError: cannot import name 'deform_conv_cuda'
```





### vision



```shell
cd /home/github/vision/references/detection
export CUDA_VISIBLE_DEVICES=0
python train.py --data-path /home/dataset/coco --model fasterrcnn_resnet50_fpn
```



```shell
  File "/root/anaconda3/envs/pytorch/lib/python3.6/site-packages/torchvision/datasets/coco.py", line 112, in __getitem__

    img = Image.open(os.path.join(self.root, path)).convert('RGB')
  File "/root/anaconda3/envs/pytorch/lib/python3.6/site-packages/PIL/Image.py", line 2770, in open
    fp = builtins.open(filename, "rb")
FileNotFoundError: [Errno 2] No such file or directory: '/home/dataset/coco/train2014/000000447187.jpg'
```





### PyTorch-YOLOv3



```shell
https://github.com/eriklindernoren/PyTorch-YOLOv3.git
cd PyTorch-YOLOv3
```



训练

```shell
export CUDA_VISIBLE_DEVICES=1
python train.py --data_config config/coco.data  --pretrained_weights weights/darknet53.conv.74
```



预测

```shell
python detect.py --image_folder data/samples/ --weights_path checkpoints/yolov3_ckpt_3.pth --conf_thres 0.6
```

