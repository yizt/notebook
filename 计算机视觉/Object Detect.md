[TOC]

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