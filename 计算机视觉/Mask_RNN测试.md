[TOC]

## 项目信息

项目地址：https://github.com/matterport/Mask_RCNN

docker地址：https://hub.docker.com/r/waleedka/modern-deep-learning/



FPN的地址: https://github.com/DeanDon/FPN-keras

docker pull waleedka/modern-deep-learning



```shell
docker run -it -p 8888:8888 -p 6006:6006 -v /opt/github/Mask_RCNN-master:/host waleedka/modern-deep-learning
```



```
pip3 install imgaug
```



## 源码分析

### ProposalLayer

```
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """
```

根据anchor分数和NMS选择anchor;并进行回归生成最终的proposals



#### tf.image.non_max_suppression

```python
  """Greedily selects a subset of bounding boxes in descending order of score.

  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system.  Note that this
  algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather operation`.  For example:
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size, iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)

  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    name: A name for the operation (optional).
    
```



### DetectionTargetLayer

```python
"""Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinements.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
```

采样proposals生成检测边框的分类目标和回归目标、



#### overlaps_graph

​        tf的iou计算

## 错误记录

1：

```
Traceback (most recent call last):
  File "train_VOC_fpn.py", line 233, in <module>
    layers='heads')
  File "/opt/github/FPN-keras/model_FPN.py", line 2185, in train
    "validation_data": next(val_generator),
  File "/opt/github/FPN-keras/model_FPN.py", line 1657, in data_generator
    load_image_gt_without_mask(dataset, config, image_id, augment=augment, use_mini_mask=config.USE_MINI_MASK)
  File "/opt/github/FPN-keras/model_FPN.py", line 1276, in load_image_gt_without_mask
    image = dataset.load_image(image_id)
  File "/opt/github/FPN-keras/utils.py", line 358, in load_image
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
cv2.error: /io/opencv/modules/imgproc/src/color.cpp:11079: error: (-215) scn == 3 || scn == 4 in function cvtColor
```



解决方法：使用skimage



2：

```shell
mrcnn_class_logits     (TimeDistributed)
/root/anaconda3/envs/keras2.0.8/lib/python3.5/site-packages/tensorflow/python/ops/gradients_impl.py:100: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
Epoch 1/2
Exception in thread Thread-3:
Traceback (most recent call last):
  File "/root/anaconda3/envs/keras2.0.8/lib/python3.5/threading.py", line 914, in _bootstrap_inner
    self.run()
  File "/root/anaconda3/envs/keras2.0.8/lib/python3.5/threading.py", line 862, in run
    self._target(*self._args, **self._kwargs)
  File "/root/anaconda3/envs/keras2.0.8/lib/python3.5/site-packages/keras/utils/data_utils.py", line 568, in data_generator_task
    generator_output = next(self._generator)
ValueError: generator already executing

annotation_path:/opt/dataset/00020_annotated_flat/template/annotations/image-000047.template060.xml
annotation_path:/opt/dataset/00020_annotated_flat/template/annotations/image-000138.template060.xml
Traceback (most recent call last):
  File "train_VOC_fpn.py", line 234, in <module>
    layers='heads')
  File "/opt/github/FPN-keras/model_FPN.py", line 2202, in train
    **fit_kwargs
  File "/root/anaconda3/envs/keras2.0.8/lib/python3.5/site-packages/keras/legacy/interfaces.py", line 87, in wrapper
    return func(*args, **kwargs)
  File "/root/anaconda3/envs/keras2.0.8/lib/python3.5/site-packages/keras/engine/training.py", line 2011, in fit_generator
    generator_output = next(output_generator)
StopIteration
```



解决方法：

```python
"use_multiprocessing": True,
```





3: 

```shell
Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x7fc10f5e5940>>
Traceback (most recent call last):
  File "/root/anaconda3/envs/keras2.0.8/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 707, in __del__
TypeError: 'NoneType' object is not callable
```

