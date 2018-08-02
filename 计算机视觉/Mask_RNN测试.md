

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

