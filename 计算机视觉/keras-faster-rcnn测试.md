[TOC]



## 安装

环境 python3.6 , keras 2.2.0   tensorflow-gpu 1.9.0   (keras 2.0.3版本报错) 



```shell
git clone https://github.com/yhenon/keras-frcnn
```



训练

```python
!python train_frcnn.py --input_weight_path /opt/pretrained_model/vgg16_weights_tf_dim_ordering_tf_kernels.h5 \
--network vgg \
--path /opt/dataset/00020_annotated_flat/template/VOCLike \
--num_epochs 10
```



```python
!python test_frcnn.py --network vgg \
--path /opt/dataset/00020_annotated_flat/template/VOCLike/VOC2007/JPEGImages \
```







## 知识点

图像resize

generator

iou计算

gt生成(正负样本，anchor,h,w,gt)   

iou>0.7, 每个gt最好的那个iou, 确保每个gt都有一个anchor,



rpn预测边框

rpn to roi(正负样本)



最终预测



keras自定义层

keras自定义损失函数

generator

 A generator or an instance of `Sequence` (`keras.utils.Sequence`)

tensorflow的张量运算



损失函数个数与output相同





## 问题记录

1：

```python
!python train_frcnn.py --input_weight_path /opt/pretrained_model/vgg16_weights_tf_dim_ordering_tf_kernels.h5 \
--network vgg \
--path /opt/dataset/00020_annotated_flat/template/VOCLike \
--num_epochs 10
```



```shell
keep_dims is deprecated, use keepdims instead
Starting training
Epoch 1/10
Exception: You must feed a value for placeholder tensor 'time_distributed_3/keras_learning_phase' with dtype bool
	 [[Node: time_distributed_3/keras_learning_phase = Placeholder[dtype=DT_BOOL, shape=<unknown>, _device="/job:localhost/replica:0/task:0/device:GPU:0"]()]]
	 [[Node: add_6/_371 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_3619_add_6", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]

Caused by op 'time_distributed_3/keras_learning_phase', defined at:
  File "train_frcnn.py", line 128, in <module>
    classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)
  File "/opt/github/keras-frcnn/keras_frcnn/vgg.py", line 116, in classifier
    out = TimeDistributed(Dropout(0.5))(out)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/keras/engine/topology.py", line 578, in __call__
    output = self.call(inputs, **kwargs)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/keras/layers/wrappers.py", line 177, in call
    y = self.layer.call(inputs)  # (num_samples * timesteps, ...)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/keras/layers/core.py", line 111, in call
    training=training)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py", line 2433, in in_train_phase
    training = learning_phase()
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py", line 103, in learning_phase
    name='keras_learning_phase')
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/tensorflow/python/ops/array_ops.py", line 1734, in placeholder
    return gen_array_ops.placeholder(dtype=dtype, shape=shape, name=name)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/tensorflow/python/ops/gen_array_ops.py", line 4924, in placeholder
    "Placeholder", dtype=dtype, shape=shape, name=name)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/tensorflow/python/framework/op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 3414, in create_op
    op_def=op_def)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 1740, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InvalidArgumentError (see above for traceback): You must feed a value for placeholder tensor 'time_distributed_3/keras_learning_phase' with dtype bool
	 [[Node: time_distributed_3/keras_learning_phase = Placeholder[dtype=DT_BOOL, shape=<unknown>, _device="/job:localhost/replica:0/task:0/device:GPU:0"]()]]
	 [[Node: add_6/_371 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_3619_add_6", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]

Exception: You must feed a value for placeholder tensor 'time_distributed_3/keras_learning_phase' with dtype bool
	 [[Node: time_distributed_3/keras_learning_phase = Placeholder[dtype=DT_BOOL, shape=<unknown>, _device="/job:localhost/replica:0/task:0/device:GPU:0"]()]]
	 [[Node: add_6/_371 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_3619_add_6", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]

Caused by op 'time_distributed_3/keras_learning_phase', defined at:
  File "train_frcnn.py", line 128, in <module>
    classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)
  File "/opt/github/keras-frcnn/keras_frcnn/vgg.py", line 116, in classifier
    out = TimeDistributed(Dropout(0.5))(out)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/keras/engine/topology.py", line 578, in __call__
    output = self.call(inputs, **kwargs)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/keras/layers/wrappers.py", line 177, in call
    y = self.layer.call(inputs)  # (num_samples * timesteps, ...)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/keras/layers/core.py", line 111, in call
    training=training)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py", line 2433, in in_train_phase
    training = learning_phase()
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py", line 103, in learning_phase
    name='keras_learning_phase')
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/tensorflow/python/ops/array_ops.py", line 1734, in placeholder
    return gen_array_ops.placeholder(dtype=dtype, shape=shape, name=name)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/tensorflow/python/ops/gen_array_ops.py", line 4924, in placeholder
    "Placeholder", dtype=dtype, shape=shape, name=name)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/tensorflow/python/framework/op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 3414, in create_op
    op_def=op_def)
  File "/root/anaconda3/envs/keras2.0.3/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 1740, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InvalidArgumentError (see above for traceback): You must feed a value for placeholder tensor 'time_distributed_3/keras_learning_phase' with dtype bool
	 [[Node: time_distributed_3/keras_learning_phase = Placeholder[dtype=DT_BOOL, shape=<unknown>, _device="/job:localhost/replica:0/task:0/device:GPU:0"]()]]
	 [[Node: add_6/_371 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_3619_add_6", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]
```



解决方法：升级keras和tensorflow到最新版本



2：



```shell
2018-07-31 10:55:24.341656: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Sum Total of in-use chunks: 7.29GiB
2018-07-31 10:55:24.341669: I tensorflow/core/common_runtime/bfc_allocator.cc:680] Stats: 
Limit:                  7903929959
InUse:                  7825838080
MaxInUse:               7825876992
NumAllocs:                    6813
MaxAllocSize:           2389377024

2018-07-31 10:55:24.341698: W tensorflow/core/common_runtime/bfc_allocator.cc:279] ****************************************************************************************************
2018-07-31 10:55:24.341714: W tensorflow/core/framework/op_kernel.cc:1318] OP_REQUIRES failed at strided_slice_op.cc:247 : Resource exhausted: OOM when allocating tensor with shape[1,270,480,256] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
Exception: OOM when allocating tensor with shape[1,270,480,256] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
	 [[Node: training_1/Adam/gradients/roi_pooling_conv_1/strided_slice_149_grad/StridedSliceGrad = StridedSliceGrad[Index=DT_INT32, T=DT_FLOAT, begin_mask=9, ellipsis_mask=0, end_mask=9, new_axis_mask=0, shrink_axis_mask=0, _device="/job:localhost/replica:0/task:0/device:GPU:0"](training_1/Adam/gradients/roi_pooling_conv_1/strided_slice_149_grad/Shape-0-0-VecPermuteNCHWToNHWC-LayoutOptimizer, roi_pooling_conv_1/strided_slice_149/stack, roi_pooling_conv_1/strided_slice_149/stack_1, roi_pooling_conv_1/strided_slice_4/stack_2, training_1/Adam/gradients/roi_pooling_conv_1/resize_images_29/ResizeBilinear_grad/ResizeBilinearGrad)]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
```



解决方法：num_regions调小

```
num_regions = 32
```



