[TOC]

## 训练VOC

### 转换数据

```shell
WORK_DIR=/opt/dataset
PASCAL_ROOT="${WORK_DIR}/VOCdevkit/VOC2012"

# Remove the colormap in the ground truth annotations.
SEG_FOLDER="${PASCAL_ROOT}/SegmentationClass"
SEMANTIC_SEG_FOLDER="${PASCAL_ROOT}/SegmentationClassRaw"

export PYTHONPATH=$PYTHONPATH:/opt/github/TFmodels/research:/opt/github/TFmodels/research/slim
echo "Removing the color map in ground truth annotations..."
python ./remove_gt_colormap.py \
  --original_gt_folder="${SEG_FOLDER}" \
  --output_dir="${SEMANTIC_SEG_FOLDER}"


# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${PASCAL_ROOT}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${PASCAL_ROOT}/JPEGImages"
LIST_FOLDER="${PASCAL_ROOT}/ImageSets/Segmentation"

echo "Converting PASCAL VOC 2012 dataset..."
python ./build_voc2012_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"
```













## 问题记录

1：

```python
device:GPU:0 with 7543 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:02:00.0, compute capability: 6.1)
>> Converting image 1/1464 shard 0Traceback (most recent call last):
  File "./build_voc2012_data.py", line 142, in <module>
    tf.app.run()
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/platform/app.py", line 126, in run
    _sys.exit(main(argv))
  File "./build_voc2012_data.py", line 138, in main
    _convert_dataset(dataset_split)
  File "./build_voc2012_data.py", line 117, in _convert_dataset
    image_data = tf.gfile.FastGFile(image_filename, 'r').read()
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/lib/io/file_io.py", line 127, in read
    pywrap_tensorflow.ReadFromStream(self._read_buf, length, status))
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/lib/io/file_io.py", line 95, in _prepare_value
    return compat.as_str_any(val)
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/util/compat.py", line 111, in as_str_any
    return as_str(value)
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/util/compat.py", line 88, in as_text
    return bytes_or_text.decode(encoding)
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte

```



解决方法: build_voc2012_data.py 文件 r改为rb

```
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()

```



参考：https://github.com/tensorflow/models/issues/3903



2：

```python
2018-08-10 15:20:11.489101: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7543 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:02:00.0, compute capability: 6.1)
>> Converting image 1/1464 shard 0Traceback (most recent call last):
  File "./build_voc2012_data.py", line 142, in <module>
    tf.app.run()
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/platform/app.py", line 126, in run
    _sys.exit(main(argv))
  File "./build_voc2012_data.py", line 138, in main
    _convert_dataset(dataset_split)
  File "./build_voc2012_data.py", line 129, in _convert_dataset
    image_data, filenames[i], height, width, seg_data)
  File "/opt/github/TFmodels/research/deeplab/datasets/build_data.py", line 146, in image_seg_to_tfexample
    'image/filename': _bytes_list_feature(filename),
  File "/opt/github/TFmodels/research/deeplab/datasets/build_data.py", line 128, in _bytes_list_feature
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
TypeError: '2007_000032' has type str, but expected one of: bytes

```



解决方法：修改build_data.py文件

```python
def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  x = bytes(values,encoding= 'utf-8') if isinstance(values,str) else values
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))
```





3：

```python
/root/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
WARNING:tensorflow:From /root/anaconda3/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/base.py:198: retry (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Use the retry module or similar alternatives.
INFO:tensorflow:Training on trainval set
Traceback (most recent call last):
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 510, in _apply_op_helper
    preferred_dtype=default_dtype)
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1040, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/constant_op.py", line 235, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/constant_op.py", line 214, in constant
    value, dtype=dtype, shape=shape, verify_shape=verify_shape))
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/tensor_util.py", line 433, in make_tensor_proto
    _AssertCompatible(values, dtype)
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/tensor_util.py", line 344, in _AssertCompatible
    (dtype.name, repr(mismatch), type(mismatch).__name__))
TypeError: Expected int32, got 4.0 of type 'float' instead.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/github/TFmodels/research/deeplab/train.py", line 347, in <module>
    tf.app.run()
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/platform/app.py", line 126, in run
    _sys.exit(main(argv))
  File "/opt/github/TFmodels/research/deeplab/train.py", line 243, in main
    model_variant=FLAGS.model_variant)
  File "/opt/github/TFmodels/research/deeplab/utils/input_generator.py", line 168, in get
    dynamic_pad=True)
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/training/input.py", line 989, in batch
    name=name)
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/training/input.py", line 763, in _batch
    dequeued = queue.dequeue_many(batch_size, name=name)
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/ops/data_flow_ops.py", line 483, in dequeue_many
    self._queue_ref, n=n, component_types=self._dtypes, name=name)
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/ops/gen_data_flow_ops.py", line 3476, in queue_dequeue_many_v2
    component_types=component_types, timeout_ms=timeout_ms, name=name)
  File "/root/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 519, in _apply_op_helper
    repr(values), type(values).__name__))
TypeError: Expected int32 passed to parameter 'n' of op 'QueueDequeueManyV2', got 4.0 of type 'float' instead.

```



解决方法：增加int转换

```
  clone_batch_size = int(FLAGS.train_batch_size / config.num_clones)
```





参考：https://github.com/tensorflow/models/issues/3661