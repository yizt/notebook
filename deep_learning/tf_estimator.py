# -*- coding: utf-8 -*-
"""
 @File    : tf_estimator.py
 @Time    : 2019-10-12 14:01
 @Author  : yizuotian
 @Description    : tf.est
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
import numpy as np
from absl import flags
import os
import cv2
import datetime
import codecs
import tensorflow as tf

tf.enable_eager_execution()

from object_detection import model_hparams
from object_detection import model_lib


def input_generator(image_paths):
    """Yield encoded strings from sorted_inputs."""
    for i, image_path in enumerate(image_paths):
        # img_np=load_image_into_numpy_array(image_path)
        img_np = np.array(Image.open(image_path)).astype(np.float32)
        yield {"image": img_np[np.newaxis, ...], "true_image_shape": np.array(img_np.shape, np.int32)[np.newaxis, ...]}
        # yield img_np,np.array(img_np.shape,np.uint8)

def video_generator(video_path, batch_size=8):
    """
    读取视频
    :param video_path:
    :param batch_size:
    :return:
    """
    cap = cv2.VideoCapture(video_path.decode())
    # 初始化batch_image_np
    counter = []  # 计数
    batch_image_np = np.zeros([batch_size, 300, 300, 3], np.float32)
    # 图像的shape
    image_shape = np.array([300, 300, 3], np.int32)
    batch_image_shape = np.stack([image_shape] * batch_size, axis=0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame = cv2.resize(frame, (300, 300))  # 转为300*300
        batch_image_np[len(counter)] = frame.astype(np.float32)[:, :, ::-1]  # BGR to RGB
        counter.append(1)
        if len(counter) == batch_size:
            counter = []  # 计数清零
            yield {'image': batch_image_np, 'true_image_shape': batch_image_shape}
    # 处理最后一个batch
    if len(counter) > 0:
        yield {'image': batch_image_np[:len(counter)], 'true_image_shape': batch_image_shape[:len(counter)]}
    cap.release()


def image_generator(image_dir, batch_size=8):
    """
    读取目录下所有图像
    :param image_dir:
    :param batch_size:
    :return: dict{'image': batch_image_np, 'true_image_shape': batch_image_shape}
    """
    image_names = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, image_name) for image_name in image_names]
    # 初始化batch_image_np
    counter = []  # 计数
    batch_image_np = np.zeros([batch_size, 300, 300, 3], np.float32)
    # 图像的shape
    image_shape = np.array([300, 300, 3], np.int32)
    batch_image_shape = np.stack([image_shape] * batch_size, axis=0)
    for filename, file_path in zip(image_names, image_paths):
        image_np = np.array(Image.open(file_path))
        image_np = cv2.resize(image_np, (300, 300)).astype(np.float32)
        batch_image_np[len(counter)] = image_np
        counter.append(1)
        if len(counter) == batch_size:
            # batch_image_np = np.stack(image_np_list, axis=0)
            counter = []  # 计数清零
            yield {'image': batch_image_np, 'true_image_shape': batch_image_shape}
    # 处理最后一个batch
    if len(counter) > 0:
        print(len(counter))
        yield {'image': batch_image_np[:len(counter)], 'true_image_shape': batch_image_shape[:len(counter)]}


def test():
    ckpt_dir = '/sdb/tmp/users/yizt/data/20191005.80000.150000.5/output_model'
    pipe_line_config = '/sdb/tmp/users/yizt/data/20191005.80000.150000.5/pipeline.config'
    config = tf.estimator.RunConfig(model_dir=ckpt_dir)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(None),
        sample_1_of_n_eval_examples=1,
        pipeline_config_path=pipe_line_config)
    estimator = train_and_eval_dict['estimator']

    image_dir = '/sdb/tmp/truck_crane/part1'
    image_path_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

    def input_fn():
        """Created batched dataset of encoded inputs."""
        ds = tf.data.Dataset.from_generator(
            input_generator, {"image": tf.float32, "true_image_shape": tf.int32},
            output_shapes={"image": tf.TensorShape([None, 480, 848, 3]), "true_image_shape": tf.TensorShape([None, 3])},
            args=[image_path_list])
        return ds

    rs = estimator.predict(input_fn, yield_single_examples=False)

    import datetime
    for i, x in enumerate(rs):
        if i % 100 == 0:
            print("============={}==============={:06d}=============".format(datetime.datetime.now(), i))
            print(x)


def main(unused_argv):
    flags.mark_flag_as_required('ckpt_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    # 场景对应的类别id
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    config = tf.estimator.RunConfig(model_dir=FLAGS.ckpt_dir)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(None),
        sample_1_of_n_eval_examples=1,
        pipeline_config_path=FLAGS.pipeline_config_path)
    estimator = train_and_eval_dict['estimator']

    def input_fn():
        if FLAGS.input_type == 'video':
            ds = tf.data.Dataset.from_generator(
                video_generator, {"image": tf.float32, "true_image_shape": tf.int32},
                output_shapes={"image": tf.TensorShape([None, 300, 300, 3]),
                               "true_image_shape": tf.TensorShape([None, 3])},
                args=[FLAGS.video_path, FLAGS.batch_size])
        else:
            ds = tf.data.Dataset.from_generator(
                image_generator, {"image": tf.float32, "true_image_shape": tf.int32},
                output_shapes={"image": tf.TensorShape([None, 300, 300, 3]),
                               "true_image_shape": tf.TensorShape([None, 3])},
                args=[FLAGS.image_dir, FLAGS.batch_size])
        return ds

    rs = estimator.predict(input_fn)
    start = datetime.datetime.now()
    # 写预测结果到文件中
    with codecs.open(FLAGS.txt_output, mode='w', encoding='utf-8') as w:
        for i, r in enumerate(rs):
            boxes = r['detection_boxes']
            scores = r['detection_scores']
            classes = r['detection_classes'].astype(np.uint8)
            # num_detections = r['num_detections']
            w.write("scores:{};classes:{}\n".format(','.join(scores.astype(np.str)),
                                                    ','.join(classes.astype(np.str))))
            if i % 500 == 0:
                print("============={}==============={:06d}=============".format(datetime.datetime.now(), i))
    print(datetime.datetime.now() - start)
