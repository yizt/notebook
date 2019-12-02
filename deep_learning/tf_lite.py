# -*- coding: utf-8 -*-
"""
 @File    : tf_lite.py
 @Time    : 2019/12/2 下午5:24
 @Author  : yizuotian
 @Description    :
"""

from tensorflow import lite


def convert_keras(keras_file, output_tflite):
    custom_objects = {"ProposalLayer": ProposalLayer}
    converter = lite.TFLiteConverter.from_keras_model_file(keras_file,
                                                           custom_objects=custom_objects)
    tflite_model = converter.convert()
    open(output_tflite, "wb").write(tflite_model)


if __name__ == '__main__':
    file_path = '/Users/yizuotian/pyspace/keras-ssd/ssd-mobile.h5'
    tflite_path = '/Users/yizuotian/pyspace/keras-ssd/ssd-mobile.tflite'
    convert_keras(file_path, tflite_path)
