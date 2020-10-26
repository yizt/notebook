# -*- coding: utf-8 -*-
"""
 @File    : ctc_tf2_keras.py
 @Time    : 2020/10/23 上午7:53
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np
from tensorflow import keras


class TextImage(object):
    def __init__(self, alpha, im_h, im_w, max_chars, min_chars):
        """

        :param alpha:
        :param im_h:
        :param im_w:
        :param max_chars:
        :param min_chars:
        """
        self.alpha = alpha
        self.im_h = im_h
        self.im_w = im_w
        self.max_chars = max_chars
        self.min_chars = min_chars

    def gen(self):
        char_len = np.random.randint(self.min_chars, self.max_chars + 1)
        chars = np.random.choice(list(self.alpha), char_len)
        text = ''.join(chars)
        im = np.ones((self.im_h, self.im_w), dtype=np.uint8) * 255
        cv2.putText(im, text=text,
                    org=(np.random.randint(self.im_w // len(text)),
                         int(self.im_h * 0.7 + np.random.rand() * 3)),
                    fontFace=np.random.choice([1, 5, 6, 7]),
                    fontScale=1. + 0.1 * np.random.randn(),
                    color=np.random.randint(150), thickness=1)
        return im, text


def norm_im(im_gray):
    img = im_gray[:, :, np.newaxis]  # [H,W,C]
    # 标准化
    image = img.astype(np.float32) / 255.
    image -= np.mean(image)
    return image


class Generator(keras.utils.Sequence):
    def __init__(self, text_image: TextImage, batch_size=32):
        self.text_image = text_image
        self.im_h = text_image.im_h
        self.im_w = text_image.im_w
        self.alpha = text_image.alpha
        self.max_len = text_image.max_chars
        self.input_len = int(text_image.im_w / 4 - 3)
        self.batch_size = batch_size

    def __getitem__(self, item):
        image_np = np.zeros((self.batch_size, self.im_h, self.im_w, 1))
        indices_np = np.ones((self.batch_size, self.max_len)) * -1  # 默认-1

        input_len_np = np.ones((self.batch_size, 1)) * self.input_len
        target_len_np = np.zeros((self.batch_size, 1))
        for i in range(self.batch_size):
            im_gray, text = self.text_image.gen()
            image = norm_im(im_gray)
            indices = [self.text_image.alpha.index(char) for char in text]
            target_len = len(text)

            # 赋值到batch中
            image_np[i] = image
            indices_np[i, :target_len] = indices
            target_len_np[i, 0] = target_len

        return image_np, np.concatenate([indices_np, input_len_np, target_len_np], axis=-1)

    def __len__(self):
        return 2000 // self.batch_size


def conv_block(inputs, out_channels, kernel_size=3, stride=1, padding='same', bn=False):
    x = keras.layers.Conv2D(out_channels, kernel_size, stride, padding)(inputs)
    if bn:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    return x


def make_small_crnn(h, w, num_classes, num_base_filters=16):
    inputs = keras.layers.Input(shape=(h, w, 1))

    x = inputs
    # conv block 1
    x = conv_block(x, num_base_filters)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    # conv block 2
    x = conv_block(x, num_base_filters * 2)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # conv block 3
    x = conv_block(x, num_base_filters * 4)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 1))(x)

    # conv block 4
    x = conv_block(x, num_base_filters * 8, bn=True)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 1))(x)

    # conv block 5
    x = conv_block(x, num_base_filters * 8, kernel_size=2, bn=True, padding='valid')

    # [B,1,W,C]=>[B,W,C]
    x = keras.layers.Reshape(target_shape=(-1, num_base_filters * 8))(x)

    x = keras.layers.Dense(num_classes)(x)
    x = keras.layers.Activation('softmax')(x)

    return keras.models.Model(inputs=inputs, outputs=x)


def custom_ctc():
    def loss(y_true, y_pred):
        batch_labels = y_true[:, :-2]
        input_length = y_true[:, -2:-1]
        label_length = y_true[:, -1:]

        return keras.backend.ctc_batch_cost(batch_labels, y_pred, input_length, label_length)

    return loss


def train():
    text_image = TextImage('abc', 32, 96, 5, 2)
    model = make_small_crnn(32, 96, len(text_image.alpha) + 1, 8)
    model.summary()
    optimizer = keras.optimizers.Adadelta(1e-3)
    model.compile(loss=custom_ctc(), optimizer=optimizer)

    generator = Generator(text_image, 32)
    model.fit_generator(generator, epochs=3)


if __name__ == '__main__':
    train()
