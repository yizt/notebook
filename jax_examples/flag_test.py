#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2024-06-07 13:49:04
@ LastEditors: yizuotian
@ LastEditTime: 2024-06-07 13:49:05
@ FilePath: /octo/examples/flag_test.py
@ Description: Do edit!
"""

from absl import flags, app
import ml_collections
from ml_collections import config_flags

config_dict = ml_collections.ConfigDict({
    'field1': 1,
    'field2': 'tom',
    'nested': {
        'field': 2.23,
    }
})

FLAGS = flags.FLAGS
config_flags.DEFINE_config_dict('config_dict', config_dict)
config_flags.DEFINE_config_file('config_file', './config.py')

def main(_):
    print(FLAGS.config_dict)
    print(FLAGS.config_file)

if __name__ == "__main__":
    """
    python flag_test.py --config_file=config.py --config_file.field3=1
    python flag_test.py --config_file=config.py
    python flag_test.py
    """
    app.run(main)