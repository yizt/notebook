#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2024-06-07 13:49:58
@ LastEditors: yizuotian
@ LastEditTime: 2024-06-07 13:49:59
@ FilePath: /octo/examples/config.py
@ Description: Do edit!
"""

import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.field3 = 3
    config.field4 = 'jerry'
    config.nested2 = ml_collections.ConfigDict()
    config.nested2.field5 = 2.33
    
    return config