#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2025-03-04 18:46:03
@ LastEditors: yizuotian
@ LastEditTime: 2025-03-04 18:46:04
@ FilePath: /mqtt_examples/thread_env_test.py
@ Description: Do edit!
"""
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import os
import threading
 
thread_local = threading.local()
 
def set_env(key, value):
    setattr(thread_local, key, value)
 
def get_env(key):
    return getattr(thread_local, key, None)
 
# 在线程中使用
def worker(env_var):
    # set_env('MY_VAR', 'some_value')
    value = get_env(env_var)
    logger.info(f"{threading.current_thread().name}:{value}")

def get_env(env_var):
    value = os.environ[env_var]
    logger.info(f"{threading.current_thread().name}:{value}")
 
def init_worker():
    # 设置特定环境变量
    os.environ['MY_ENV_VAR'] = 'default_value'
 
with ThreadPoolExecutor(5, initializer=init_worker) as executor:
    for _ in range(5):
        executor.submit(get_env,('PYTHONPATH'))
    
    get_env('PYTHONPATH')
