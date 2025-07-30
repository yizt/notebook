#!/usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
import threading


from loguru import logger
import sys
import threading

# 向日志记录中注入当前线程名称
logger = logger.patch(lambda record: record.update(thread_name=threading.current_thread().name))
# 定义过滤器：仅允许主线程的日志
def filter_main_thread(record):
    print('')
    return record["thread_name"] != "data_forward"

# 添加控制台Handler，应用过滤器
logger.remove()
logger.add(sys.stdout, filter=filter_main_thread)

# 启动子线程
thread = threading.Thread(target=worker,name="data_forward")

# # 定义全局开关
# ENABLE_SUBTHREAD_LOGGING = False

# # 在日志记录前检查线程类型
# def disable_subthread_logging(record):
#     if not ENABLE_SUBTHREAD_LOGGING:
#         return threading.current_thread() is threading.main_thread()
#     return True

# 添加控制台Handler，应用过滤器
logger.remove()
logger.add(sys.stdout, filter=filter_main_thread)

# 测试主线程日志

# 子线程函数
def worker():
    logger.info("This log is from a sub-thread and will be filtered out.")



if __name__=='__main__':
    logger.info("This log is from the main thread.")

    # 启动子线程
    thread = threading.Thread(target=worker,name="data_forward")
    thread.start()
    thread.join()
