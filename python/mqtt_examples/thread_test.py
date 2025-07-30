#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2025-02-27 10:27:50
@ LastEditors: yizuotian
@ LastEditTime: 2025-02-27 10:27:50
@ FilePath: /mqtt_examples/thread_test.py
@ Description: Do edit!
"""

import threading
import time

# 创建一个事件对象
stop_event = threading.Event()

def worker():
    """工作线程函数"""
    print("Worker: 启动工作线程...")
    while not stop_event.is_set():  # 检查事件是否被设置
        print("Worker: 正在工作...")
        time.sleep(1)
    print("Worker: 收到停止信号，退出工作线程。")

def main():
    """主程序"""
    # 创建并启动工作线程
    thread = threading.Thread(target=worker)
    thread.start()

    # 主线程等待一段时间后发送停止信号
    time.sleep(5)
    print("Main: 发送停止信号...")
    stop_event.set()  # 设置事件状态为“已设置”

    # 等待工作线程退出
    thread.join()
    print("Main: 工作线程已退出，程序结束。")

if __name__ == "__main__":
    main()
