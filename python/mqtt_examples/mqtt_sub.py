#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2025-02-13 14:34:19
@ LastEditors: yizuotian
@ LastEditTime: 2025-02-13 14:34:20
@ FilePath: /mqtt_examples/mqtt_pub.py
@ Description: mqtt 发布者
"""
import paho.mqtt.client as mqtt

# 连接回调
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # 订阅所有以 "sensor/" 开头的主题
    client.subscribe("sensor/#")

# 消息接收回调
def on_message(client, userdata, msg):
    print(f"Received: {msg.payload.decode()} on topic {msg.topic},{userdata=}")



def test():
    """
    @ description: 
    @ param {type} 
    @ return: 
    """
    # 创建客户端
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    # 连接 MQTT Broker（以公共测试服务器为例）
    #client.connect("test.mosquitto.org", 1883, 60)
    client.connect("127.0.0.1", 1883, 60)

    # 保持网络循环
    client.loop_forever()

if __name__ == "__main__":
	test()
