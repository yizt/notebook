#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2025-02-13 15:06:58
@ LastEditors: yizuotian
@ LastEditTime: 2025-02-13 15:06:59
@ FilePath: /mqtt_examples/mqtt_response.py
@ Description: Do edit!
"""
import paho.mqtt.client as mqtt
from datetime import datetime

def on_connect(client, userdata, flags, rc):
    print(f"Response Client Connected with code {rc}")
    # 订阅请求主题
    client.subscribe("request/topic")

def on_message(client, userdata, msg):
    print(f"Received Request: {msg.payload.decode()} from {msg.topic}")
    # 处理请求
    response_payload = f"Current time is {datetime.now()}"
    # 提取请求客户端的 ID
    request_client_id = msg.topic.split("/")[-1]
    # 发布响应到响应主题
    response_topic = f"response/topic/{request_client_id}"
    client.publish(response_topic, payload=response_payload, qos=1)
    print(f"Sent Response: {response_payload} to {response_topic}")

def on_disconnect(client, userdata, rc):
    print(f"Response Client Disconnected with code {rc}")


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
    client.on_disconnect = on_disconnect

    # 连接 Broker
    client.connect("test.mosquitto.org", 1883)

    # 保持网络循环
    client.loop_forever()

if __name__ == "__main__":
	test()
