#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2025-02-13 14:54:24
@ LastEditors: yizuotian
@ LastEditTime: 2025-02-13 14:54:25
@ FilePath: /mqtt_examples/mqtt_request.py
@ Description: Do edit!
"""
import paho.mqtt.client as mqtt
import uuid

# 生成唯一客户端 ID
CLIENT_ID = str(uuid.uuid4())

def on_connect(client, userdata, flags, rc):
    print(f"Request Client Connected with code {rc}")
    # 订阅响应主题
    client.subscribe(f"response/topic/{CLIENT_ID}")

def on_message(client, userdata, msg):
    print(f"Received Response: {msg.payload.decode()} on {msg.topic}")

def on_disconnect(client, userdata, rc):
    print(f"Request Client Disconnected with code {rc}")



def test():
    """
    @ description: 
    @ param {type} 
    @ return: 
    """
    # 创建客户端
    client = mqtt.Client(client_id=CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    # 连接 Broker
    client.connect("test.mosquitto.org", 1883)

    # 发送请求
    request_topic = "request/topic"
    request_payload = "What is the current time?"
    client.publish(request_topic, payload=request_payload, qos=1)
    print(f"Sent Request: {request_payload} to {request_topic}")

    # 保持网络循环
    #client.loop_forever()
    client.disconnect()

if __name__ == "__main__":
	test()
