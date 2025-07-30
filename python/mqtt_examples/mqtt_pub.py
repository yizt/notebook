#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2025-02-13 14:38:44
@ LastEditors: yizuotian
@ LastEditTime: 2025-02-13 14:38:45
@ FilePath: /mqtt_examples/mqtt_pub.py
@ Description: Do edit!
"""
import paho.mqtt.client as mqtt
import time


def test():
    """
    @ description: 
    @ param {type} 
    @ return: 
    """
    client = mqtt.Client()

    # 连接 Broker
    #client.connect("test.mosquitto.org", 1883, 60)
    client.connect("127.0.0.1", 1883, 60)

    # 发布消息到不同主题
    for i in range(5):
        topic = f"sensor/temperature/{i}"
        payload = f"25.{i}°C"
        client.publish(topic, payload)
        print(f"Published: {payload} to {topic}")
        time.sleep(1)

    client.disconnect()

if __name__ == "__main__":
	test()
