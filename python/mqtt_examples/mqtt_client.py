#!/usr/bin/env python
# -*- coding: utf-8 -*-
import paho.mqtt.client as mqtt
import time

class MQTTClient:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to Broker!")
            client.subscribe("command/#")
        else:
            print(f"Connection failed with code {rc}")

    def on_message(self, client, userdata, msg):
        print(f"[Command] {msg.topic}: {msg.payload.decode()}")

    def on_disconnect(self, client, userdata, rc):
        print(f"Disconnected with code {rc}")

    def connect(self):
        self.client.will_set("status/device", "offline", qos=1)
        self.client.connect("127.0.0.1", 1883)
        self.client.loop_start()

    def publish_data(self):
        while True:
            self.client.publish("sensor/data", "sample_data", qos=1)
            time.sleep(5)

if __name__ == "__main__":
    mqtt_client = MQTTClient()
    mqtt_client.connect()
    mqtt_client.publish_data()