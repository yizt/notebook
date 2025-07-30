import json
import threading
from abc import ABC, abstractmethod

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

try:
    import zmq
except ImportError:
    zmq = None

class PubSubClient(ABC):
    """消息发布订阅客户端抽象基类"""
    
    @abstractmethod
    def connect(self):
        """连接服务端"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    def publish(self, topic, message):
        """发布消息"""
        pass
    
    @abstractmethod
    def subscribe(self, topic, callback):
        """订阅主题"""
        pass

class MQTTClient(PubSubClient):
    """MQTT客户端实现"""
    
    def __init__(self, config):
        if mqtt is None:
            raise RuntimeError("paho-mqtt library not installed")
            
        self.config = config
        self.client = mqtt.Client()
        self.callbacks = {}
        
        # 设置回调
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}")

    def _on_message(self, client, userdata, msg):
        callback = self.callbacks.get(msg.topic)
        if callback:
            try:
                payload = json.loads(msg.payload.decode())
            except json.JSONDecodeError:
                payload = msg.payload.decode()
            callback(msg.topic, payload)

    def connect(self):
        self.client.connect(
            self.config.get('host', 'localhost'),
            self.config.get('port', 1883)
        )
        self.client.loop_start()

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, topic, message):
        if isinstance(message, dict):
            message = json.dumps(message)
        self.client.publish(topic, message)

    def subscribe(self, topic, callback):
        self.client.subscribe(topic)
        self.callbacks[topic] = callback

class ZMQClient(PubSubClient):
    """ZeroMQ客户端实现"""
    
    def __init__(self, config):
        if zmq is None:
            raise RuntimeError("pyzmq library not installed")
            
        self.config = config
        self.context = zmq.Context()
        self.socket = None
        self.running = False
        self.callbacks = {}
        self.thread = None
        
    def connect(self):
        mode = self.config.get('mode', 'sub')  # pub/sub
        if mode == 'pub':
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.config.get('port', 5555)}")
        else:
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect(self.config.get('endpoint', 'tcp://localhost:5555'))
            self.socket.setsockopt_string(zmq.SUBSCRIBE, '')
            
        self.running = True
        self.thread = threading.Thread(target=self._message_loop)
        self.thread.start()

    def _message_loop(self):
        while self.running:
            try:
                topic = self.socket.recv_string()
                message = self.socket.recv_string()
                callback = self.callbacks.get(topic)
                if callback:
                    try:
                        msg = json.loads(message)
                    except json.JSONDecodeError:
                        msg = message
                    callback(topic, msg)
            except zmq.ZMQError:
                break

    def disconnect(self):
        self.running = False
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join()

    def publish(self, topic, message):
        if isinstance(message, dict):
            message = json.dumps(message)
        self.socket.send_string(topic, zmq.SNDMORE)
        self.socket.send_string(message)

    def subscribe(self, topic, callback):
        self.callbacks[topic] = callback

class PubSubAdapter:
    """统一消息适配器"""
    
    def __init__(self, protocol='mqtt', config=None):
        self.protocol = protocol
        self.config = config or {}
        self.client = self._create_client()
        
    def _create_client(self):
        if self.protocol == 'mqtt':
            return MQTTClient(self.config)
        elif self.protocol == 'zmq':
            return ZMQClient(self.config)
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol}")
    
    def connect(self):
        self.client.connect()
    
    def disconnect(self):
        self.client.disconnect()
    
    def publish(self, topic, message):
        self.client.publish(topic, message)
    
    def subscribe(self, topic, callback):
        self.client.subscribe(topic, callback)

# MQTT 配置
mqtt_config = {
    'host': 'broker.hivemq.com',
    'port': 1883
}

# ZMQ 配置
zmq_config = {
    'mode': 'sub',  # 或 'pub'
    'port': 5555,
    'endpoint': 'tcp://localhost:5555'
}

# 创建客户端（切换协议只需修改protocol参数）
client = PubSubAdapter(
    protocol='mqtt',  # 或 'zmq'
    config=mqtt_config
)

# 连接服务端
client.connect()

# 订阅消息
def on_message(topic, msg):
    print(f"Received [{topic}]: {msg}")

client.subscribe('sensors/temperature', on_message)

# 发布消息
client.publish('sensors/temperature', {'value': 25.6})

# 断开连接
client.disconnect()