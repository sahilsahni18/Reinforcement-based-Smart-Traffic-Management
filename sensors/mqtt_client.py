import json
import queue
import threading
import paho.mqtt.client as mqtt # type: ignore

sensor_queue = queue.Queue()

BROKER = 'mqtt.example.com'
TOPIC = 'city/intersection/+/count'


def on_connect(client, userdata, flags, rc):
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    payload = json.loads(msg.payload)
    # e.g. {'intersection_id': 1, 'count': 12, 'timestamp': 123456789}
    sensor_queue.put(payload)


def start_listener():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, 1883, 60)
    thread = threading.Thread(target=client.loop_forever)
    thread.daemon = True
    thread.start()
