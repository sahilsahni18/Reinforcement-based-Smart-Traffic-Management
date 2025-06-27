import time
import cv2
import numpy as np
from data_collection import capture_frame
from object_detection import detect_objects
from object_tracker import VehicleTracker
from sensors.mqtt_client import start_listener, sensor_queue
import torch
from rl_agent.networks import QNetwork # type: ignore

# Initialize
tracker = VehicleTracker()
start_listener()

# Load DQN
state_dim = 4
action_dim = 2
model = QNetwork(state_dim, action_dim)
model.load_state_dict(torch.load('rl_agent/dqn_model.pth'))
model.eval()

# Signal control stub
def actuate_signal(action):
    # send to GPIO or MQTT
    print(f"Setting signal phase {action}")

# Metrics logging
time_log = []

while True:
    frame = capture_frame()            # from data_collection.py
    dets = detect_objects(frame)       # returns list of [x1,y1,x2,y2,score,class]
    rects = [[x1,y1,x2,y2,score] for x1,y1,x2,y2,score,_ in dets]
    tracks = tracker.update(np.array(rects))

    # Build state (example: counts per lane)
    state = np.zeros(state_dim)
    # TODO: fill state from tracks + sensor_queue
    if not sensor_queue.empty():
        sensor = sensor_queue.get()
        state[0] = sensor['count']

    # Select action
    state_t = torch.FloatTensor(state).unsqueeze(0)
    action = model(state_t).argmax().item()
    actuate_signal(action)

    # Log metric
    time_log.append((time.time(), len(tracks)))

    time.sleep(1)  # loop delay