"""Real sensor diagnostic. Intentionally contains NO ROS publisher or motor API."""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import select
import struct
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Imu
from livelybot_serial.msg import MotorState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.geometry import quaternion_matrix_xyzw, yaw_matrix


def read_exact(stream, size, timeout=5.):
    deadline = time.monotonic()+timeout
    chunks = []
    remaining = size
    while remaining:
        wait = deadline-time.monotonic()
        if wait <= 0 or not select.select([stream], [], [], max(0., wait))[0]:
            raise TimeoutError('RGB-D stream timed out')
        data = os.read(stream.fileno(), remaining)
        if not data:
            raise EOFError('RGB-D stream closed')
        chunks.append(data)
        remaining -= len(data)
    return b''.join(chunks)


@dataclass
class Frame:
    time: float
    rgb: np.ndarray
    depth: np.ndarray


class Sensors:
    def __init__(self):
        self.lock = threading.Lock()
        self.imu = deque(maxlen=500)
        self.joints = {}
        self.head = {}
        self.subs = [rospy.Subscriber('/imu/data', Imu, self.on_imu, queue_size=20)]
        for side in ('l', 'r'):
            for name in ('hip_pitch','hip_roll','thigh','calf','ankle_pitch','ankle_roll',
                         'shoulder_pitch','shoulder_roll','upper_arm','elbow'):
                joint = side+'_'+name+'_joint'
                self.subs.append(rospy.Subscriber('/livelybot_real_real/'+joint+'_controller/state',
                    MotorState, self.on_joint, callback_args=joint, queue_size=1))
        for joint in ('head_pitch_joint','head_yaw_joint'):
            self.subs.append(rospy.Subscriber('/livelybot_real_real/'+joint+'_controller/state',
                MotorState,self.on_head,callback_args=joint,queue_size=1))

    def on_imu(self, msg):
        q = msg.orientation
        values = np.array([q.x,q.y,q.z,q.w])
        if msg.orientation_covariance[0] < 0 or not np.isfinite(values).all() or abs(np.linalg.norm(values)-1) > .05:
            return
        with self.lock:
            self.imu.append((time.monotonic(), values, float(msg.header.stamp.to_sec())))

    def on_joint(self, msg, joint):
        with self.lock:
            self.joints[joint] = (time.monotonic(), float(msg.pos), float(msg.vel))

    def on_head(self, msg, joint):
        with self.lock:
            self.head[joint] = (time.monotonic(),float(msg.pos),float(msg.vel))

    def snapshot(self, timestamp):
        with self.lock:
            nearest = min(self.imu, key=lambda v:abs(v[0]-timestamp)) if self.imu else None
            joints = dict(self.joints)
            head = dict(self.head)
        if nearest is None or abs(nearest[0]-timestamp) > .15:
            raise ValueError('No fresh IMU orientation near camera host receipt')
        return nearest, joints, head


def capture_worker(stream, width, height, latest):
    def offer(value):
        # Consumer may empty the queue between full() and get(); do not turn
        # that ordinary race into a camera failure.
        while True:
            try:
                latest.put_nowait(value)
                return
            except queue.Full:
                try:
                    latest.get_nowait()
                except queue.Empty:
                    pass
    try:
        while True:
            packet = read_exact(stream, 8+width*height*7, timeout=18.)
            timestamp = struct.unpack('<d', packet[:8])[0]
            n = width*height
            frame = Frame(timestamp,
                np.frombuffer(packet, np.uint8, n*3, 8).reshape(height,width,3).copy(),
                np.frombuffer(packet, '<f4', n, 8+n*3).reshape(height,width).copy())
            offer(frame)
    except Exception as exc:
        offer(exc)


def main():
    # The legacy name now enters the same shared pipeline as run_visual.py.
    from deploy.shared_navigation import main as shared_main
    return shared_main()


if __name__ == '__main__':
    raise SystemExit(main())
