import math

import libemg
import numpy as np


class Device:
    def __init__(self, device: str):
        if 'myo' in device:
            name = 'myo'
            fs = 200
            streamer = libemg.streamers.myo_streamer
            num_channels = 8
        elif 'emager' in device:
            name = 'emager'
            fs = 1010
            streamer = libemg.streamers.emager_streamer
            num_channels = 64
        elif 'oymotion' in device:
            # fs = 1000   # analyze_hardware says it isn't sampling fast enough (670 Hz)...
            name = 'oymotion'
            fs = 670
            streamer = libemg.streamers.oymotion_streamer
            num_channels = 8
        elif 'sifi' in device:
            name = 'sifi'
            fs = 1500
            streamer = libemg.streamers.sifi_bioarmband_streamer
            num_channels = 8
        else:
            raise ValueError(f"Unexpected value for device. Got: {device}.")

        self.name = name
        self.fs = fs
        self.streamer = streamer
        self.num_channels = num_channels

    def stream(self):
        p, smi = self.streamer()
        return p, smi


def get_frame_coordinates(movement_type = 'within', period=2, cycles=10, rest_time=5, fps=24):
    coordinates = []
    duration = int(cycles*period + rest_time)
    t = np.linspace(0, duration-rest_time, fps*(duration-rest_time))
    cycle = np.sin(2*math.pi*(1/period)*t)
    coordinates.append(cycle)
    coordinates.append(np.zeros(fps*rest_time))

    coordinates = np.concatenate(coordinates)
    if movement_type == 'within':
        dof1 = np.vstack((coordinates, np.zeros_like(coordinates))).T
        dof2 = np.vstack((np.zeros_like(coordinates), coordinates)).T
    elif movement_type == 'combined':
        dof1 = np.vstack((coordinates, coordinates)).T
        dof2 = np.vstack((coordinates, -coordinates)).T
    else:
        raise ValueError(f"Unexpected value for movement_type. Got: {movement_type}.")
    final_coordinates = np.vstack((dof1, dof2)) 
    return final_coordinates
