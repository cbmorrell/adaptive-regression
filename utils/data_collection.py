from pathlib import Path
import time

import libemg
import numpy as np


def collect_data(online_data_handler, media_folder, data_folder, num_reps):
    try:
        # Ensure data directory exists
        Path(data_folder).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Directory already exists
        if any(Path(data_folder).iterdir()):
            # Folder is not empty
            result = input(f'Data folder {data_folder} is not empty. Do you want to overwrite (y/n)?')
            if result != 'y':
                print(f'Skipping {data_folder}.')
                return
    
    # Create GUI
    args = {
        "media_folder"         : media_folder,
        "data_folder"          : data_folder,
        "num_reps"             : num_reps,
        "rest_time"            : 1,
        "auto_advance"         : False,
    }
    gui = libemg.gui.GUI(online_data_handler, args=args, debug=False, gesture_width=500, gesture_height=500)
    gui.start_gui()


def append_to_file(filename, data):
    with open(filename, 'a') as file:
        file.write(data)



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
            streamer = lambda: libemg.streamers.sifi_bioarmband_streamer(name="BioPoint_v1_1",ecg=False, eda=False, imu=False, ppg=False,
                                                                         bridge_version="1.1.3")
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


def cleanup_hardware(p):
    print("Performing clean-up...")
    p.signal.set()
    time.sleep(3)
    print("Clean-up finished.")


def get_frame_coordinates(movement_type = 'within', period=2, cycles=10, rest_time=5, FPS=24):
    import math
    coordinates = []
    duration = int(cycles*period + rest_time)
    t = np.linspace(0, duration-rest_time, FPS*(duration-rest_time))
    cycle = np.sin(2*math.pi*(1/period)*t)
    coordinates.append(cycle)
    coordinates.append(np.zeros(FPS*rest_time))

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
