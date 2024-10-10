from pathlib import Path
import shutil
import time
import os

import libemg
import cv2 as cv
import numpy as np
from PIL import Image


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
    
    # Calculate rep time from video (maybe add this to libemg?)
    media_file = Path(media_folder, 'collection.mp4').absolute()
    assert media_file.exists(), f"Media file {media_file} does not exist."
    video = cv.VideoCapture(media_file.as_posix())
    fps = 24
    rep_time = video.get(cv.CAP_PROP_FRAME_COUNT) / fps
    
    # Copy labels file to data directory
    labels_file = Path(media_folder, 'collection.txt').absolute().as_posix()
    shutil.copy(labels_file, Path(data_folder, 'labels.txt').absolute().as_posix())
    
    # Create GUI
    args = {
        "media_folder"         : media_folder,
        "data_folder"          : data_folder,
        "num_reps"             : num_reps,
        "rep_time"             : rep_time,
        "rest_time"            : 1,
        "auto_advance"         : True,
        "visualization_horizon": 5000,
        "visualization_rate"   : 24,
    }
    gui = libemg.gui.GUI(online_data_handler, args=args, debug=False, gesture_width=500, gesture_height=500)
    gui.start_gui()
    # TODO: Maybe add check to make sure that data is still being read (only returning the same value over across all channels)


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
            streamer = lambda: libemg.streamers.sifi_bioarmband_streamer(name="BioPoint_v1_1",ecg=False, eda=False, imu=False, ppg=False)
            num_channels = 8
        else:
            raise ValueError(f"Unexpected value for device. Got: {device}.")

        self.name = name
        self.fs = fs
        self.streamer = streamer
        self.num_channels = num_channels

    def stream(self):
        p, smi = self.streamer()
        return smi




def cleanup_hardware(p):
    print("Performing clean-up...")
    p.signal.set()
    time.sleep(3)
    print("Clean-up finished.")
