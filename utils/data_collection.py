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

def make_collection_videos():
    libemg.gui.GUI(None).download_gestures([2, 3, 6, 7], 'images')
    gestures = ["Hand_Open", "Hand_Close", "Pronation", "Supination"]
    pictures = {}
    for g in gestures:
        pictures[g] = Image.open(f"images/{g}.png")
    animator = libemg.animator.ScatterPlotAnimator(output_filepath=Path('images', 'within.mp4').absolute().as_posix(),
                                                    axis_images={"N":pictures["Supination"],
                                                                "E":pictures["Hand_Open"],
                                                                "S":pictures["Pronation"],
                                                                "W":pictures["Hand_Close"]},
                                                    show_direction=True,
                                                    show_countdown=True)
    coordinates = get_frame_coordinates(movement_type='within')
    animator.save_plot_video(coordinates, title='Regression Training', save_coordinates=True, verbose=True)

    animator = libemg.animator.ScatterPlotAnimator(output_filepath=Path('images', 'combined.mp4').absolute().as_posix(),
                                                    axis_images={"N":pictures["Supination"],
                                                                "E":pictures["Hand_Open"],
                                                                "S":pictures["Pronation"],
                                                                "W":pictures["Hand_Close"]},
                                                    show_direction=True,
                                                    show_countdown=True)
    coordinates = get_frame_coordinates(movement_type='within')
    animator.save_plot_video(coordinates, title='Regression Training', save_coordinates=True, verbose=True)
