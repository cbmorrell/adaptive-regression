import libemg
from PIL import Image
import numpy as np
import os
from pathlib import Path

def make_collection_ABADWFWE_gif():
    gestures = ["Radial_Deviation", "Ulnar_Deviation", "Wrist_Flexion", "Wrist_Extension"]
    pictures = {}
    for g in gestures:
        pictures[g] = Image.open(f"images/{g}.png")
    animator = libemg.animator.ScatterPlotAnimator(output_filepath=f"images/collection.mp4",
                                                        axis_images={"N":pictures["Radial_Deviation"],
                                                                    "E":pictures["Wrist_Extension"],
                                                                    "S":pictures["Ulnar_Deviation"],
                                                                    "W":pictures["Wrist_Flexion"]},
                                                    show_direction=True)
    coordinates = get_coordinates()
    animator.save_plot_video(coordinates, title='Regression Training', save_coordinates=True, verbose=True)

def make_collection_HOHCWPWS_gif():
    gestures = ["Hand_Open", "Hand_Close", "Pronation","Supination"]
    pictures = {}
    for g in gestures:
        pictures[g] = Image.open(f"images/{g}.png")
    animator = libemg.animator.ScatterPlotAnimator(output_filepath=f"images/collection.mp4",
                                                        axis_images={"N":pictures["Supination"],
                                                                    "E":pictures["Hand_Open"],
                                                                    "S":pictures["Pronation"],
                                                                    "W":pictures["Hand_Close"]},
                                                    show_direction=True)
    coordinates = get_coordinates()
    animator.save_plot_video(coordinates, title='Regression Training', save_coordinates=True, verbose=True)
    
def get_coordinates(period=2, cycles=10, rest_time=5, FPS=24):
    import math
    coordinates = []
    duration = int(cycles*period + rest_time)
    t = np.linspace(0, duration-rest_time, FPS*(duration-rest_time))
    cycle = np.sin(2*math.pi*(1/period)*t)
    coordinates.append(cycle)
    coordinates.append(np.zeros(FPS*rest_time))

    coordinates = np.concatenate(coordinates)
    dof1 = np.vstack((coordinates, np.zeros_like(coordinates))).T
    dof2 = np.vstack((np.zeros_like(coordinates), coordinates)).T
    final_coordinates = np.vstack((dof1, dof2)) 
    return final_coordinates

if __name__ == "__main__":
    # make_collection_ABADWFWE_gif()
    make_collection_HOHCWPWS_gif()