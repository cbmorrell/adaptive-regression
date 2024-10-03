import libemg
from utils import setup_live_processing
from config import Config
from models import Memory#, MLP
from PIL import Image
import numpy as np
import os
from pathlib import Path
config = Config()

class ScreenGuidedTraining:
    def __init__(self):
        pass

    def run(self):
        p, odh, smi = setup_live_processing()
        self.p = p
        args = {
            "online_data_handler": odh,
            "streamer":p,
            "media_folder": config.DC_image_location,
            "data_folder":  f"{config.DC_data_location}trial_{config.model}/",
            "num_reps":     config.DC_reps,
            "rep_time":     config.DC_rep_time,
            "rest_time":    config.DC_rest_time,
            "auto_advance": True
        }
        gui = libemg.gui.GUI(
            online_data_handler=args["online_data_handler"],
            args=args, debug=False)

        if not os.path.exists(f"{config.DC_image_location}collection.mp4"):
            make_collection_gif(gui)
        
        
        gui.start_gui()
    
    def get_hardware_process(self):
        return self.p
    

def load_sgt_data():
    # parse offline data into an offline data handler
    dataset_folder = f'./'
    package_function = lambda x, y: True
    metadata_fetchers = [libemg.data_handler.FilePackager(libemg.data_handler.RegexFilter("images/", ".txt", ["collection"], "labels"), package_function)]
        
    offdh = libemg.data_handler.OfflineDataHandler()
    offdh.get_data(dataset_folder, 
                   [libemg.data_handler.RegexFilter("_R_","_emg.csv",["0","1","2"], "reps"),
                    libemg.data_handler.RegexFilter("subject", "/",[str(config.subjectID)], "subjects")],
                   metadata_fetchers,
                    ",")
    return offdh

import torch
def offdh_to_memory():
    offdh = load_sgt_data()
    train_windows, train_metadata = offdh.parse_windows(config.window_length, config.window_increment,metadata_operations={"labels": "last_sample"})
    fe = libemg.feature_extractor.FeatureExtractor()
    features = fe.extract_features(config.features, train_windows, config.feature_dictionary)
    features = torch.hstack([torch.tensor(features[key], dtype=torch.float32) for key in features.keys()])
    targets = torch.tensor(torch.eye(5)[train_metadata["classes"],:], dtype=torch.float32)

    memory = Memory()
    memory.add_memories(experience_data = features,
                        experience_targets = targets)
    memory.memories_stored = features.shape[0]
    memory.shuffle()
    return memory


def make_collection_gif(gui):
    gui.download_gestures([2,3,6,7], config.DC_image_location)
    pictures = {}
    gestures = ["Hand_Close", "Hand_Open", "Pronation", "Supination"]
    for g in gestures:
        pictures[g] = Image.open(f"{config.DC_image_location}{g}.png")
    animator = libemg.animator.ScatterPlotAnimator(output_filepath=f"{config.DC_image_location}collection.mp4",
                                                        axis_images={"N":pictures["Supination"],
                                                                    "E":pictures["Hand_Open"],
                                                                    "S":pictures["Pronation"],
                                                                    "W":pictures["Hand_Close"]})
    coordinates = get_coordinates()

    animator.save_plot_video(coordinates, title='Regression Training', save_coordinates=True, verbose=True)
    
    
def get_coordinates(steady_state_time=1, ramp_time=1, cycles=3, FPS=24):
    coordinates = []
    for c in range(cycles):
        # A cycle is like:

        coordinates.append(np.zeros(steady_state_time*FPS)) # steady state (0)
        
        coordinates.append(np.linspace(0, 1,FPS*ramp_time))  # ramp up (+1)
        coordinates.append(np.ones(FPS*steady_state_time))   # steady state (+1)
        coordinates.append(np.linspace(1, 0, FPS*ramp_time)) # ramp down (+1)
        
        coordinates.append(np.linspace(0, -1, ramp_time*FPS))# ramp up (-1)
        coordinates.append(-1*np.ones(FPS*steady_state_time))# steady state (-1)
        coordinates.append(np.linspace(-1, 0, FPS*ramp_time)) # ramp down (-1)

        coordinates.append(np.zeros(steady_state_time*FPS)) # steady state (0)

        coordinates.append(np.linspace(0, -1, ramp_time*FPS))# ramp up (-1)
        coordinates.append(-1*np.ones(FPS*steady_state_time))# steady state (-1)
        coordinates.append(np.linspace(-1, 0, FPS*ramp_time)) # ramp down (-1)

        coordinates.append(np.linspace(0, 1,FPS*ramp_time))  # ramp up (+1)
        coordinates.append(np.ones(FPS*steady_state_time))   # steady state (+1)
        coordinates.append(np.linspace(1, 0, FPS*ramp_time)) # ramp down (+1)

    coordinates = np.concatenate(coordinates)
    dof1 = np.vstack((coordinates, np.zeros_like(coordinates))).T
    dof2 = np.vstack((np.zeros_like(coordinates), coordinates)).T
    final_coordinates = np.vstack((dof1, dof2)) 
    return final_coordinates


import pickle
def prepare_model_from_sgt():
    # prepare inner model
    sgt_memory = offdh_to_memory()
    mdl = MLP(input_shape=config.input_shape)
    mdl.fit(config.DC_epochs, shuffle_every_epoch=True, memory=sgt_memory)
    # prepare pc thresholds
    th_min_dic, th_max_dic = mdl.get_pc_thresholds(sgt_memory)
    # install mdl and thresholds to EMGClassifier
    offline_classifier = libemg.emg_predictor.EMGRegressor()
    offline_classifier.__setattr__("classifier", mdl)
    offline_classifier.__setattr__("feature_params", config.feature_dictionary)
    # save EMGClassifier to file
    with open(f"{config.DC_data_location}trial_{config.model}/sgt_mdl.pkl", 'wb') as handle:
        pickle.dump(offline_classifier, handle)

# def prepare_blank_model():
#     # for zero-shot initialization
#     mdl = MLP(input_shape=config.input_shape)
#     th_min_dic = {i:config.WENG_SPEED_MIN for i in range(5)}
#     th_max_dic = {i:config.WENG_SPEED_MAX for i in range(5)}
#     # install mdl and thresholds to EMGClassifier
#     offline_classifier = libemg.emg_classifier.EMGClassifier()
#     offline_classifier.__setattr__("classifier", mdl)
#     offline_classifier.__setattr__("velocity", True)
#     offline_classifier.__setattr__("th_min_dic", th_min_dic)
#     offline_classifier.__setattr__("th_max_dic", th_max_dic)
#     offline_classifier.__setattr__("velocity_metric_handle", velocity_metric_handle)
#     offline_classifier.__setattr__("velocity_mapping_handle", velocity_mapping_handle)
#     offline_classifier.__setattr__("feature_params", config.feature_dictionary)
#     # save EMGClassifier to file
#     with open(f"{config.DC_data_location}trial_{config.model}/zs_mdl.pkl", 'wb') as handle:
#         pickle.dump(offline_classifier, handle)

# import matplotlib.pyplot as plt
# def visualize_sgt():
#     offdh = load_sgt_data()
#     windows, metadata = offdh.parse_windows(config.window_length, config.window_increment)
#     fe = libemg.feature_extractor.FeatureExtractor()
#     features = fe.extract_features(config.features, windows, config.feature_dictionary)
#     fe.visualize_feature_space(features, "PCA",metadata["classes"], render=False)
#     plt.figure(1)
#     plt.savefig(f"{config.DC_data_location}trial_{config.model}/sgt_feature_space.png")
#     plt.close('all')