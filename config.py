from pathlib import Path

import libemg
import numpy as np
import os
import pickle
import torch
from PIL import Image

from utils.models import Memory, MLP
from utils.data_collection import Device

class Config:
    def __init__(self, subject_id: str, model: int, stage: str):
        # usb towards the hand
        assert model in [1, 2], f"Unexpected value for model. Got: {model}."
        assert stage in ['sgt', 'fitts'], f"Unexpected value for stage. Got: {stage}."
        self.subjectID = subject_id
        self.model = model # 1: sgt, 2: sgt-ciil
        self.stage = stage  #sgt, fitts
        self.device = Device('sifi')

        self.get_device_parameters()
        self.get_feature_parameters()
        self.get_datacollection_parameters()
        self.get_classifier_parameters()
        self.get_training_hyperparameters()

    def get_device_parameters(self):
        if self.device.name == "sifi":
            self.window_length_s = 196 #ms
            self.window_increment_s = 56 #ms

        self.window_length = int((self.window_length_s*self.device.fs)/1000)
        self.window_increment = int((self.window_increment_s*self.device.fs)/1000)
        if self.stage == "sgt":
            self.log_to_file = False
        else:
            self.log_to_file = True

    def get_classifier_parameters(self):
        self.oc_output_format = "probabilities"
        self.shared_memory_items = [["model_output", (100,3), np.double], #timestamp, class prediction, confidence
                                    ["model_input", (100,1+self.input_shape), np.double], # timestamp, <- features ->
                                    ["adapt_flag", (1,1), np.int32],
                                    ["active_flag", (1,1), np.int8]]

        if self.model == 1: # baseline sgt no adaptation
            self.model_name       = "MLP"
            self.negative_method  = "mixed"
            self.loss_function    = "MSELoss"
            self.relabel_method   = None
            self.initialization   = "SGT"
            self.adaptation       = False
            self.adapt_PCs        = False
        elif self.model == 2: # sgt adaptation
            self.model_name       = "MLP"
            self.negative_method  = "mixed"
            self.loss_function    = "MSELoss"
            self.relabel_method   = "LabelSpreading"
            self.initialization   = "SGT"
            self.adaptation       = True
            self.adapt_PCs        = True
        self.WENG_SPEED_MIN = -20
        self.WENG_SPEED_MAX = -13
        self.lower_PC_percentile = 0.1
        self.upper_PC_percentile = 0.9
        self.to_NM_percentile = -0.25
    
    def get_feature_parameters(self):
        self.features = ["WENG"]
        # self.feature_dictionary = {"WENG_fs": 1500}
        self.feature_dictionary = {}
        fe = libemg.feature_extractor.FeatureExtractor()
        fake_window = np.random.randn(1,self.device.num_channels,self.window_length)
        returned_features = fe.extract_features(self.features, fake_window, self.feature_dictionary)
        self.input_shape = sum([returned_features[key].shape[1] for key in returned_features.keys()])

    def get_datacollection_parameters(self):
        self.DC_image_location = "images/"
        self.DC_data_location  = "data/subject" + str(self.subjectID) + "/sgt/"
        self.DC_reps           = 5
        self.DC_rep_time       = 3
        self.DC_rest_time      = 1
        self.DC_epochs         = 150

    def get_training_hyperparameters(self):
        self.batch_size = 64
        self.learning_rate = 2e-3
        self.adaptation_epochs = 5
        self.visualize_training = False
    
    def get_game_parameters(self):
        self.game_time = 600

    def setup_model(self, odh, save_dir, smi):
        if self.stage == "fitts":
            model_to_load = f"Data/subject{self.subjectID}/sgt/trial_{self.model}/sgt_mdl.pkl"
        else:
            raise ValueError(f"Tried to setup model when stage isn't set to 'fitts'. Got: {self.stage}.")
        with open(model_to_load, 'rb') as handle:
            loaded_mdl = pickle.load(handle)

    
        # offline_classifier.__setattr__("feature_params", loaded_mdl.feature_params)
        feature_list = self.features

        if smi is None:
            smm = False
        else:
            smm = True
        classifier = libemg.emg_predictor.OnlineEMGRegressor(offline_regressor=loaded_mdl,
                                                                window_size=self.window_length,
                                                                window_increment=self.window_increment,
                                                                online_data_handler=odh,
                                                                features=feature_list,
                                                                file_path = save_dir,
                                                                file=True,
                                                                smm=smm,
                                                                smm_items=smi,
                                                                std_out=False)
        classifier.predictor.model.net.eval()
        classifier.run(block=False)
        return classifier

    def load_sgt_data(self):
        # parse offline data into an offline data handler
        dataset_folder = f'./'
        package_function = lambda x, y: True
        metadata_fetchers = [libemg.data_handler.FilePackager(libemg.data_handler.RegexFilter("images/", ".txt", ["collection"], "labels"), package_function)]
            
        offdh = libemg.data_handler.OfflineDataHandler()
        offdh.get_data(dataset_folder, 
                    [libemg.data_handler.RegexFilter("_R_","_emg.csv",["0","1","2","3","4"], "reps"),
                        libemg.data_handler.RegexFilter("subject", "/",[str(self.subjectID)], "subjects")],
                    metadata_fetchers,
                        ",")
        return offdh

    def offdh_to_memory(self):
        offdh = self.load_sgt_data()
        train_windows, train_metadata = offdh.parse_windows(self.window_length, self.window_increment,metadata_operations={"labels": "last_sample"})
        fe = libemg.feature_extractor.FeatureExtractor()
        features = fe.extract_features(self.features, train_windows, self.feature_dictionary)
        features = torch.hstack([torch.tensor(features[key], dtype=torch.float32) for key in features.keys()])
        targets = torch.tensor(train_metadata["labels"], dtype=torch.float32)

        memory = Memory()
        memory.add_memories(experience_data = features,
                            experience_targets = targets)
        memory.memories_stored = features.shape[0]
        memory.shuffle()
        return memory

    def prepare_model_from_sgt(self):
        # prepare inner model
        sgt_memory = self.offdh_to_memory()
        mdl = MLP(self)
        mdl.fit(shuffle_every_epoch=True, memory=sgt_memory)
        # install mdl and thresholds to EMGClassifier
        offline_classifier = libemg.emg_predictor.EMGRegressor(mdl)
        
        # offline_classifier.__setattr__("feature_params", config.feature_dictionary)
        # save EMGClassifier to file
        with open(f"{self.DC_data_location}trial_{self.model}/sgt_mdl.pkl", 'wb') as handle:
            pickle.dump(offline_classifier, handle)

    def setup_live_processing(self):
        smi = self.device.stream()
        odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi)
        if self.log_to_file:
            odh.log_to_file()
        
        return odh, smi

    def make_sgt(self):
        odh, smi = self.setup_live_processing()
        args = {
            "online_data_handler": odh,
            "streamer": self.device.p,
            "media_folder": self.DC_image_location,
            "data_folder":  f"{self.DC_data_location}trial_{self.model}/",
            "num_reps":     self.DC_reps,
            "rep_time":     self.DC_rep_time,
            "rest_time":    self.DC_rest_time,
            "auto_advance": True
        }
        gui = libemg.gui.GUI(
            online_data_handler=args["online_data_handler"],
            args=args, debug=False)

        if not os.path.exists(f"{self.DC_image_location}collection.mp4"):
            raise FileNotFoundError("Couldn't find collection.mp4 file. Please generate training animation.")
        
        return gui
