from pathlib import Path

import libemg
import numpy as np
import os
import pickle
import torch

from utils.models import MLP
from utils.adaptation import Memory
from utils.data_collection import Device, collect_data

class Config:
    def __init__(self, subject_id: str, model: str, stage: str):
        # usb towards the hand
        self.subject_id = subject_id
        self.model = model
        self.stage = stage
        self.device = Device('sifi')
        self.odh = None

        self.get_device_parameters()
        self.get_feature_parameters()
        self.get_datacollection_parameters()
        self.get_classifier_parameters()
        self.get_training_hyperparameters()

    @property
    def model_is_adaptive(self):
        if self.model in ['within-sgt', 'combined-sgt']: # baseline sgt no adaptation
            return False
        elif self.model in ['within-ciil', 'combined-ciil']: # sgt adaptation
            return True
        else:
            raise ValueError(f"Unexpected value for self.model. Got: {self.model}.")


    def get_device_parameters(self):
        window_length_ms = 150 #ms
        window_increment_ms = 40 #ms

        self.window_length = int((window_length_ms*self.device.fs)/1000)
        self.window_increment = int((window_increment_ms*self.device.fs)/1000)
        if self.stage == "sgt":
            # Do we want to log during user learning phase too????
            self.log_to_file = False
        else:
            self.log_to_file = True

    def get_classifier_parameters(self):
        self.oc_output_format = "probabilities"
        self.shared_memory_items = [["model_output", (100,3), np.double], #timestamp, class prediction, confidence
                                    ["model_input", (100,1+self.input_shape), np.double], # timestamp, <- features ->
                                    ["adapt_flag", (1,1), np.int32],
                                    ["active_flag", (1,1), np.int8]]

        if self.model_is_adaptive:
            self.model_name       = "MLP"
            self.negative_method  = "mixed"
            self.loss_function    = "MSELoss"
            self.relabel_method   = "LabelSpreading"
            self.initialization   = "SGT"
        else:
            self.model_name       = "MLP"
            self.negative_method  = "mixed"
            self.loss_function    = "MSELoss"
            self.relabel_method   = None
            self.initialization   = "SGT"

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
        self.DC_data_location = Path('data', 'subject', self.subject_id, 'sgt').absolute().as_posix()
        self.DC_reps           = 5
        self.DC_epochs         = 150
        self.DC_model_file = Path(self.DC_data_location, f"trial_{self.model}", 'sgt_mdl.pkl').absolute().as_posix()

    def get_training_hyperparameters(self):
        self.batch_size = 64
        self.learning_rate = 2e-3
        self.adaptation_epochs = 5
        self.visualize_training = False
    
    def get_game_parameters(self):
        self.game_time = 600

    def setup_model(self):
        if self.stage != 'fitts':
            raise ValueError(f"Tried to setup model when stage isn't set to 'fitts'. Got: {self.stage}.")

        with open(self.DC_model_file, 'rb') as handle:
            loaded_mdl = pickle.load(handle)
    
        # offline_classifier.__setattr__("feature_params", loaded_mdl.feature_params)
        feature_list = self.features


        if self.model_is_adaptive:
            smm = True
            smi = [
                ['model_input', (100, 1 + (8 * self.device.num_channels)), np.double], # timestamp <- features ->
                ['model_output', (100, 3), np.double]  # timestamp, prediction 1, prediction 2... (assumes 2 DOFs)
            ]
        else:
            smm = False
            smi = None
        model = libemg.emg_predictor.OnlineEMGRegressor(offline_regressor=loaded_mdl,
                                                                window_size=self.window_length,
                                                                window_increment=self.window_increment,
                                                                online_data_handler=self.odh,
                                                                features=feature_list,
                                                                file_path=Path(self.DC_model_file).parent.as_posix(),
                                                                file=True,
                                                                smm=smm,
                                                                smm_items=smi,
                                                                std_out=False)
        model.predictor.model.net.eval()
        model.run(block=False)
        return model

    def load_sgt_data(self):
        # parse offline data into an offline data handler
        dataset_folder = f'./'
        package_function = lambda x, y: True
        metadata_fetchers = [libemg.data_handler.FilePackager(libemg.data_handler.RegexFilter("images/", ".txt", ["collection"], "labels"), package_function)]
            
        offdh = libemg.data_handler.OfflineDataHandler()
        offdh.get_data(dataset_folder, 
                    [libemg.data_handler.RegexFilter("_R_","_emg.csv",["0","1","2","3","4"], "reps"),
                        libemg.data_handler.RegexFilter("subject", "/",[str(self.subject_id)], "subjects")],
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
        with open(self.DC_model_file, 'wb') as handle:
            pickle.dump(offline_classifier, handle)

    def setup_live_processing(self):
        if self.odh is not None:
            return self.odh

        smi = self.device.stream()
        self.odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi)
        if self.log_to_file:
            self.odh.log_to_file()
        return self.odh

    def make_sgt(self):
        self.setup_live_processing()

        if not os.path.exists(f"{self.DC_image_location}collection.mp4"):
            raise FileNotFoundError("Couldn't find collection.mp4 file. Please generate training animation.")
        
        collect_data(self.odh, self.DC_image_location, f"{self.DC_data_location}trial_{self.model}/", self.DC_reps)
