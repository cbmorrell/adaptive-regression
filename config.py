import libemg
import numpy as np
import os

class Config:
    def __init__(self):
        # usb towards the hand
        self.subjectID = 1000
        self.model = 1 # 1: sgt, 2: sgt-ciil
        self.stage = "fitts" #sgt, 

        self.get_device_parameters()
        self.get_feature_parameters()
        self.get_datacollection_parameters()
        self.get_classifier_parameters()
        self.get_training_hyperparameters()

    def setup_folders(self):
        if not os.path.exists("data"):
            os.mkdir("data")
        if not os.path.exists(f"data/subject{self.subjectID}"):
            os.mkdir(         f"data/subject{self.subjectID}")
        if not os.path.exists(f"data/subject{self.subjectID}/{self.stage}"):
            os.mkdir(         f"data/subject{self.subjectID}/{self.stage}")
        if not os.path.exists(f"data/subject{self.subjectID}/{self.stage}/trial_{self.model}"):
            os.mkdir(         f"data/subject{self.subjectID}/{self.stage}/trial_{self.model}")


    def get_device_parameters(self):
        self.device = "sifi"
        if self.device == "sifi":
            self.window_length_s = 196 #ms
            self.window_increment_s = 56 #ms
            self.num_channels = 8
            self.sampling_rate = 1500
            self.streamer_handle = libemg.streamers.sifi_bioarmband_streamer
            self.streamer_arguments = {"shared_memory_items": [["emg", (3000, 8), np.double],
                                                               ["emg_count", (1,1), np.int32 ],],
                                       "emg":True,
                                       "eda":False,
                                       "imu":False,
                                       "ppg":False
                                       }
        self.window_length = int((self.window_length_s*self.sampling_rate)/1000)
        self.window_increment = int((self.window_increment_s*self.sampling_rate)/1000)
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
        fake_window = np.random.randn(1,self.num_channels,self.window_length)
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