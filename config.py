from pathlib import Path
from multiprocessing import Lock

import libemg
import numpy as np
import os
import pickle
import torch

from utils.models import MLP
from utils.adaptation import Memory
from utils.data_collection import Device, collect_data, make_collection_videos

class Config:
    def __init__(self, subject_id: str, model: str, stage: str, device: str):
        # usb towards the hand
        self.subject_id = subject_id
        self.model = model
        self.stage = stage
        self.device = Device(device)
        self.fi = libemg.filtering.Filter(self.device.fs)
        self.fi.install_common_filters()

        self.get_device_parameters()
        self.get_feature_parameters()
        self.get_datacollection_parameters()
        self.get_model_parameters()
        self.get_training_hyperparameters()
        self.get_adaptation_parameters()
        self.get_game_parameters()

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

    def get_model_parameters(self):
        self.loss_function = 'L1'
        self.shared_memory_items = [["model_output", (100,3), np.double], #timestamp, <DOFs>
                                    ["model_input", (100,1+self.input_shape), np.double], # timestamp, <- features ->
                                    ["environment_output", (100, 1+2), np.double],  # timestamp, <2 DoF optimal direction>, <2 DoF CIIL direction>
                                    ["adapt_flag", (1,1),  np.int32],
                                    ["active_flag", (1,1), np.int8],
                                    ["memory_update_flag", (1,1), np.int8]]
        for item in self.shared_memory_items:
            item.append(Lock())

    def get_feature_parameters(self):
        self.features = ["WENG"]
        # self.feature_dictionary = {"WENG_fs": 1500}
        self.feature_dictionary = {}
        fe = libemg.feature_extractor.FeatureExtractor()
        fake_window = np.random.randn(1,self.device.num_channels,self.window_length)
        returned_features = fe.extract_features(self.features, fake_window, self.feature_dictionary)
        self.input_shape = sum([returned_features[key].shape[1] for key in returned_features.keys()])

    def get_datacollection_parameters(self):
        self.DC_image_location = "images/abduct-adduct-flexion-extension/"
        self.DC_data_location = Path('data', self.subject_id, self.model).absolute().as_posix()
        self.DC_reps           = 5
        self.DC_epochs         = 150
        self.DC_model_file = Path(self.DC_data_location, 'sgt_mdl.pkl').absolute().as_posix()

    def get_adaptation_parameters(self):
        self.AD_model_file = Path(self.DC_data_location, 'ad_mdl.pkl').absolute().as_posix()

    def get_training_hyperparameters(self):
        self.batch_size = 64
        self.learning_rate = 2e-3
        self.adaptation_epochs = 5
        self.visualize_training = False
    
    def get_game_parameters(self):
        # self.game_time = 300
        self.game_time = 240

    def setup_online_model(self, online_data_handler, model_type):
        if self.stage == 'sgt':
            raise ValueError(f"Tried to setup online model when stage is set to 'sgt'.")

        self.prepare_model_from_sgt()

        if model_type == 'adaptation':
            model_file = self.DC_model_file
        elif model_type == 'validation':
            model_file = self.AD_model_file
        else:
            raise ValueError(f"Unexpected value for model_type. Got: {model_type}.")

        with open(model_file, 'rb') as handle:
            loaded_mdl = pickle.load(handle)
    
        if self.model_is_adaptive:
            smm = True
            smi = [
                ['model_input', (100, 1 + (8 * self.device.num_channels)), np.double], # timestamp <- features ->
                ['model_output', (100, 3), np.double]  # timestamp, prediction 1, prediction 2... (assumes 2 DOFs)
            ]
        else:
            smm = False
            smi = None

        model = libemg.emg_predictor.OnlineEMGRegressor(
            offline_regressor=loaded_mdl,
            window_size=self.window_length,
            window_increment=self.window_increment,
            online_data_handler=online_data_handler,
            features=self.features,
            file_path=Path(self.DC_model_file).parent.as_posix() + '/', # '/' needed to store model_output.txt in correct directory
            file=True,
            smm=smm,
            smm_items=smi,
            std_out=False
        )
        model.predictor.model.net.eval()
        model.run(block=False)
        return model

    def load_sgt_data(self):
        # parse offline data into an offline data handler
        package_function = lambda x, y: Path(x).parent == Path(y).parent
        metadata_fetchers = [libemg.data_handler.FilePackager(libemg.data_handler.RegexFilter('/', ".txt", ["labels"], "labels"), package_function)]
            
        offdh = libemg.data_handler.OfflineDataHandler()
        regex_filters = [
            libemg.data_handler.RegexFilter("_R_","_emg.csv",["0","1","2","3","4"], "reps"),
            libemg.data_handler.RegexFilter("/", "/",[str(self.subject_id)], "subjects"),
            libemg.data_handler.RegexFilter('/', '/', [self.model], 'model_data')
        ]
        offdh.get_data(self.DC_data_location, regex_filters, metadata_fetchers, ",")
        return offdh

    def offdh_to_memory(self):
        offdh = self.load_sgt_data()
        self.fi.filter(offdh)   # always apply filter to offline data

        train_windows, train_metadata = offdh.parse_windows(self.window_length, self.window_increment, metadata_operations={"labels": "last_sample"})
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
        if Path(self.DC_model_file).exists():
            return
        # prepare inner model
        sgt_memory = self.offdh_to_memory()
        mdl = MLP(self.input_shape, self.batch_size, self.learning_rate, self.loss_function, Path(self.DC_model_file).with_name('loss.csv').as_posix())
        print('Fitting model...')
        mdl.fit(num_epochs=self.DC_epochs, shuffle_every_epoch=True, memory=sgt_memory)
        # install mdl and thresholds to EMGClassifier
        offline_regressor = libemg.emg_predictor.EMGRegressor(mdl)
        
        # offline_classifier.__setattr__("feature_params", config.feature_dictionary)
        # save EMGClassifier to file
        with open(self.DC_model_file, 'wb') as handle:
            pickle.dump(offline_regressor, handle)


    def setup_live_processing(self):
        p, smi = self.device.stream()
        odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi)    # can't make this a field b/c pickling will throw an error
        if self.stage != 'sgt':
            # Only want raw data during SGT
            odh.install_filter(self.fi)
        if self.log_to_file:
            odh.log_to_file(file_path=self.DC_data_location + '/')
        return odh, p

    def start_sgt(self, online_data_handler):
        make_collection_videos()
        collect_data(online_data_handler, self.DC_image_location, self.DC_data_location, self.DC_reps)
