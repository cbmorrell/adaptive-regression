import shutil
import time
from pathlib import Path
from multiprocessing import Lock, Process
import logging
import traceback

import libemg
import numpy as np
import pickle
import torch
from PIL import Image

from utils.models import MLP
from utils.adaptation import Memory, WROTE, WAITING, DONE_TASK, make_pseudo_labels
from utils.data_collection import Device, collect_data, get_frame_coordinates

# Balanced latin square for 4 conditions
LATIN_SQUARE = np.array([
    [0, 1, 2, 3],
    [1, 2, 0, 3],
    [2, 3, 1, 0],
    [3, 0, 2, 1]
])

class Experiment:
    def __init__(self, subject_id: str, stage: str, device: str):
        # usb towards the hand
        self.subject_id = subject_id
        self.stage = stage
        self.device = Device(device)
        self.fi = libemg.filtering.Filter(self.device.fs)
        self.fi.install_common_filters()

        # Determine model based on latin square
        subject_idx = (int(self.subject_id[-3:]) - 1) % len(LATIN_SQUARE)
        self.model_order = LATIN_SQUARE[subject_idx]
        models = ['within-sgt', 'combined-sgt', 'ciil', 'oracle']
        completed_models = [path.stem for path in Path('data', self.subject_id).glob('*') if path.is_dir()]

        for model_idx in self.model_order[:len(completed_models)]:
            assert models[model_idx] in completed_models, f"Mismatched latin square order. Expected model {models[model_idx]} to be completed, but couldn't find data."

        self.model = models[self.model_order[len(completed_models)]]

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
        elif self.model in ['ciil', 'oracle']: # sgt adaptation
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
        self.animation_location = 'animation'
        self.image_location = 'images/'
        self.data_directory = Path('data', self.subject_id, self.model).absolute().as_posix()
        self.num_reps = 5
        self.num_train_epochs = 150
        self.sgt_model_file = Path(self.data_directory, 'sgt_mdl.pkl').absolute().as_posix()

    def get_adaptation_parameters(self):
        self.adaptation_model_file = Path(self.data_directory, 'ad_mdl.pkl').absolute().as_posix()
        self.num_adaptation_epochs = 5

    def get_training_hyperparameters(self):
        self.batch_size = 64
        self.learning_rate = 2e-3
    
    def get_game_parameters(self):
        # self.game_time = 300
        self.game_time = 240

    def setup_online_model(self, online_data_handler, model_type):
        if self.stage == 'sgt':
            raise ValueError(f"Tried to setup online model when stage is set to 'sgt'.")

        self.prepare_model_from_sgt()

        if model_type == 'adaptation':
            model_file = self.sgt_model_file
            previous_stage = 'sgt'
        elif model_type == 'validation':
            model_file = self.adaptation_model_file
            previous_stage = 'adaptation'
        else:
            raise ValueError(f"Unexpected value for model_type. Got: {model_type}.")

        if not Path(model_file).exists():
            raise FileNotFoundError(f"{model_file} not found. Please run {previous_stage} first.")

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
            file_path=Path(self.sgt_model_file).parent.as_posix() + '/', # '/' needed to store model_output.txt in correct directory
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
        offdh.get_data(self.data_directory, regex_filters, metadata_fetchers, ",")
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
        if Path(self.sgt_model_file).exists():
            return
        # prepare inner model
        sgt_memory = self.offdh_to_memory()
        mdl = MLP(self.input_shape, self.batch_size, self.learning_rate, self.loss_function, Path(self.sgt_model_file).with_name('loss.csv').as_posix())
        print('Fitting model...')
        mdl.fit(num_epochs=self.num_train_epochs, shuffle_every_epoch=True, memory=sgt_memory)
        # install mdl and thresholds to EMGClassifier
        offline_regressor = libemg.emg_predictor.EMGRegressor(mdl)
        
        # offline_classifier.__setattr__("feature_params", config.feature_dictionary)
        # save EMGClassifier to file
        with open(self.sgt_model_file, 'wb') as handle:
            pickle.dump(offline_regressor, handle)


    def setup_live_processing(self):
        p, smi = self.device.stream()
        odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi)    # can't make this a field b/c pickling will throw an error
        if self.stage != 'sgt':
            # Only want raw data during SGT
            odh.install_filter(self.fi)
        if self.log_to_file:
            odh.log_to_file(file_path=self.data_directory + '/')
        return odh, p

    def start_sgt(self, online_data_handler):
        self.make_collection_video('within')
        self.make_collection_video('combined')
        collect_data(online_data_handler, self.animation_location, self.data_directory, self.num_reps)

    def make_collection_video(self, video):
        libemg.gui.GUI(None).download_gestures([2, 3, 6, 7], self.image_location)
        gestures = ["Hand_Open", "Hand_Close", "Pronation", "Supination"]
        pictures = {}
        for g in gestures:
            pictures[g] = Image.open(Path(self.image_location, f"{g}.png"))

        filepath = Path(self.animation_location, f"{video}.mp4").absolute()
        if filepath.exists():
            return

        print(f"Creating {video} collection video...")
        filepath.mkdir(parents=True, exist_ok=True)
        animator = libemg.animator.ScatterPlotAnimator(output_filepath=filepath.as_posix(),
                                                        axis_images={"N":pictures["Supination"],
                                                                    "E":pictures["Hand_Open"],
                                                                    "S":pictures["Pronation"],
                                                                    "W":pictures["Hand_Close"]},
                                                        show_direction=True,
                                                        show_countdown=True)
        coordinates = get_frame_coordinates(movement_type=video)
        animator.save_plot_video(coordinates, title='Regression Training', save_coordinates=True, verbose=True)

    def start_adapting(self, emg_predictor):
        if not self.model_is_adaptive and not Path(self.adaptation_model_file).exists():
            # Model should not be adapted, so we save the SGT model as the adapted model
            shutil.copy(self.sgt_model_file, self.adaptation_model_file)
            print(f"Model {self.model} should not be adapted - copied SGT model ({self.sgt_model_file}) to adaptive model ({self.adaptation_model_file}).")
            return

        memoryProcess = Process(target=self._memory_manager, daemon=True)
        memoryProcess.start()

        adaptProcess = Process(target=self._adapt_manager, daemon=True, args=(emg_predictor, ))
        adaptProcess.start()

    def _adapt_manager(self, emg_predictor):
        if not self.model_is_adaptive:
            raise ValueError(f"Model {self.model} should not be adapted.")
        logging.basicConfig(filename=Path(self.data_directory, "adaptmanager.log"),
                            filemode='w',
                            encoding="utf-8",
                            level=logging.INFO)

        smm = libemg.shared_memory_manager.SharedMemoryManager()
        for item in self.shared_memory_items:
            smm.find_variable(*item)

        # initialize the memomry
        memory = self.offdh_to_memory()
        memory_id = 0
        num_memories = 0

        # initial time
        start_time = time.perf_counter()

        # variables to save and stuff
        adapt_round = 0
        
        time.sleep(3)
        
        while (time.perf_counter() - start_time) < self.game_time:
            try:
                data = smm.get_variable('memory_update_flag')[0, 0]
                if data == WROTE:
                    # we were signalled we have data to load
                    # append this data to our memory
                    t1 = time.perf_counter()
                    new_memory = Memory()
                    new_memory.from_file(self.data_directory, memory_id)
                    print(f"Loaded {memory_id} memory")
                    memory += new_memory
                    del_t = time.perf_counter() - t1
                    memory_id += 1
                    logging.info(f"ADAPTMANAGER: ADDED MEMORIES, \tCURRENT SIZE: {len(memory)}; \tLOAD TIME: {del_t:.2f}s")
                # if we still have no memories (rare edge case)
                if not len(memory):
                    logging.info("NO MEMORIES -- SKIPPED TRAINING")
                    t1 = time.perf_counter()
                    time.sleep(3)
                    del_t = time.perf_counter() - t1
                    logging.info(f"ADAPTMANAGER: WAITING - round {adapt_round}; \tWAIT TIME: {del_t:.2f}s")
                elif num_memories != len(memory):
                    # abstract decoders/fake abstract decoder/sgt
                    num_memories = len(memory)
                    t1 = time.perf_counter()
                    emg_predictor.model.adapt(memory, num_epochs=self.num_adaptation_epochs)
                    del_t = time.perf_counter() - t1
                    logging.info(f"ADAPTMANAGER: ADAPTED - round {adapt_round}; \tADAPT TIME: {del_t:.2f}s")
                    
                    with open(Path(self.data_directory, 'mdl' + str(adapt_round) + '.pkl'), 'wb') as handle:
                        pickle.dump(emg_predictor, handle)

                    smm.modify_variable("adapt_flag", lambda x: adapt_round)
                    print(f"Adapted {adapt_round} times")
                    adapt_round += 1

                # signal to the memory manager we are idle and waiting for data
                smm.modify_variable('memory_update_flag', lambda _: WAITING)
                logging.info("ADAPTMANAGER: WAITING FOR DATA")
                time.sleep(0.5)
            except:
                logging.error("ADAPTMANAGER: "+traceback.format_exc())
        else:
            print("AdaptManager Finished!")
            smm.modify_variable('memory_update_flag', lambda _: DONE_TASK)
            memory.write(self.data_directory, 1000)
            with open(Path(self.data_directory, 'ad_mdl.pkl'), 'wb') as handle:
                pickle.dump(emg_predictor, handle)

    def _memory_manager(self):
        logging.basicConfig(filename=Path(self.data_directory, "memorymanager.log"),
                            filemode='w',
                            encoding="utf-8",
                            level=logging.INFO)

        smm = libemg.shared_memory_manager.SharedMemoryManager()
        for item in self.shared_memory_items:
            smm.find_variable(*item)

        # initialize the memory
        memory = Memory()

        start_time = time.perf_counter()

        num_written = 0
        total_samples_unfound = 0
        last_timestamp = 0.

        while True:
            try:
                memory_data = smm.get_variable('memory_update_flag')[0, 0]

                if memory_data == DONE_TASK:
                    # Task is complete
                    del_t = time.perf_counter() - start_time
                    logging.info(f"MEMORYMANAGER: GOT DONE FLAG AT {del_t:.2f}s")
                    break

                environment_data = smm.get_variable('environment_output')[0]
                timestamp = environment_data[0]
                if timestamp == last_timestamp:
                    # No new data has been received
                    continue
                last_timestamp = timestamp
                
                result = make_pseudo_labels(environment_data, smm, approach=self.model)
                if result is None:
                    total_samples_unfound += 1
                    continue
                adaptation_data, adaptation_labels, adaptation_direction, adaptation_type, timestamp = result 
                if (adaptation_data.shape[0]) != (adaptation_labels.shape[0]):
                    continue
                memory.add_memories(adaptation_data, adaptation_labels, adaptation_direction, adaptation_type, timestamp)

                if memory_data == WAITING:
                    # write memory to file
                    if len(memory):# never write an empty memory
                        t1 = time.perf_counter()
                        memory.write(self.data_directory, num_written)
                        del_t = time.perf_counter() - t1
                        logging.info(f"MEMORYMANAGER: WROTE FILE: {num_written},\t lines:{len(memory)},\t unfound: {total_samples_unfound},\t WRITE TIME: {del_t:.2f}s")
                        num_written += 1
                        memory = Memory()
                        smm.modify_variable('memory_update_flag', lambda _: WROTE)    # tell adapt manager that it has new data
            except:
                logging.error("MEMORYMANAGER: "+traceback.format_exc())
        print('memory_manager finished!')
