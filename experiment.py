import json
import shutil
import time
from pathlib import Path
from multiprocessing import Lock, Process
import logging
import traceback
from dataclasses import dataclass, asdict
from typing import ClassVar

import libemg
import numpy as np
import pickle
import torch
from PIL import Image

from utils.models import MLP
from utils.adaptation import Memory, WROTE, WAITING, DONE_TASK, make_pseudo_labels, AdaptationFitts, ADAPTATION_TIME
from utils.data_collection import Device, collect_data, get_frame_coordinates


MODELS = ('ciil', 'combined-sgt', 'oracle', 'within-sgt')   # TODO: change the order of these based on desired presentation if we recollect pilots (will need to be consistent for entire study)


@dataclass(frozen=True)
class Config:
    LOSS_FUNCTION: ClassVar[str] = 'L1'
    IMAGE_DIRECTORY: ClassVar[str] = 'images/'
    NUM_TRAIN_EPOCHS: ClassVar[int] = 150
    NUM_ADAPTATION_EPOCHS: ClassVar[int] = 5
    BATCH_SIZE: ClassVar[int] = 64
    LEARNING_RATE: ClassVar[float] = 2e-3
    WINDOW_LENGTH_MS: ClassVar[int] = 150 #ms
    WINDOW_INCREMENT_MS: ClassVar[int] = 40 #ms

    subject_directory: str
    dominant_hand: str
    device_name: str
    condition_idx: int

    def __post_init__(self):
        Path(self.subject_directory).mkdir(parents=True, exist_ok=True)

    @property
    def device(self):
        return Device(self.device_name)

    @property
    def model_is_adaptive(self):
        if self.model in ['within-sgt', 'combined-sgt']: # baseline sgt no adaptation
            return False
        elif self.model in ['ciil', 'oracle']: # sgt adaptation
            return True
        else:
            raise ValueError(f"Unexpected value for self.model. Got: {self.model}.")

    @property
    def window_length(self):
        return int((self.WINDOW_LENGTH_MS*self.device.fs)/1000)

    @property
    def window_increment(self):
        return int((self.WINDOW_INCREMENT_MS*self.device.fs)/1000)

    @property
    def features(self):
        return ['WENG']

    @property
    def feature_dictionary(self):
        return {'WENG_fs': self.device.fs}

    @property
    def subject_id(self):
        return Path(self.subject_directory).stem

    @property
    def condition_order(self):
        # Balanced latin square for 4 conditions
        conditions = np.array(MODELS)
        latin_square = np.array([
            [0, 1, 2, 3],
            [1, 2, 0, 3],
            [2, 3, 1, 0],
            [3, 0, 2, 1]
        ])

        # Determine model based on latin square
        subject_id = Path(self.subject_directory).stem
        subject_idx = (int(subject_id[-3:]) - 1) % len(latin_square)
        condition_idx_order = latin_square[subject_idx]
        return conditions[condition_idx_order]

    @property
    def model(self):
        return self.condition_order[self.condition_idx]
    
    @property
    def use_combined_data(self):
        return 'combined' in self.model

    @property
    def animation_location(self):
        if self.use_combined_data:
            animation_location = Path('animation', 'combined').absolute().as_posix()
        else:
            animation_location = Path('animation', 'within').absolute().as_posix()
        return animation_location

    @property
    def data_directory(self):
        return Path(self.subject_directory, self.model).absolute().as_posix()

    @property
    def num_reps(self):
        if self.model_is_adaptive:
            return 1
        elif 'combined' in self.model:
            return 3    # 3 of each video
        else:
            return 6    # 6 50-second videos makes 5 minutes

    @property
    def sgt_model_file(self):
        return Path(self.data_directory, 'sgt_mdl.pkl').absolute().as_posix()

    @property
    def adaptation_model_file(self):
        return Path(self.data_directory, 'adaptation_mdl.pkl').absolute().as_posix()

    @property
    def adaptation_fitts_file(self):
        return Path(self.sgt_model_file).with_name('adaptation_fitts.pkl').as_posix()

    @property
    def validation_fitts_file(self):
        return Path(self.sgt_model_file).with_name('validation_fitts.pkl').as_posix()

    @property
    def mapping(self):
        if self.dominant_hand == 'right':
            return 'polar-'
        elif self.dominant_hand == 'left':
            return 'polar+'
        else:
            raise ValueError(f"Unexpected value for handedness. Got: {self.dominant_hand}.")

    def save(self):
        config_file = Path(self.sgt_model_file).with_name('config.json')
        config_file.parent.mkdir(parents=True, exist_ok=False) # if the path already exists we'd have a latin square error, so throw an error
        with open(config_file, 'w') as f:
            json.dump(asdict(self), f)
        print('Saved to: ', config_file.as_posix())

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            config = Config(**json.load(f))
        return config


class Experiment:
    def __init__(self, config: Config, stage: str):
        self.config = config

        # Check for latin square order
        completed_models = [path.stem for path in Path(self.config.subject_directory).glob('*') 
                            if path.is_dir() and path.stem in self.config.condition_order and any(path.iterdir()) and path.stem != self.config.model]
        expected_models = self.config.condition_order[:self.config.condition_idx]
        assert set(completed_models) == set(expected_models), f"Mismatched latin square order. Expected {expected_models} to be completed, but got {completed_models}."

        self.stage = stage
        if self.stage == 'sgt':
            self.emg_log_filepath = None
            self.model_file = None
            self.install_filter = False
            self.adapt = False
        elif self.stage == 'adaptation':
            self.emg_log_filepath = Path(self.config.data_directory, "adaptation_").as_posix()
            self.model_file = self.config.sgt_model_file
            self.install_filter = True
            self.adapt = True
        elif self.stage == 'validation':
            self.emg_log_filepath = Path(self.config.data_directory, "validation_").as_posix()
            self.model_file = self.config.adaptation_model_file if self.config.model_is_adaptive else self.config.sgt_model_file    # only adaptive models have an adaptation model file
            self.install_filter = True
            self.adapt = False


        fe = libemg.feature_extractor.FeatureExtractor()
        fake_window = np.random.randn(1,self.config.device.num_channels,self.config.window_length)
        returned_features = fe.extract_features(self.config.features, fake_window, self.config.feature_dictionary)
        self.input_shape = sum([returned_features[key].shape[1] for key in returned_features.keys()])

        self.fi = libemg.filtering.Filter(self.config.device.fs)
        self.fi.install_common_filters()

        self.shared_memory_items = [["model_output", (100,3), np.double], #timestamp, <DOFs>
                                    ["model_input", (100,1+self.input_shape), np.double], # timestamp, <- features ->
                                    ["environment_output", (100, 3), np.double],  # timestamp, <2 DoF optimal direction>, 
                                    ["adapt_flag", (1,1),  np.int32],
                                    ["active_flag", (1,1), np.int8],
                                    ["memory_update_flag", (1,1), np.int8]]
        for item in self.shared_memory_items:
            item.append(Lock())

    def setup_online_model(self, online_data_handler):
        if self.model_file is None or self.emg_log_filepath is None:
            raise ValueError(f"Stage {self.stage} should not use an online model (model_file is None).")
        self.prepare_model_from_sgt()

        if not Path(self.model_file).exists():
            raise FileNotFoundError(f"Model file {self.model_file} not found.")

        with open(self.model_file, 'rb') as handle:
            loaded_mdl = pickle.load(handle)

        model = libemg.emg_predictor.OnlineEMGRegressor(
            offline_regressor=loaded_mdl,
            window_size=self.config.window_length,
            window_increment=self.config.window_increment,
            online_data_handler=online_data_handler,
            features=self.config.features,
            file_path=self.emg_log_filepath,
            file=True,
            smm=self.config.model_is_adaptive,
            smm_items=self.shared_memory_items,
            std_out=False
        )
        model.predictor.model.net.eval()
        model.run(block=False)
        return model

    def load_sgt_data(self):
        # parse offline data into an offline data handler
        def package_function(x, y):
            x_path = Path(x)
            y_path = Path(y)
            return x_path.parent == y_path.parent and x_path.name[2] == y_path.name[2]

        metadata_fetchers = [libemg.data_handler.FilePackager(libemg.data_handler.RegexFilter('/C_', ".txt", ['0', '1'], "labels"), package_function)]
            
        offdh = libemg.data_handler.OfflineDataHandler()
        regex_filters = [
            libemg.data_handler.RegexFilter("_R_","_emg.csv",["0","1","2","3","4"], "reps"),
            libemg.data_handler.RegexFilter("/", "/",[str(self.config.subject_id)], "subjects"),
            libemg.data_handler.RegexFilter('/', '/', [self.config.model], 'model_data')
        ]
        offdh.get_data(self.config.data_directory, regex_filters, metadata_fetchers, ",")
        return offdh

    def offdh_to_memory(self):
        offdh = self.load_sgt_data()
        self.fi.filter(offdh)   # always apply filter to offline data

        train_windows, train_metadata = offdh.parse_windows(self.config.window_length, self.config.window_increment, metadata_operations={"labels": "last_sample"})
        fe = libemg.feature_extractor.FeatureExtractor()
        features = fe.extract_features(self.config.features, train_windows, self.config.feature_dictionary)
        features = torch.hstack([torch.tensor(features[key], dtype=torch.float32) for key in features.keys()])
        targets = torch.tensor(train_metadata["labels"], dtype=torch.float32)

        memory = Memory()
        memory.add_memories(experience_data = features,
                            experience_targets = targets)
        memory.memories_stored = features.shape[0]
        memory.shuffle()
        return memory

    def prepare_model_from_sgt(self):
        if Path(self.config.sgt_model_file).exists():
            return
        # prepare inner model
        sgt_memory = self.offdh_to_memory()
        mdl = MLP(self.input_shape, self.config.BATCH_SIZE, self.config.LEARNING_RATE, self.config.LOSS_FUNCTION, Path(self.config.sgt_model_file).with_name('loss.csv').as_posix())
        print('Fitting SGT model...')
        mdl.fit(num_epochs=self.config.NUM_TRAIN_EPOCHS, shuffle_every_epoch=True, memory=sgt_memory)
        # install mdl and thresholds to EMGClassifier
        offline_regressor = libemg.emg_predictor.EMGRegressor(mdl)
        offline_regressor.install_feature_parameters(self.config.feature_dictionary)
        
        # save EMGClassifier to file
        with open(self.config.sgt_model_file, 'wb') as handle:
            pickle.dump(offline_regressor, handle)

    def setup_live_processing(self):
        p, smi = self.config.device.stream()
        odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi)    # can't make this a field b/c pickling will throw an error
        if self.install_filter:
            # Only want raw data during SGT
            odh.install_filter(self.fi)
        if self.emg_log_filepath is not None:
            odh.log_to_file(file_path=self.emg_log_filepath)
        return odh, p

    def start_sgt(self, online_data_handler):
        self.make_collection_video('within')
        if self.config.use_combined_data:
            self.make_collection_video('combined')

        collect_data(online_data_handler, self.config.animation_location, self.config.data_directory, self.config.num_reps)

    def make_collection_video(self, video):
        libemg.gui.GUI(None).download_gestures([2, 3, 6, 7], self.config.IMAGE_DIRECTORY)
        gestures = ["Hand_Open", "Hand_Close", "Pronation", "Supination"]
        pictures = {}
        for g in gestures:
            pictures[g] = Image.open(Path(self.config.IMAGE_DIRECTORY, f"{g}.png"))

        filepath = Path(self.config.animation_location, f"{video}.mp4").absolute()
        if filepath.exists():
            return

        filepath.parent.mkdir(parents=True, exist_ok=True)
        matching_video_paths = [path for path in filepath.parent.parent.rglob('*') if path.name == filepath.name]
        if len(matching_video_paths) > 0:
            # Found matching file
            assert len(matching_video_paths) == 1, f"Found multiple matching video names {filepath}."
            matching_video_path = matching_video_paths[0]
            shutil.copy(matching_video_path, filepath)
            shutil.copy(matching_video_path.with_suffix('.txt'), filepath.with_suffix('.txt'))
            return

        print(f"Creating {video} collection video...")
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
        assert self.config.model_is_adaptive, f"Attempted to perform adaptation for non-adaptive model. Terminating script."
        memory_process = Process(target=self._memory_manager, daemon=True)
        memory_process.start()

        adapt_process = Process(target=self._adapt_manager, daemon=True, args=(emg_predictor, ))
        adapt_process.start()
        return memory_process, adapt_process

    def _adapt_manager(self, emg_predictor):
        if not self.config.model_is_adaptive:
            raise ValueError(f"Model {self.config.model} should not be adapted.")
        logging.basicConfig(filename=Path(self.config.data_directory, "adaptmanager.log"),
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
        
        while (time.perf_counter() - start_time) < ADAPTATION_TIME:
            try:
                data = smm.get_variable('memory_update_flag')[0, 0]
                if data == WROTE:
                    # we were signalled we have data to load
                    # append this data to our memory
                    t1 = time.perf_counter()
                    new_memory = Memory()
                    new_memory.from_file(self.config.data_directory, memory_id)
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
                    emg_predictor.model.adapt(memory, num_epochs=self.config.NUM_ADAPTATION_EPOCHS)
                    del_t = time.perf_counter() - t1
                    logging.info(f"ADAPTMANAGER: ADAPTED - round {adapt_round}; \tADAPT TIME: {del_t:.2f}s")
                    
                    with open(Path(self.config.data_directory, 'adaptation_mdl' + str(adapt_round) + '.pkl'), 'wb') as handle:
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
            memory.write(self.config.data_directory, 1000)
            with open(self.config.adaptation_model_file, 'wb') as handle:
                pickle.dump(emg_predictor, handle)

    def _memory_manager(self):
        logging.basicConfig(filename=Path(self.config.data_directory, "memorymanager.log"),
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
                
                result = make_pseudo_labels(environment_data, smm, approach=self.config.model)
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
                        memory.write(self.config.data_directory, num_written)
                        del_t = time.perf_counter() - t1
                        logging.info(f"MEMORYMANAGER: WROTE FILE: {num_written},\t lines:{len(memory)},\t unfound: {total_samples_unfound},\t WRITE TIME: {del_t:.2f}s")
                        num_written += 1
                        memory = Memory()
                        smm.modify_variable('memory_update_flag', lambda _: WROTE)    # tell adapt manager that it has new data
            except:
                logging.error("MEMORYMANAGER: "+traceback.format_exc())
        print('memory_manager finished!')

    def run_isofitts(self, online_data_handler):
        online_regressor = self.setup_online_model(online_data_handler)
        if self.adapt:
            memory_process, adapt_process = self.start_adapting(online_regressor.predictor)
            save_file = self.config.adaptation_fitts_file
        else:
            save_file = self.config.validation_fitts_file
            memory_process = None
            adapt_process = None
        isofitts = AdaptationFitts(self.shared_memory_items, save_file=save_file, adapt=self.adapt, mapping=self.config.mapping)
        isofitts.run()

        if memory_process is not None:
            memory_process.join()
        
        if adapt_process is not None:
            adapt_process.join()
