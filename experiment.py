import shutil
import time
from pathlib import Path
from multiprocessing import Lock, Process
import logging
import traceback
from dataclasses import dataclass, field
from typing import ClassVar

import libemg
import numpy as np
import pickle
import torch
from PIL import Image

from utils.models import MLP
from utils.adaptation import Memory, WROTE, WAITING, DONE_TASK, make_pseudo_labels
from utils.data_collection import Device, collect_data, get_frame_coordinates

# Balanced latin square for 4 conditions
MODELS = np.array(['ciil', 'combined-sgt', 'oracle', 'within-sgt'])
LATIN_SQUARE = np.array([
    [0, 1, 2, 3],
    [1, 2, 0, 3],
    [2, 3, 1, 0],
    [3, 0, 2, 1]
])


@dataclass(frozen=True)
class Config:
    LOSS_FUNCTION: ClassVar[str] = 'L1'
    IMAGE_DIRECTORY: ClassVar[str] = 'images/'
    NUM_TRAIN_EPOCHS: ClassVar[int] = 150
    NUM_ADAPTATION_EPOCHS: ClassVar[int] = 5
    BATCH_SIZE: ClassVar[int] = 64
    LEARNING_RATE: ClassVar[float] = 2e-3
    GAME_TIME: ClassVar[int] = 240  # TODO: Have adaptation game time and validation game time
    NUM_CIRCLES: ClassVar[int] = 8
    NUM_TRIALS: ClassVar[int] = 2000  # set a large number so it will be triggered by time instead of trials
    DWELL_TIME: ClassVar[float] = 2.0
    WINDOW_LENGTH_MS: ClassVar[int] = 150 #ms
    WINDOW_INCREMENT_MS: ClassVar[int] = 40 #ms

    subject_id: str
    stage: str
    device: Device
    features: list = field(init=False, repr=False, default_factory=lambda: ['WENG'])

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
    def log_to_file(self):
        # Do we want to log during user learning phase too????
        return self.stage != 'sgt'

    @property
    def feature_dictionary(self):
        return {'WENG_fs': self.device.fs}

    @property
    def subject_directory(self):
        return Path('data', self.subject_id)

    @property
    def model(self):
        # Determine model based on latin square
        subject_idx = (int(self.subject_id[-3:]) - 1) % len(LATIN_SQUARE)
        model_mask = LATIN_SQUARE[subject_idx]
        ordered_models = MODELS[model_mask]
        completed_models = [path.stem for path in self.subject_directory.glob('*') if path.is_dir() and path.stem in ordered_models and any(path.iterdir())]
        expected_models = ordered_models[:len(completed_models)]
        assert np.all(np.sort(completed_models) == np.sort(expected_models)), f"Mismatched latin square order. Expected {expected_models} to be completed, but got {completed_models}."

        if self.stage == 'sgt':
            model = ordered_models[len(completed_models)]
        else:
            assert len(completed_models) >= 1, f"Got 0 completed models. Please perform SGT first."
            model = ordered_models[len(completed_models) - 1]
        return model

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
        return self.subject_directory.joinpath(self.model).absolute().as_posix()

    @property
    def num_reps(self):
        return 1 if self.model_is_adaptive else 5

    @property
    def sgt_model_file(self):
        return Path(self.data_directory, 'sgt_mdl.pkl').absolute().as_posix()

    @property
    def adaptation_model_file(self):
        return Path(self.data_directory, 'ad_mdl.pkl').absolute().as_posix()

    @property
    def adaptation_fitts_file(self):
        return Path(self.sgt_model_file).with_name('ad_fitts.pkl').as_posix()

    @property
    def validation_fitts_file(self):
        # self.game_time = 300
        return Path(self.sgt_model_file).with_name('val_fitts.pkl').as_posix()


class Experiment:
    def __init__(self, config: Config):
        self.config = config
        self.fi = libemg.filtering.Filter(self.config.device.fs)
        self.fi.install_common_filters()

        self.shared_memory_items = [["model_output", (100,3), np.double], #timestamp, <DOFs>
                                    ["model_input", (100,1+self.input_shape), np.double], # timestamp, <- features ->
                                    ["environment_output", (100, 1+2), np.double],  # timestamp, <2 DoF optimal direction>, <2 DoF CIIL direction>
                                    ["adapt_flag", (1,1),  np.int32],
                                    ["active_flag", (1,1), np.int8],
                                    ["memory_update_flag", (1,1), np.int8]]
        for item in self.shared_memory_items:
            item.append(Lock())



    @property
    def input_shape(self):
        fe = libemg.feature_extractor.FeatureExtractor()
        fake_window = np.random.randn(1,self.config.device.num_channels,self.config.window_length)
        returned_features = fe.extract_features(self.config.features, fake_window, self.config.feature_dictionary)
        return sum([returned_features[key].shape[1] for key in returned_features.keys()])


    def setup_online_model(self, online_data_handler, model_type):
        if self.config.stage == 'sgt':
            raise ValueError(f"Tried to setup online model when stage is set to 'sgt'.")

        self.prepare_model_from_sgt()

        if model_type == 'adaptation':
            model_file = self.config.sgt_model_file
            previous_stage = 'sgt'
        elif model_type == 'validation':
            model_file = self.config.adaptation_model_file
            previous_stage = 'adaptation'
        else:
            raise ValueError(f"Unexpected value for model_type. Got: {model_type}.")

        if not Path(model_file).exists():
            raise FileNotFoundError(f"{model_file} not found. Please run {previous_stage} first.")

        with open(model_file, 'rb') as handle:
            loaded_mdl = pickle.load(handle)
    
        if self.config.model_is_adaptive:
            smm = True
            smi = [
                ['model_input', (100, 1 + (8 * self.config.device.num_channels)), np.double], # timestamp <- features ->
                ['model_output', (100, 3), np.double]  # timestamp, prediction 1, prediction 2... (assumes 2 DOFs)
            ]
        else:
            smm = False
            smi = None

        model = libemg.emg_predictor.OnlineEMGRegressor(
            offline_regressor=loaded_mdl,
            window_size=self.config.window_length,
            window_increment=self.config.window_increment,
            online_data_handler=online_data_handler,
            features=self.config.features,
            file_path=Path(self.config.sgt_model_file).parent.as_posix() + '/', # '/' needed to store model_output.txt in correct directory
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
        package_function = lambda x, y: Path(x).parent == Path(y).parent and x[2] == y[2]
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
        print('Fitting model...')
        mdl.fit(num_epochs=self.config.NUM_TRAIN_EPOCHS, shuffle_every_epoch=True, memory=sgt_memory)
        # install mdl and thresholds to EMGClassifier
        offline_regressor = libemg.emg_predictor.EMGRegressor(mdl)
        
        # offline_classifier.__setattr__("feature_params", config.feature_dictionary)
        # save EMGClassifier to file
        with open(self.config.sgt_model_file, 'wb') as handle:
            pickle.dump(offline_regressor, handle)


    def setup_live_processing(self):
        p, smi = self.config.device.stream()
        odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi)    # can't make this a field b/c pickling will throw an error
        if self.config.stage != 'sgt':
            # Only want raw data during SGT
            odh.install_filter(self.fi)
        if self.config.log_to_file:
            odh.log_to_file(file_path=self.config.data_directory + '/')
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
        if not self.config.model_is_adaptive and not Path(self.config.adaptation_model_file).exists():
            # Model should not be adapted, so we save the SGT model as the adapted model
            shutil.copy(self.config.sgt_model_file, self.config.adaptation_model_file)
            print(f"Model {self.config.model} should not be adapted - copied SGT model ({self.config.sgt_model_file}) to adaptive model ({self.config.adaptation_model_file}).")
            return

        memoryProcess = Process(target=self._memory_manager, daemon=True)
        memoryProcess.start()

        adaptProcess = Process(target=self._adapt_manager, daemon=True, args=(emg_predictor, ))
        adaptProcess.start()

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
        
        while (time.perf_counter() - start_time) < self.config.GAME_TIME:
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
                    
                    with open(Path(self.config.data_directory, 'mdl' + str(adapt_round) + '.pkl'), 'wb') as handle:
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
            with open(Path(self.config.data_directory, 'ad_mdl.pkl'), 'wb') as handle:
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
