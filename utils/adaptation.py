from pathlib import Path
import random
import pickle
import logging
import time
from multiprocessing import Lock
import traceback

import torch
import numpy as np
import libemg


WAITING = 0
WROTE = 1
DONE_TASK = -1000


class AdaptationIsoFitts(libemg.environments.isofitts.IsoFitts):
    def __init__(self, controller: libemg.environments.controllers.Controller, prediction_map: dict | None = None, num_circles: int = 30, num_trials: int = 15, dwell_time: float = 3, timeout: float = 30, velocity: float = 25, save_file: str | None = None, width: int = 1250, height: int = 750, fps: int = 60, proportional_control: bool = True):
        super().__init__(controller, prediction_map, num_circles, num_trials, dwell_time, timeout, velocity, save_file, width, height, fps, proportional_control)
        self.smm = libemg.shared_memory_manager.SharedMemoryManager()
        # TODO: WE MAKE A NEW LOCK HERE, SO MAYBE THIS IS WHY WE'RE HAVING ISSUES
        self.smm.find_variable('environment_output', (100, 3), np.double, Lock())

    def _log(self, label, timestamp):
        # Write to log_dictionary
        super()._log(label, timestamp)

        # Want to send the timestamp and the optimal direction
        optimal_direction = np.array(self.log_dictionary['goal_circle'][-1]) - np.array(self.log_dictionary['cursor_position'][-1])
        optimal_direction[1] *= -1  # multiply by -1 b/c pygame origin is top left, so a lower target has a higher y value
        output = np.array([timestamp, optimal_direction[0], optimal_direction[1]], dtype=np.double)
        self.smm.modify_variable('environment_output', lambda x: np.vstack((output, x))[:x.shape[0]])  # ensure we don't take more than original array size

    def run(self):
        super().run()
        self.smm.modify_variable('environment_output', lambda x: np.vstack((np.array([DONE_TASK, DONE_TASK, DONE_TASK]), x))[:x.shape[0]])


class Memory:
    def __init__(self, max_len=None):
        # What are the current targets for the model?
        self.experience_targets    = []
        # What are the inputs for the saved experiences?
        self.experience_data       = []
        # What are the correct options (given the context)
        self.experience_context    = []
        # What was the outcome (P or N)
        self.experience_outcome    = []
        # How many memories do we have?
        self.experience_timestamps = []
        self.memories_stored = 0
    
    def __len__(self):
        return self.memories_stored
    
    def __add__(self, other_memory):
        assert type(other_memory) == Memory
        if len(other_memory):
            if not len(self):
                self.experience_targets    = other_memory.experience_targets
                self.experience_data       = other_memory.experience_data
                self.experience_context    = other_memory.experience_context
                self.experience_outcome    = other_memory.experience_outcome
                self.experience_timestamps = other_memory.experience_timestamps
                self.memories_stored       = other_memory.memories_stored
            else:
                self.experience_targets = torch.cat((self.experience_targets,other_memory.experience_targets))
                self.experience_data = torch.vstack((self.experience_data,other_memory.experience_data))
                self.experience_context = np.concatenate((self.experience_context,other_memory.experience_context))
                self.experience_outcome = np.concatenate((self.experience_outcome, other_memory.experience_outcome)) 
                self.experience_timestamps.extend(other_memory.experience_timestamps)
                self.memories_stored += other_memory.memories_stored
        return self
        
    def add_memories(self, experience_data, experience_targets, experience_context=[], experience_outcome=[], experience_timestamps=[]):
        if len(experience_targets):
            if not len(self):
                self.experience_targets = experience_targets
                self.experience_data    = experience_data
                self.experience_context = experience_context
                self.experience_outcome = experience_outcome
                self.experience_timestamps = experience_timestamps
                self.memories_stored    += len(experience_targets)
            else:
                self.experience_targets = torch.cat((self.experience_targets,experience_targets))
                self.experience_data = torch.vstack((self.experience_data,experience_data))
                self.experience_context = np.concatenate((self.experience_context,experience_context))
                self.experience_outcome = np.concatenate((self.experience_outcome, experience_outcome)) 
                self.experience_timestamps.extend(experience_timestamps)
                self.memories_stored += len(experience_targets)
    
    def shuffle(self):
        if len(self):
            indices = list(range(len(self)))
            random.shuffle(indices)
            # shuffle the keys
            self.experience_targets = self.experience_targets[indices]
            self.experience_data    = self.experience_data[indices]
            # SGT does not have these fields
            if len(self.experience_context):
                self.experience_context = self.experience_context[indices]
                self.experience_outcome = [self.experience_outcome[i] for i in indices]
                self.experience_timestamps = [self.experience_timestamps[i] for i in indices]

        
    def unshuffle(self):
        # TODO: Test that this works using timestamp instead of IDs
        unshuffle_ids = [i[0] for i in sorted(enumerate(self.experience_timestamps), key=lambda x:x[1])]
        if len(self):
            self.experience_targets = self.experience_targets[unshuffle_ids]
            self.experience_data    = self.experience_data[unshuffle_ids]
            # SGT does not have these fields
            if len(self.experience_context):
                self.experience_context = self.experience_context[unshuffle_ids]
                self.experience_outcome = [self.experience_outcome[i] for i in unshuffle_ids]
                self.experience_timestamps = [self.experience_timestamps[i] for i in unshuffle_ids]

    def write(self, save_dir, num_written=""):
        with open(save_dir + f'classifier_memory_{num_written}.pkl', 'wb') as handle:
            pickle.dump(self, handle)
    
    def read(self, save_dir):
        with open(save_dir +  'classifier_memory.pkl', 'rb') as handle:
            loaded_content = pickle.load(self, handle)
            self.experience_targets = loaded_content.experience_targets
            self.experience_data    = loaded_content.experience_data
            self.experience_context = loaded_content.experience_context
            self.experience_outcome = loaded_content.experience_outcome
            self.memories_stored    = loaded_content.memories_stored
            self.experience_timestamps = loaded_content.experience_timestamps
    
    def from_file(self, save_dir, memory_id):
        with open(save_dir + f'classifier_memory_{memory_id}.pkl', 'rb') as handle:
            obj = pickle.load(handle)
        self.experience_targets = obj.experience_targets
        self.experience_data    = obj.experience_data
        self.experience_context = obj.experience_context
        self.experience_outcome = obj.experience_outcome
        self.memories_stored    = obj.memories_stored
        self.experience_timestamps = obj.experience_timestamps


def adapt_manager(save_dir, online_classifier, config):
    logging.basicConfig(filename=Path(save_dir, "adaptmanager.log"),
                        filemode='w',
                        encoding="utf-8",
                        level=logging.INFO)
    # this is where we receive commands from the memoryManager
    # in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # in_sock.bind(("localhost", in_port))

    # this is where we write commands to the memoryManger
    # managers only own their input sockets
    # out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # out_sock.bind(("localhost", out_port))

    # shared_memory_items = online_classifier.smi
    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for item in config.shared_memory_items:
        smm.find_variable(*item)
    # smm.find_variable('memory', (1, 1), np.dtype('U10'), Lock())
    # smm.find_variable('adapt_flag', (1, 1), np.int32, Lock())
    emg_predictor = online_classifier.predictor

    # initialize the memomry
    memory = Memory()
    memory_id = 0

    # initial time
    start_time = time.perf_counter()

    # variables to save and stuff
    adapt_round = 0
    
    time.sleep(3)
    done = False
    
    while time.perf_counter() - start_time < config.game_time:
        try:
            data = smm.get_variable('memory_update_flag')
            if data == WROTE:
                # we were signalled we have data to load
                # append this data to our memory
                t1 = time.perf_counter()
                new_memory = Memory()
                new_memory.from_file(save_dir, memory_id)
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
            else:
                # abstract decoders/fake abstract decoder/sgt
                if config.adaptation:
                    t1 = time.perf_counter()
                    emg_predictor.model.adapt(memory)
                    del_t = time.perf_counter() - t1
                    logging.info(f"ADAPTMANAGER: ADAPTED - round {adapt_round}; \tADAPT TIME: {del_t:.2f}s")
                    
                    with open(online_classifier.options['file_path'] +  'mdl' + str(adapt_round) + '.pkl', 'wb') as handle:
                        pickle.dump(emg_predictor, handle)

                    smm.modify_variable("adapt_flag", lambda x: adapt_round)
                    print(f"Adapted {adapt_round} times")
                    adapt_round += 1
                else:
                    t1 = time.perf_counter()
                    time.sleep(5)
                    del_t = time.perf_counter() - t1
                    logging.info(f"ADAPTMANAGER: WAITING - round {adapt_round}; \tWAIT TIME: {del_t:.2f}s")
                    adapt_round += 1
            
            # signal to the memory manager we are idle and waiting for data
            smm.modify_variable('memory_update_flag', lambda _: WAITING)
            logging.info("ADAPTMANAGER: WAITING FOR DATA")
            time.sleep(0.5)
        except:
            logging.error("ADAPTMANAGER: "+traceback.format_exc())
    else:
        print("AdaptManager Finished!")
        memory.write(save_dir, 1000)



def memory_manager(save_dir, shared_memory_items):
    logging.basicConfig(filename=Path(save_dir, "memorymanager.log"),
                        filemode='w',
                        encoding="utf-8",
                        level=logging.INFO)

    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for item in smi:
        smm.find_variable(*item)

    # initialize the memory
    memory = Memory()

    start_time = time.perf_counter()

    num_written = 0
    total_samples_written = 0
    total_samples_unfound = 0
    done=False

    while not done:
        try:
            environment_data = smm.get_variable('environment_output')[0]

            # May need to handle receiving no data...
            # Previous code also didn't allow both to happen in one loop
                
            if np.all(environment_data == DONE_TASK):
                done = True
                del_t = time.perf_counter() - start_time
                logging.info(f"MEMORYMANAGER: GOT DONE FLAG AT {del_t:.2f}s")
            else:
                result = make_pseudo_labels(environment_data, smm)
                if result is None:
                    total_samples_unfound += 1
                    continue
                adaptation_data, adaptation_labels, adaptation_direction, adaptation_type, timestamp = result 
                if (adaptation_data.shape[0]) != (adaptation_labels.shape[0]):
                    continue
                memory.add_memories(adaptation_data, adaptation_labels, adaptation_direction,adaptation_type, timestamp)

            memory_data = smm.get_variable('memory_update_flag')

            if memory_data == WAITING:
                # write memory to file
                if len(memory):# never write an empty memory
                    t1 = time.perf_counter()
                    memory.write(save_dir, num_written)
                    del_t = time.perf_counter() - t1
                    logging.info(f"MEMORYMANAGER: WROTE FILE: {num_written},\t lines:{len(memory)},\t unfound: {total_samples_unfound},\t WRITE TIME: {del_t:.2f}s")
                    num_written += 1
                    memory = Memory()
                    # in_sock.sendto("WROTE".encode("utf-8"), ("localhost", out_port))
                    smm.modify_variable('memory', lambda _: WROTE)    # tell adapt manager that it has new data
        except:
            logging.error("MEMORYMANAGER: "+traceback.format_exc())
    

def make_pseudo_labels(environment_data, smm):
    timestamp = environment_data[0]
    optimal_direction = environment_data[1:]

    # find the features: data - timestamps, <- features ->
    feature_data = smm.get_variable("model_input")
    feature_data_index = np.where(feature_data[:,0] == timestamp)
    features = torch.tensor(feature_data[feature_data_index,1:].squeeze())
    if len(feature_data_index[0]) == 0:
        return None
    
    # find the predictions info: 
    prediction_data = smm.get_variable("model_output")
    prediction_data_index = np.where(prediction_data[:,0] == timestamp)
    prediction = prediction_data[prediction_data_index, 1:]

    # Designed this thinking we'd have 1 prediction here, but that doesn't seem to be the case so far
    # NOTE: Timestamps aren't matching up for some reason, so we stop getting here...
    outcomes = torch.tensor(['P' if np.sign(x) == np.sign(y) else 'N' for x, y in zip(prediction, optimal_direction)])
    # TODO: need to handle case where we're in the target...

    adaptation_data      = []
    adaptation_data.append(features)
    adaptation_direction = []
    adaptation_direction.append(optimal_direction)
    adaptation_outcome   = []
    adaptation_outcome.append(outcomes)
    timestamp = [timestamp]


    # Project prediction onto optimal direction
    adaptation_labels = (torch.dot(prediction, optimal_direction) / torch.dot(optimal_direction, optimal_direction)) * optimal_direction
    positive_mask = torch.where(outcomes == 'P')

    if sum(positive_mask) == 1:
        # One component was correct, so we normalize it to the correct component
        correct_component = adaptation_labels[positive_mask]
        desired_value = prediction[positive_mask]
        adaptation_labels = adaptation_labels * (desired_value / correct_component)


    # adaptation_labels = torch.vstack(adaptation_labels).type(torch.float32)
    # adaptation_labels = adaptation_labels / np.linalg.norm(adaptation_labels)  # get unit vector for labels
    # For labels I think we have 2 choices:
    # 1. Anchor based on the seeded data (or seeded + data we add). So we say the highest MAV (probably within that DOF?) is 100% and normalize by that. The issue here is finding the proportional control for each DOF and combinations.
    # 2. Assume they move quicker the further they are from the target. What do we normalize by? The screen size? Some function of the target radius and initial distance from cursor?
    adaptation_data = torch.vstack(adaptation_data).type(torch.float32)
    adaptation_direction = np.array(adaptation_direction)
    adaptation_outcome   = np.array(adaptation_outcome)
    return adaptation_data, adaptation_labels, adaptation_direction, adaptation_outcome, timestamp
