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
DONE_TASK = -10


class AdaptationIsoFitts(libemg.environments.isofitts.IsoFitts):
    def __init__(self, shared_memory_items, controller: libemg.environments.controllers.Controller, num_circles: int = 30, num_trials: int = 15, dwell_time: float = 3, save_file: str | None = None,
                  fps: int = 60):
        super().__init__(controller, num_circles=num_circles, num_trials=num_trials, dwell_time=dwell_time, save_file=save_file, fps=fps)
        self.smm = libemg.shared_memory_manager.SharedMemoryManager()
        for sm_item in shared_memory_items:
            self.smm.create_variable(*sm_item)

    def _log(self, label, timestamp):
        # Write to log_dictionary
        super()._log(label, timestamp)

        # Want to send the timestamp and the optimal direction
        optimal_direction = np.array(self.log_dictionary['goal_circle'][-1]) - np.array(self.log_dictionary['cursor_position'][-1])
        optimal_direction[1] *= -1  # multiply by -1 b/c pygame origin is top left, so a lower target has a higher y value
        output = np.array([timestamp, optimal_direction[0], optimal_direction[1]], dtype=np.double)
        self.smm.modify_variable('environment_output', lambda x: np.vstack((output, x))[:x.shape[0]])  # ensure we don't take more than original array size
        if self.smm.get_variable('memory_update_flag')[0, 0] == DONE_TASK:
            self.done = True

    def run(self):
        super().run()


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
        with open(Path(save_dir, f"classifier_memory_{num_written}.pkl"), 'wb') as handle:
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
        with open(Path(save_dir, f"classifier_memory_{memory_id}.pkl"), 'rb') as handle:
            obj = pickle.load(handle)
        self.experience_targets = obj.experience_targets
        self.experience_data    = obj.experience_data
        self.experience_context = obj.experience_context
        self.experience_outcome = obj.experience_outcome
        self.memories_stored    = obj.memories_stored
        self.experience_timestamps = obj.experience_timestamps


def adapt_manager(save_dir, emg_predictor, config):
    logging.basicConfig(filename=Path(save_dir, "adaptmanager.log"),
                        filemode='w',
                        encoding="utf-8",
                        level=logging.INFO)

    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for item in config.shared_memory_items:
        smm.find_variable(*item)

    # initialize the memomry
    memory = Memory()
    # memory = config.offdh_to_memory()
    memory_id = 0
    num_memories = 0

    # initial time
    start_time = time.perf_counter()

    # variables to save and stuff
    adapt_round = 0
    
    time.sleep(3)
    
    while (time.perf_counter() - start_time) < config.game_time:
        try:
            data = smm.get_variable('memory_update_flag')[0, 0]
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
            elif num_memories != len(memory):
                # abstract decoders/fake abstract decoder/sgt
                num_memories = len(memory)
                if config.model_is_adaptive:
                    t1 = time.perf_counter()
                    emg_predictor.model.adapt(memory)
                    del_t = time.perf_counter() - t1
                    logging.info(f"ADAPTMANAGER: ADAPTED - round {adapt_round}; \tADAPT TIME: {del_t:.2f}s")
                    
                    with open(Path(save_dir, 'mdl' + str(adapt_round) + '.pkl'), 'wb') as handle:
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
        smm.modify_variable('memory_update_flag', lambda _: DONE_TASK)
        memory.write(save_dir, 1000)
        with open(Path(save_dir, 'ad_mdl.pkl'), 'wb') as handle:
            pickle.dump(emg_predictor, handle)



def memory_manager(save_dir, shared_memory_items):
    logging.basicConfig(filename=Path(save_dir, "memorymanager.log"),
                        filemode='w',
                        encoding="utf-8",
                        level=logging.INFO)

    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for item in shared_memory_items:
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
            
            result = make_pseudo_labels(environment_data, smm, approach=2)
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
                    memory.write(save_dir, num_written)
                    del_t = time.perf_counter() - t1
                    logging.info(f"MEMORYMANAGER: WROTE FILE: {num_written},\t lines:{len(memory)},\t unfound: {total_samples_unfound},\t WRITE TIME: {del_t:.2f}s")
                    num_written += 1
                    memory = Memory()
                    smm.modify_variable('memory_update_flag', lambda _: WROTE)    # tell adapt manager that it has new data
        except:
            logging.error("MEMORYMANAGER: "+traceback.format_exc())
    print('memory_manager finished!')


def project_prediction(prediction, optimal_direction):
    return (np.dot(prediction, optimal_direction) / np.dot(optimal_direction, optimal_direction)) * optimal_direction


def distance_to_proportional_control(optimal_direction):
    # result = np.sqrt(np.linalg.norm(optimal_direction / 800))
    result = 0.2 + (0.8 / (1 + np.exp(-0.05 * (np.linalg.norm(optimal_direction) - 250))))
    return min(1, result)


def make_pseudo_labels(environment_data, smm, approach):
    timestamp = environment_data[0]
    optimal_direction = environment_data[1:]
    target_radius = 40  # defined in Fitts

    # find the features: data - timestamps, <- features ->
    feature_data = smm.get_variable("model_input")
    feature_data_index = np.where(feature_data[:,0] == timestamp)
    features = np.array(feature_data[feature_data_index,1:].squeeze())
    if len(feature_data_index[0]) == 0:
        return None
    
    # find the predictions info: 
    prediction_data = smm.get_variable("model_output")
    prediction_data_index = np.where(prediction_data[:,0] == timestamp)[0]
    prediction = prediction_data[prediction_data_index, 1:].squeeze()

    in_target = np.linalg.norm(optimal_direction) < target_radius
    outcomes = np.array(['P' if np.sign(x) == np.sign(y) else 'N' for x, y in zip(prediction, optimal_direction)])
    positive_mask = outcomes == 'P'
    num_positive_components = positive_mask.sum()

    if in_target and num_positive_components != 2:
        # In the target, but predicting the wrong direction
        adaptation_labels = np.array([0, 0])
        outcomes = ['N', 'N']
    elif num_positive_components == 2:
        # Correct quadrant
        adaptation_labels = project_prediction(prediction, optimal_direction)
    elif num_positive_components == 1:
        # Off quadrant - one component is correct
        adaptation_labels = np.zeros_like(prediction)
        if approach == 2:
            # adaptation_labels[positive_mask] = np.linalg.norm(prediction) * np.sign(prediction[positive_mask])
            adaptation_labels[positive_mask] = np.sign(prediction[positive_mask])
            adaptation_labels[~positive_mask] = 0.
        else:
            raise ValueError(f"Unexpected value for approach. Got: {approach}.")
    else:
        # Wrong quadrant - ignore this
        return None

    # adaptation_labels *= np.linalg.norm(prediction) / np.linalg.norm(adaptation_labels) # ensure label has the same magnitude as the prediction
    adaptation_labels *= distance_to_proportional_control(optimal_direction) / np.linalg.norm(adaptation_labels)    # ensure label has magnitude based on distance
    print(positive_mask, prediction, adaptation_labels)
    # adaptation_labels = project_prediction(prediction, optimal_direction)

        # if (positive_mask.sum() == 1) and (approach == 2):
        #     # Snap to component
        #     adaptation_labels[positive_mask] = np.linalg.norm(prediction)
        #     adaptation_labels[~positive_mask] = 0.

        # if positive_mask.sum() == 2:
        #     # Apply proportionality
        #     adaptation_labels = project_prediction(prediction, optimal_direction)

        #     # TODO: Propotional control based on distance (determined based on piloting subjects and seeing where subjects slow down)
        # elif positive_mask.sum() == 1:
        #     # Off quadrant
        #     if approach == 1:
        #         # Project prediction onto optimal direction
        #         # adaptation_labels = (torch.dot(prediction, optimal_direction) / torch.dot(optimal_direction, optimal_direction)) * optimal_direction
        #         adaptation_labels = project_prediction(prediction, optimal_direction)

        #         # Removed this idea b/c it will likely mean some users will slow down since every wrong direction you're shrinking the size of the vector
        #         # positive_mask = torch.where(outcomes == 'P')

        #         # if sum(positive_mask) == 1:
        #         #     # One component was correct, so we normalize it to the correct component
        #         #     correct_component = adaptation_labels[positive_mask]
        #         #     desired_value = prediction[positive_mask]
        #         #     adaptation_labels = adaptation_labels * (desired_value / correct_component)

        #     elif approach == 2:
        #         # Snap to component
        #         adaptation_labels = project_prediction(prediction, optimal_direction)
        #         negative_mask = torch.where(outcomes == 'N')[0]
        #         adaptation_labels[negative_mask] = 0.
        #     else:
        #         raise ValueError(f"Unexpected value for approach. Got: {approach}.")
        # else:
        #     # Wrong quadrant - ignore this
        #     return None

    # adaptation_labels = torch.vstack(adaptation_labels).type(torch.float32)
    # adaptation_labels = adaptation_labels / np.linalg.norm(adaptation_labels)  # get unit vector for labels
    # For labels I think we have 2 choices:
    # 1. Anchor based on the seeded data (or seeded + data we add). So we say the highest MAV (probably within that DOF?) is 100% and normalize by that. The issue here is finding the proportional control for each DOF and combinations.
    # 2. Assume they move quicker the further they are from the target. What do we normalize by? The screen size? Some function of the target radius and initial distance from cursor?
    timestamp = [timestamp]
    adaptation_labels = torch.from_numpy(adaptation_labels).type(torch.float32).unsqueeze(0)
    adaptation_data = torch.tensor(features).type(torch.float32).unsqueeze(0)
    adaptation_direction = np.expand_dims(np.array(optimal_direction), axis=0)
    adaptation_outcome   = np.expand_dims(np.array(outcomes), axis=0)
    return adaptation_data, adaptation_labels, adaptation_direction, adaptation_outcome, timestamp
