import random
import pickle
import logging
import time
from multiprocessing import Lock
import traceback

import torch
import numpy as np
import libemg


class AdaptationIsoFitts(libemg.environments.isofitts.IsoFitts):
    def __init__(self, controller: libemg.environments.controllers.Controller, prediction_map: dict | None = None, num_circles: int = 30, num_trials: int = 15, dwell_time: float = 3, timeout: float = 30, velocity: float = 25, save_file: str | None = None, width: int = 1250, height: int = 750, fps: int = 60, proportional_control: bool = True):
        super().__init__(controller, prediction_map, num_circles, num_trials, dwell_time, timeout, velocity, save_file, width, height, fps, proportional_control)
        self.smm = libemg.shared_memory_manager.SharedMemoryManager()
        self.smm.find_variable('environment_output', (100, 3), np.float32, Lock())

    def _log(self, label, timestamp):
        # Write to log_dictionary
        super()._log(label, timestamp)

        # Want to send the timestamp and the optimal direction
        optimal_direction = np.array(self.log_dictionary['goal_circle'][-1]) - np.array(self.log_dictionary['cursor_position'][-1])
        optimal_direction[1] *= -1  # multiply by -1 b/c pygame origin is top left, so a lower target has a higher y value
        message = f"{timestamp} {optimal_direction[0]} {optimal_direction[1]}"
        self.smm.modify_variable('environment_output', lambda _: message)


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
        # What is the id of the experience
        self.experience_ids        = []
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
                self.experience_ids        = other_memory.experience_ids
                self.experience_outcome    = other_memory.experience_outcome
                self.experience_timestamps = other_memory.experience_timestamps
                self.memories_stored       = other_memory.memories_stored
            else:
                self.experience_targets = torch.cat((self.experience_targets,other_memory.experience_targets))
                self.experience_data = torch.vstack((self.experience_data,other_memory.experience_data))
                self.experience_context = np.concatenate((self.experience_context,other_memory.experience_context))
                self.experience_ids.extend(list(range(self.memories_stored, self.memories_stored + other_memory.memories_stored)))
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
                self.experience_ids     = list(range(len(experience_targets)))
                self.experience_outcome = experience_outcome
                self.experience_timestamps = experience_timestamps
                self.memories_stored    += len(experience_targets)
            else:
                self.experience_targets = torch.cat((self.experience_targets,experience_targets))
                self.experience_data = torch.vstack((self.experience_data,experience_data))
                self.experience_context = np.concatenate((self.experience_context,experience_context))
                self.experience_ids.extend(list(range(self.memories_stored, self.memories_stored + len(experience_targets))))
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
            self.experience_ids     = [self.experience_ids[i] for i in indices]
            # SGT does not have these fields
            if len(self.experience_context):
                self.experience_context = self.experience_context[indices]
                self.experience_outcome = [self.experience_outcome[i] for i in indices]
                self.experience_timestamps = [self.experience_timestamps[i] for i in indices]

        
    def unshuffle(self):
        unshuffle_ids = [i[0] for i in sorted(enumerate(self.experience_ids), key=lambda x:x[1])]
        if len(self):
            self.experience_targets = self.experience_targets[unshuffle_ids]
            self.experience_data    = self.experience_data[unshuffle_ids]
            # SGT does not have these fields
            if len(self.experience_context):
                self.experience_context = self.experience_context[unshuffle_ids]
                self.experience_outcome = [self.experience_outcome[i] for i in unshuffle_ids]
                self.experience_ids     = [self.experience_ids[i] for i in unshuffle_ids]
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
            self.experience_ids     = loaded_content.experience_ids
            self.memories_stored    = loaded_content.memories_stored
            self.experience_timestamps = loaded_content.experience_timestamps
    
    def from_file(self, save_dir, memory_id):
        with open(save_dir + f'classifier_memory_{memory_id}.pkl', 'rb') as handle:
            obj = pickle.load(handle)
        self.experience_targets = obj.experience_targets
        self.experience_data    = obj.experience_data
        self.experience_context = obj.experience_context
        self.experience_outcome = obj.experience_outcome
        self.experience_ids     = obj.experience_ids
        self.memories_stored    = obj.memories_stored
        self.experience_timestamps = obj.experience_timestamps


def adapt_manager(adapt, smi, save_dir, offline_predictor):
    logging.basicConfig(filename=save_dir + "adaptmanager.log",
                        filemode='w',
                        encoding="utf-8",
                        level=logging.INFO)
    
    
    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for item in smi:
        smm.find_variable(*item)


    # initialize the memomry
    memory = Memory()
    memory_id = 0

    # initial time
    start_time = time.perf_counter()

    # variables to save and stuff
    adapt_round = 0
    
    time.sleep(3)
    done = False
    
    while not done:
            # see what we have available:
                
            # 1. Check if we have an empty memory buffer -- if so time.sleep and continue
            # logging.info("NO MEMORIES -- SKIPPED TRAINING")
            # t1 = time.perf_counter()
            # time.sleep(3)
            # del_t = time.perf_counter() - t1
            # logging.info(f"ADAPTMANAGER: WAITING - round {adapt_round}; \tWAIT TIME: {del_t:.2f}s")
            

            # 2. Check if we have memory to load, load it
            # t1 = time.perf_counter()
            # new_memory = Memory()
            # new_memory.from_file(save_dir, memory_id)
            # print(f"Loaded {memory_id} memory")
            # memory += new_memory
            # del_t = time.perf_counter() - t1
            # memory_id += 1
            # logging.info(f"ADAPTMANAGER: ADDED MEMORIES, \tCURRENT SIZE: {len(memory)}; \tLOAD TIME: {del_t:.2f}s")
                
            # 3. Perform incremental learning step
            # - adapt
            # logging.info(f"ADAPTMANAGER: ADAPTED - round {adapt_round}; \tADAPT TIME: {del_t:.2f}s")
            # with open(online_classifier.options['file_path'] +  'mdl' + str(adapt_round) + '.pkl', 'wb') as handle:
                #     pickle.dump(emg_classifier, handle)

            # smm.modify_variable("adapt_flag", lambda x: adapt_round)
            # print(f"Adapted {adapt_round} times")
            # adapt_round += 1
                
            # 4. Check if done
            # done = True
            
            
        # print("AdaptManager Finished!")
        # memory.write(save_dir, 1000)
        pass



def memory_manager(smi, save_dir):
    logging.basicConfig(filename=save_dir + "memorymanager.log",
                        filemode='w',
                        encoding="utf-8",
                        level=logging.INFO)

    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for item in smi:
        smm.find_variable(*item)

    # this is where we send out commands to classifier
    # managers only own their input sockets
    # out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # out_sock.bind(("localhost", out_port))

    # initialize the memory
    memory = Memory()

    start_time = time.perf_counter()

    num_written = 0
    total_samples_written = 0
    total_samples_unfound = 0
    waiting_flag = 0
    done=False

    while not done:
        try:
            # see what we have available:
            ready_to_read, ready_to_write, in_error = \
                select.select([in_sock, unity_sock], [], [],0)
            for sock in ready_to_read:
                received_data, _ = sock.recvfrom(512)
                data = received_data.decode("utf-8")
                
                if sock == unity_sock:
                    if data == "DONE":
                        done = True
                        del_t = time.perf_counter() - start_time
                        logging.info(f"MEMORYMANAGER: GOT DONE FLAG AT {del_t:.2f}s")
                    else:
                        result = decode_unity(data, smm, negative_method)
                        if result == None:
                            total_samples_unfound += 1
                            continue
                        adaptation_data, adaptation_labels, adaptation_direction, adaptation_type, timestamp = result 
                        if (adaptation_data.shape[0]) != (adaptation_labels.shape[0]):
                            continue
                        memory.add_memories(adaptation_data, adaptation_labels, adaptation_direction,adaptation_type, timestamp)
                    # print(memory.memories_stored)
                elif sock == in_sock or waiting_flag:
                    if data == "WAITING" or waiting_flag:
                        # write memory to file
                        waiting_flag = 1
                        if len(memory):# never write an empty memory
                            t1 = time.perf_counter()
                            memory.write(save_dir, num_written)
                            del_t = time.perf_counter() - t1
                            logging.info(f"MEMORYMANAGER: WROTE FILE: {num_written},\t lines:{len(memory)},\t unfound: {total_samples_unfound},\t WRITE TIME: {del_t:.2f}s")
                            num_written += 1
                            memory = Memory()
                            in_sock.sendto("WROTE".encode("utf-8"), ("localhost", out_port))
                            waiting_flag = 0
        except:
            logging.error("MEMORYMANAGER: "+traceback.format_exc())
    

def decode_unity(packet, smm, negative_method):
    class_map = ["DOWN","UP","NONE","RIGHT","LEFT", "UNUSED"]
    message_parts = packet.split(" ")
    outcome = message_parts[0]
    context = [message_parts[2], message_parts[3]]
    if context == ["UNUSED", "UNUSED"]:
        return None
    timestamp = float(message_parts[1])
    # find the features: data - timestamps, <- features ->
    feature_data = smm.get_variable("classifier_input")
    feature_data_index = np.where(feature_data[:,0] == timestamp)
    features = torch.tensor(feature_data[feature_data_index,1:].squeeze())
    if len(feature_data_index[0]) == 0:
        return None
    
    # find the predictions info: 
    prediction_data = smm.get_variable("classifier_output")
    prediction_data_index = np.where(prediction_data[:,0] == timestamp)
    prediction = prediction_data[prediction_data_index,1]
    prediction_conf = prediction_data[prediction_data_index,2]

    # PC = feature_file[idx,2]
    expected_direction = [class_map.index(context[0]),class_map.index(context[1])]
     # remove all "UNUSED"

    adaptation_labels    = []
    adaptation_data      = []
    adaptation_data.append(features)
    adaptation_direction = []
    adaptation_direction.append(expected_direction)
    adaptation_outcome   = []
    adaptation_outcome.append(outcome)
    timestamp = [timestamp]

    one_hot_matrix = torch.eye(5)

    # when going down is closer to the planet BUT you're running into a wall,
    # unused is used.
    # remove it here so its still logged in adaptation_outcome, but not as a label/target.
    expected_direction = [i for i in expected_direction if i != 5]
    if outcome == "P":
        # when its positive context, we make the adaptation target completely w/ 
        try:
            adaptation_labels.append(one_hot_matrix[int(prediction),:])
        except Exception as e:
            return None
    elif outcome == "N":
        if negative_method == "mixed":
            mixed_label = torch.zeros(5)
            for o in expected_direction:
                mixed_label += one_hot_matrix[o,:]/len(expected_direction)
            adaptation_labels.append(mixed_label)

    adaptation_labels = torch.vstack(adaptation_labels).type(torch.float32)
    adaptation_data = torch.vstack(adaptation_data).type(torch.float32)
    adaptation_direction = np.array(adaptation_direction)
    adaptation_outcome   = np.array(adaptation_outcome)
    return adaptation_data, adaptation_labels, adaptation_direction, adaptation_outcome, timestamp
