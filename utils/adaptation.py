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


def adapt_manager(in_port, out_port, save_dir, online_classifier):
    logging.basicConfig(filename=save_dir + "adaptmanager.log",
                        filemode='w',
                        encoding="utf-8",
                        level=logging.INFO)
    # this is where we receive commands from the memoryManager
    in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    in_sock.bind(("localhost", in_port))

    # this is where we write commands to the memoryManger
    # managers only own their input sockets
    # out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # out_sock.bind(("localhost", out_port))
    shared_memory_items = online_classifier.smi
    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for item in shared_memory_items:
        smm.find_variable(*item)

    emg_classifier = online_classifier.classifier

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
            # see what we have available:
            ready_to_read, ready_to_write, in_error = \
                select.select([in_sock], [], [],0)
            # if we have a message on the in_sock get the message
            # this means we have new data to load in from the memory manager
            for sock in ready_to_read:
                received_data, _ = sock.recvfrom(1024)
                data = received_data.decode("utf-8")
                # we were signalled we have data we to load
                if data == "WROTE":
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
                                        
                    if config.relabel_method == "LabelSpreading":
                        if time.perf_counter() - start_time > 120:
                            t_ls = time.perf_counter()
                            negative_memory_index = [i == "N" for i in memory.experience_outcome]
                            labels = memory.experience_targets.argmax(1)
                            labels[negative_memory_index] = -1
                            ls = LabelSpreading(kernel='knn', alpha=0.2, n_neighbors=50)
                            ls.fit(memory.experience_data.numpy(), labels)
                            current_targets = ls.transduction_

                            
                            velocity_metric = torch.mean(memory.experience_data,1)
                            # two component unsupervised GMM
                            gmm = GaussianMixture(n_components=2).fit(velocity_metric.unsqueeze(1))
                            gmm_probs       = gmm.predict_proba(velocity_metric.unsqueeze(1)) 
                            gmm_predictions = np.argmax(gmm_probs,1)
                            lower_cluster = np.argmin(gmm.means_)
                            mask = gmm_predictions == lower_cluster
                            # mask2 = np.max(gmm_probs,1) > 0.95
                            # mask = np.logical_and(mask1, mask2)
                            current_targets[mask] = 2
                            labels = torch.tensor(current_targets, dtype=torch.long)
                            memory.experience_targets = torch.eye(5)[labels]
                            del_t_ls = time.perf_counter() - t_ls
                            logging.info(f"ADAPTMANAGER: LS/GMM - round {adapt_round}; \tLS TIME: {del_t_ls:.2f}s")
                    
                    t1 = time.perf_counter()
                    emg_classifier.classifier.adapt(memory)
                    del_t = time.perf_counter() - t1
                    logging.info(f"ADAPTMANAGER: ADAPTED - round {adapt_round}; \tADAPT TIME: {del_t:.2f}s")
                    
                    if config.adapt_PCs:
                        try:
                            if time.perf_counter() - start_time > 30:
                                t1_pc = time.perf_counter()
                                old_min = emg_classifier.th_min_dic
                                old_max = emg_classifier.th_max_dic
                                new_low, new_high = emg_classifier.classifier.get_pc_thresholds(memory)# make this run on background
                                th_min_dic,th_max_dic = {}, {}
                                for i in range(5):
                                    th_min_dic[i] = new_low[i]*(0.25) + (1-0.25)*old_min[i]
                                    th_max_dic[i] = new_high[i]*(0.25) + (1-0.25)*old_max[i]
                                emg_classifier.__setattr__("th_min_dic", th_min_dic)
                                emg_classifier.__setattr__("th_max_dic", th_max_dic)
                                del_t_pc = time.perf_counter() - t1_pc
                                logging.info(f"ADAPTMANAGER: PC - round {adapt_round}; \tPC TIME: {del_t_pc:.2f}s")

                        except:
                            print("Could not set PCs")
                    with open(online_classifier.options['file_path'] +  'mdl' + str(adapt_round) + '.pkl', 'wb') as handle:
                        pickle.dump(emg_classifier, handle)

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
            in_sock.sendto("WAITING".encode("utf-8"), ("localhost", out_port))
            logging.info("ADAPTMANAGER: WAITING FOR DATA")
            time.sleep(0.5)
        except:
            logging.error("ADAPTMANAGER: "+traceback.format_exc())
    else:
        print("AdaptManager Finished!")
        memory.write(save_dir, 1000)



def memory_manager(in_port, unity_port, out_port, save_dir, negative_method, shared_memory_items):
    logging.basicConfig(filename=save_dir + "memorymanager.log",
                        filemode='w',
                        encoding="utf-8",
                        level=logging.INFO)
    
    # this is where we receive commands from the classifier
    in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    in_sock.bind(("localhost", in_port))

    # this is where we receive context from unity
    unity_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    unity_sock.bind(("localhost", unity_port))

    smm = libemg.shared_memory_manager.SharedMemoryManager()
    for item in shared_memory_items:
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
