from pathlib import Path
import random
import pickle
import math

import torch
import numpy as np
import libemg


WAITING = 0
WROTE = 1
DONE_TASK = -10
ADAPTATION_TIME = 250   # seconds
VALIDATION_TIME = 300   # seconds
TARGET_RADIUS = 40  # pixels
ISOFITTS_RADIUS = 275   # pixels
TIMEOUT = 30


class AdaptationFitts(libemg.environments.fitts.ISOFitts):
    def __init__(self, shared_memory_items, save_file: str, adapt: bool, mapping: str):
        controller = libemg.environments.controllers.RegressorController()
        game_time = ADAPTATION_TIME if adapt else VALIDATION_TIME
        config = libemg.environments.fitts.FittsConfig(
            num_trials=2000,    # set to high number so task stops based on time instead of trials
            dwell_time=2.0,
            save_file=save_file,
            fps=60,
            width=1500,
            height=750,
            target_radius=TARGET_RADIUS,
            game_time=game_time,
            mapping=mapping,
            timeout=TIMEOUT
        )
        super().__init__(controller, config, num_targets=8, target_distance_radius=ISOFITTS_RADIUS)
        self.adapt = adapt
        self.smm = libemg.shared_memory_manager.SharedMemoryManager()
        for sm_item in shared_memory_items:
            self.smm.create_variable(*sm_item)

    def _log(self, label, timestamp):
        # Write to log_dictionary
        super()._log(label, timestamp)

        if not self.adapt:
            return

        # Want to send the timestamp and the optimal direction
        optimal_direction = np.array(self.log_dictionary['goal_target'][-1]) - np.array(self.log_dictionary['cursor_position'][-1])
        optimal_direction[1] *= -1  # multiply by -1 b/c pygame origin is top left, so a lower target has a higher y value
        output = np.array([timestamp, optimal_direction[0], optimal_direction[1]], dtype=np.double)
        self.smm.modify_variable('environment_output', lambda x: np.vstack((output, x))[:x.shape[0]])  # ensure we don't take more than original array size
        if self.smm.get_variable('memory_update_flag')[0, 0] == DONE_TASK:
            self.done = True


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
        
    def add_memories(self, experience_data, experience_targets, experience_context = None, experience_outcome = None, experience_timestamps = None):
        # Add information from SGT data (if needed)
        if experience_context is None:
            experience_context = np.full_like(experience_targets, -2000)
        if experience_outcome is None:
            experience_outcome = np.full_like(experience_targets, 'N', dtype=str)
        if experience_timestamps is None:
            experience_timestamps = [-1 for _ in range(experience_targets.shape[0])]

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


def project_prediction(prediction, optimal_direction):
    return (np.dot(prediction, optimal_direction) / np.dot(optimal_direction, optimal_direction)) * optimal_direction


def distance_to_proportional_control(distance):
    # NOTE: These mappings were decided based on piloting
    result = np.sqrt((distance - TARGET_RADIUS) / ISOFITTS_RADIUS)   # normalizing by half of distance between targets
    # result = 0.2 / (1 + np.exp(-0.05 * (np.linalg.norm(optimal_direction) - 250)))    # sigmoid
    return min(1, result)   # bound result to 1


def assign_ciil_label(prediction, optimal_direction, outcomes):
    positive_mask = outcomes == 'P'
    num_positive_components = positive_mask.sum()
    # deadband_threshold = 0.1
    # below_threshold = np.linalg.norm(prediction) <= deadband_threshold
    # outside_expected_range = distance_to_proportional_control(optimal_direction) > deadband_threshold
    buffer = 1.5 * TARGET_RADIUS
    outside_expected_range = np.linalg.norm(optimal_direction) > buffer
    below_threshold = np.linalg.norm(prediction) < distance_to_proportional_control(buffer)
    # print(below_threshold, outside_expected_range, distance_to_proportional_control(buffer), buffer)

    # TODO: modify so it's based on EMG activity, not the prediction (similar to active threshold with classification but use SGT NM data)
    # TODO: Another question: should we be doing a deadband for the SGT models or no?
    if num_positive_components == 0 or (below_threshold and outside_expected_range):
        # Wrong quadrant or no motion when far from target - ignore this
        adaptation_labels = None
    elif num_positive_components == 2:
        # Correct quadrant - point prediction to optimal direction (later normalized to correct magnitude)
        adaptation_labels = np.copy(optimal_direction)
    elif num_positive_components == 1:
        # Off quadrant - one component is correct
        adaptation_labels = np.zeros_like(prediction)
        adaptation_labels[positive_mask] = np.sign(prediction[positive_mask])
        adaptation_labels[~positive_mask] = 0.
    else:
        raise ValueError(f"Unexpected value when assigning CIIL label. Got: {prediction}, {optimal_direction}, {outcomes}.")

    return adaptation_labels


def make_pseudo_labels(environment_data, smm, approach):
    timestamp = environment_data[0]
    optimal_direction = environment_data[1:]

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

    if np.linalg.norm(optimal_direction) < TARGET_RADIUS:
        # In the target
        adaptation_labels = np.array([0, 0])
        outcomes = ['N', 'N']
        # NOTE: This approach works, but it's a bit naive.
        # We're providing it data that is 0, but the user likely doesn't stop right on the target's edge. I think this is causing some drift.
        # Then once it drifts, you often try to approach the target from a certain angle, so it'll drift into the target. This is bad too b/c you aren't moving, but you're telling the model you should be going towards the target.
        # We could probably do something else naive and say that the user is comfortable giving half the radius as a buffer, but that's kind of random.
        
    else:
        # outcomes = np.array(['P' if np.sign(prediction_component) == np.sign(direction_component) and np.abs(direction_component) > TARGET_RADIUS else 'N'
        #                      for prediction_component, direction_component in zip(prediction, optimal_direction)])
        # outcomes = np.array(['P' if np.sign(x) == np.sign(y) else 'N' for x, y in zip(prediction, optimal_direction)])

        outcomes = []
        for prediction_component, direction_component in zip(prediction, optimal_direction):
            correct_direction = np.sign(prediction_component) == np.sign(direction_component)
            in_target_runway = np.abs(direction_component) <= TARGET_RADIUS
            outcome = 'P' if correct_direction and not in_target_runway else 'N'
            outcomes.append(outcome)
        outcomes = np.array(outcomes)
        if approach == 'ciil':
            adaptation_labels = assign_ciil_label(prediction, optimal_direction, outcomes)
        elif approach == 'oracle':
            adaptation_labels = np.copy(optimal_direction)
        else:
            raise ValueError(f"Unexpected value for approach. Got: {approach}.")

        if adaptation_labels is not None:
            # Normalize to correct magnitude - p = inf b/c (1, 1) should move the same speed as (1, 0)
            adaptation_labels *= distance_to_proportional_control(np.linalg.norm(optimal_direction)) / np.linalg.norm(adaptation_labels, ord=np.inf)

    # print(optimal_direction, outcomes, prediction, adaptation_labels)
    timestamp = [timestamp]
    adaptation_labels = torch.from_numpy(adaptation_labels).type(torch.float32).unsqueeze(0)
    adaptation_data = torch.tensor(features).type(torch.float32).unsqueeze(0)
    adaptation_direction = np.expand_dims(np.array(optimal_direction), axis=0)
    adaptation_outcome   = np.expand_dims(np.array(outcomes), axis=0)
    return adaptation_data, adaptation_labels, adaptation_direction, adaptation_outcome, timestamp
