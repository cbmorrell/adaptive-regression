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


class AdaptationIsoFitts(libemg.environments.fitts.PolarFitts):
    def __init__(self, shared_memory_items, controller: libemg.environments.controllers.Controller, num_trials: int = 15,
                  dwell_time: float = 3, save_file: str | None = None, fps: int = 60):
        super().__init__(controller, num_trials=num_trials, dwell_time=dwell_time, save_file=save_file, fps=fps)
        self.smm = libemg.shared_memory_manager.SharedMemoryManager()
        for sm_item in shared_memory_items:
            self.smm.create_variable(*sm_item)

    def _log(self, label, timestamp):
        # Write to log_dictionary
        super()._log(label, timestamp)

        target = self.log_dictionary['goal_target'][-1]
        target_position = np.array(target[:2])
        cursor_position = np.array(self.log_dictionary['cursor_position'][-1][:2])
        optimal_direction = target_position - cursor_position
        optimal_direction[1] *= -1  # multiply by -1 b/c pygame origin is top left, so a lower target has a higher y value

        # Convert to polar coordinates for PolarFitts
        # move this to Fitts b/c I need to determine if the target is further or closer to center of screen than cursor to determine if radius should be increased or decreased
        center_screen = np.array([self.width // 2, self.height // 2])
        radius_multiplier = -1 if np.linalg.norm(target_position - center_screen) < np.linalg.norm(cursor_position - center_screen) else 1

        optimal_polar_direction = cartesian_to_polar(optimal_direction)
        optimal_polar_direction[0] *= radius_multiplier
        # Probably need to change theta so it's between -pi/2 and pi/2 so that signs are still correct...


        output = np.array([timestamp, optimal_polar_direction[0], optimal_polar_direction[1], target[2]], dtype=np.double)
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


def distance_to_proportional_control(optimal_direction, method = 'sqrt'):
    # Convert back to cartesian since we're dealing with distance to target
    # NOTE: These mappings were decided based on piloting
    # TODO: Need to fix this now that we're doing polar (and not ISO fitts)
    if method == 'sqrt':
        result = np.sqrt(np.linalg.norm(optimal_direction / 400))   # normalizing by half of distance between targets
    elif method == 'sigmoid':
        result = 0.2 / (1 + np.exp(-0.05 * (np.linalg.norm(optimal_direction) - 250)))
    else:
        raise ValueError(f"Unexpected value for method. Got: {method}.")
    return min(1, result)   # bound result to 1


def polar_to_cartesian(p):
    r = p[0]
    theta = p[1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y])


def cartesian_to_polar(p):
    r = np.linalg.norm(p)
    theta = math.atan2(p[1], p[0])
    return np.array([r, theta])


def assign_ciil_label(prediction, optimal_direction, outcomes):
    # optimal_direction[1] = (325 * optimal_direction[1]) / math.pi
    positive_mask = outcomes == 'P'
    num_positive_components = positive_mask.sum()
    if num_positive_components == 2:
        # Correct quadrant - point prediction to optimal direction (later normalized to correct magnitude)
        adaptation_labels = np.copy(optimal_direction)
    elif num_positive_components == 1:
        # Off quadrant - one component is correct
        adaptation_labels = np.zeros_like(prediction)
        adaptation_labels[positive_mask] = np.sign(prediction[positive_mask])
        adaptation_labels[~positive_mask] = 0.
    else:
        # Wrong quadrant - ignore this
        return None

    # Convert polar to labels (based on how we handle predictions in PolarFitts)
    # adaptation_labels[1] = np.interp(adaptation_labels[1], (-math.pi / 2, math.pi / 2), ())
    adaptation_labels[1] = (325 * adaptation_labels[1]) / (math.pi)    # 325=screen radius, 25=VEL

    # Normalize to correct magnitude
    # p = inf b/c (1, 1) should move the same speed as (1, 0)
    adaptation_labels *= distance_to_proportional_control(polar_to_cartesian(optimal_direction)) / np.linalg.norm(adaptation_labels, ord=np.inf)
    print(positive_mask, prediction, adaptation_labels)
    return adaptation_labels


def assign_oracle_label(prediction, optimal_direction):
    adaptation_labels = project_prediction(prediction, optimal_direction)
    adaptation_labels *= distance_to_proportional_control(optimal_direction) / np.linalg.norm(adaptation_labels, ord=np.inf)
    return adaptation_labels


def make_pseudo_labels(environment_data, smm, approach):
    timestamp = environment_data[0]
    optimal_direction = environment_data[1:-1]    # in Polar coordinates
    target_radius = environment_data[-1]

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

    if np.abs(optimal_direction[0]) < target_radius:
        # In the target
        adaptation_labels = np.array([0, 0])
        outcomes = ['N', 'N']
        # NOTE: This approach works, but it's a bit naive.
        # We're providing it data that is 0, but the user likely doesn't stop right on the target's edge. I think this is causing some drift.
        # Then once it drifts, you often try to approach the target from a certain angle, so it'll drift into the target. This is bad too b/c you aren't moving, but you're telling the model you should be going towards the target.
        # We could probably do something else naive and say that the user is comfortable giving half the radius as a buffer, but that's kind of random.
        
    else:
        # I think this works if we're in -pi/2 to pi/2
        # optimal_direction is usually in terms of pixels on screen
        # so I think if I map from screen size to (-pi/2, pi/2) it should work
        outcomes = np.array(['P' if np.sign(x) == np.sign(y) else 'N' for x, y in zip(prediction, optimal_direction)])
        if approach == 'ciil':
            adaptation_labels = assign_ciil_label(prediction, optimal_direction, outcomes)
        elif approach == 'oracle':
            adaptation_labels = assign_oracle_label(prediction, optimal_direction)
        else:
            raise ValueError(f"Unexpected value for approach. Got: {approach}.")

    timestamp = [timestamp]
    adaptation_labels = torch.from_numpy(adaptation_labels).type(torch.float32).unsqueeze(0)
    adaptation_data = torch.tensor(features).type(torch.float32).unsqueeze(0)
    adaptation_direction = np.expand_dims(np.array(optimal_direction), axis=0)
    adaptation_outcome   = np.expand_dims(np.array(outcomes), axis=0)
    return adaptation_data, adaptation_labels, adaptation_direction, adaptation_outcome, timestamp
