from copy import deepcopy

import libemg
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import Dataset

from emg_regression.parsing import write_pickle


class DataPreparationPipeline:
    def __init__(self, filter, window_size, window_inc, feature_list = None, scale = False, augmentations = None, sequence_length = 0):
        self.filter = filter
        self.window_size = window_size
        self.window_inc = window_inc
        self.feature_list = feature_list
        self.scaler = StandardScaler() if scale else None
        self.augmentations = augmentations
        self.sequence_length = sequence_length
        self.input_size = None
        self.num_dofs = None

    def _get_emg_augmentations(self):
        if self.augmentations is None:
            return self.augmentations

        return [augmentation for augmentation in self.augmentations if isinstance(augmentation, MultiDOFMixup)]

    def _get_feature_augmentations(self):
        emg_augmentations = self._get_emg_augmentations()

        if self.augmentations is None or emg_augmentations is None:
            return self.augmentations

        return [augmentation for augmentation in self.augmentations if augmentation not in emg_augmentations]
        

    def _extract_inputs(self, odh):
        self.filter.filter(odh)

        # Parse
        grab_last_sample = lambda x: x[-1]  # can replace this with 'last_sample' once changes get merged in
        windows, metadata = odh.parse_windows(self.window_size, self.window_inc, metadata_operations=dict(labels=grab_last_sample))
        labels = metadata['labels']

        emg_augmentations = self._get_emg_augmentations()
        if emg_augmentations is not None and not self.is_fitted():
            # Only apply augmentations to training data
            augmented_windows = np.copy(windows)
            augmented_labels = np.copy(labels)
            for augmentation in emg_augmentations:
                augmented_windows, augmented_labels = augmentation(augmented_windows, augmented_labels)

            windows = np.concatenate((windows, augmented_windows), axis=0)
            labels = np.concatenate((labels, augmented_labels), axis=0)
            

        # Split into train and validation sets
        if self.feature_list is None:
            inputs = windows
        elif self.feature_list == ['FFT']:
            inputs = []
            for window in windows:
                channel_ffts = []
                for channel in window:
                    channel_ffts.append(np.fft.fft(channel).real) # need to see how setting n changes things (only needed for GPU memory)

                inputs.append(np.concatenate(channel_ffts))

            inputs = np.array(inputs)
        else:
            inputs = libemg.feature_extractor.FeatureExtractor().extract_features(self.feature_list, windows, array=True)
            assert isinstance(inputs, np.ndarray), f"Expected features to be an array. Got: {type(inputs)}."

        return inputs, labels
    
    def is_fitted(self):
        return self.input_size is not None and self.num_dofs is not None

    def fit(self, odh):
        odh_copy = deepcopy(odh)    # use a odh copy so we don't modify the original odh.data property when filtering
        # Reset values
        self.input_size = None
        self.num_dofs = None
        features, labels = self._extract_inputs(odh_copy)
        self.input_size = features.shape[-1]
        self.num_dofs = labels.shape[1]
        if self.scaler is not None:
            self.scaler.fit(features)

    def __call__(self, odh):
        assert self.is_fitted(), 'Please call .fit() before applying the pipeline.'
        odh_copy = deepcopy(odh)
        inputs, labels = self._extract_inputs(odh_copy)
        assert inputs.shape[-1] == self.input_size, f"Unexpected input size based on fitted data. Got: {inputs.shape[-1]}."
        assert labels.shape[-1] == self.num_dofs, f"Unexpected # DOFs based on fitted data. Got: {labels.shape[1]}."

        # Standardize
        if self.scaler is not None:
            inputs = self.scaler.transform(inputs)
            assert isinstance(inputs, np.ndarray), f"Expected inputs as array. Got: {type(inputs)}."

        # TODO: Probably augment the original raw data sample, but only scale based on real data.
        feature_augmentations = self._get_feature_augmentations()
        if feature_augmentations is not None and not self.is_fitted():
            augmented_samples = np.copy(inputs)
            for augmentation in feature_augmentations:
                augmented_samples = augmentation(augmented_samples)

            inputs = np.concatenate((inputs, augmented_samples), axis=0)
            labels = np.concatenate((labels, labels), axis=0)   # labels aren't changed, so line up with augmented samples

        if self.sequence_length > 0:
            assert inputs.ndim == 2, f"Expected 2D input when sequencing. Got: {inputs.ndim}."
            sequences = []
            sequence_labels = []
            for idx in range(inputs.shape[0] - self.sequence_length + 1):
                sequence = inputs[idx:idx + self.sequence_length]
                sequence_labels.append(labels[idx + self.sequence_length - 1])
                sequences.append(sequence)   

            inputs = np.array(sequences)
            labels = np.array(sequence_labels)

        return inputs, labels

    def save(self, filename):
        write_pickle(self, filename)


def transpose_windows(windows):
    # Works for both tensors and numpy arrays
    if windows.ndim == 3:
        axes = (0, 2, 1)
    else:
        axes = (1, 0)
    if isinstance(windows, np.ndarray):
        windows = np.transpose(windows, axes)
    elif isinstance(windows, torch.Tensor):
        windows = torch.permute(windows, axes)
    else:
        raise ValueError('Passed unexpected type to transpose_windows.')
    return windows


def convert_to_heatmap(inputs, shape):
    num_channels = np.prod(shape)
    if inputs.ndim == 1:
        inputs = inputs[None]

    output_shape = (inputs.shape[0], *shape)
    num_feature_types = inputs.shape[1] // num_channels
    start_idx = 0
    heatmap_features = []
    for _ in range(num_feature_types):
        heatmap = inputs[:, start_idx:start_idx + num_channels].reshape(*output_shape)
        heatmap_features.append(heatmap)
        start_idx += num_channels
    heatmap_features = torch.stack(heatmap_features)
    return heatmap_features.transpose(0, 1)   # CNN expects (N, C, H, W)

    
class TransposeWindows:
    def __call__(self, data):
        # Interface needed to work with torchvision.transforms.Compose
        windows, labels = data
        windows = transpose_windows(windows)
        return windows, labels


class ToTensor:
    def __init__(self, dtype = torch.float32, device = 'cpu'):
        self.dtype = dtype
        self.device = device

    def __call__(self, data):
        windows, labels = data
        windows = torch.from_numpy(windows).type(self.dtype).to(self.device)
        labels = torch.from_numpy(labels).type(self.dtype).to(self.device)
        return windows, labels


class ToHeatmap:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        inputs, labels = data
        heatmap_features = convert_to_heatmap(inputs, self.shape)
        return heatmap_features, labels


class Jitter:
    def __init__(self, distribution = 'gaussian'):
        self.distribution = distribution

    def __call__(self, inputs):
        if self.distribution == 'gaussian':
            noise = np.random.randn(*inputs.shape)
        elif self.distribution == 'poisson':
            raise NotImplementedError('Poisson not implemented.')
        else:
            raise ValueError(f"Unexpected value for distribution. Got: {self.distribution}.")

        return inputs + noise

        
class Scale:
    def __call__(self, inputs):
        scaling_factors = np.random.uniform(low=0.8, high=1.2)
        return inputs * scaling_factors


class MultiDOFMixup:
    def __init__(self, p = 0.2, use_alpha = True):
        self.p = p
        self.use_alpha = use_alpha

    def __call__(self, inputs, labels):
        # See this paper for what they did to help with simultaneous contractions: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10547008
        # There's also mixup for continuous targets: https://proceedings.neurips.cc/paper_files/paper/2022/hash/1626be0ab7f3d7b3c639fbfd5951bc40-Abstract-Conference.html
        assert labels.shape[1] == 2, f"Mixup only supported for 2 DOFs, but got {labels.shape[1]}."
        eps = 1e-3
        # TODO: Modify all augmentations to take in + return labels

        dof1_mask = (torch.abs(labels[:, 0]) >= eps) & (torch.abs(labels[:, 1]) < eps)
        dof2_mask = (torch.abs(labels[:, 1]) >= eps) & (torch.abs(labels[:, 0]) < eps)
        active_mask = torch.where(dof1_mask | dof2_mask)[0]

        # Get augment mask (take only the first p% of the active windows)
        augment_mask = torch.zeros(inputs.shape[0], dtype=torch.bool).to(inputs.device)
        augment_mask[active_mask[:int(self.p * inputs.shape[0])]] = True

        if self.use_alpha:
            a = float(torch.rand(1))
            b = 1 - a
        else:
            a = 1
            b = 1

        # For each value in the augment mask, pair it with the opposite DOF from a sample that isn't being augmented
        augment_dof1_mask = torch.where(augment_mask & dof1_mask)[0]
        augment_dof2_mask = torch.where(augment_mask & dof2_mask)[0]
        original_dof1_mask = torch.where(~augment_mask & dof1_mask)[0]
        original_dof2_mask = torch.where(~augment_mask & dof2_mask)[0]

        # Augment DOF 1 values (take min in case one batch is too small and we can't augment as many as desired)
        num_dof1_augmentations = min(augment_dof1_mask.shape[0], original_dof2_mask.shape[0])
        augment_dof1_mask = augment_dof1_mask[:num_dof1_augmentations]
        original_dof2_mask = original_dof2_mask[:num_dof1_augmentations]
        inputs[augment_dof1_mask] = (a * inputs[augment_dof1_mask]) + (b * inputs[original_dof2_mask])
        labels[augment_dof1_mask] = (a * labels[augment_dof1_mask]) + (b * labels[original_dof2_mask])

        # Augment DOF 2 values
        num_dof2_augmentations = min(augment_dof2_mask.shape[0], original_dof1_mask.shape[0])
        augment_dof2_mask = augment_dof2_mask[:num_dof2_augmentations]
        original_dof1_mask = original_dof1_mask[:num_dof2_augmentations]
        inputs[augment_dof2_mask] = (a * inputs[augment_dof2_mask]) + (b * inputs[original_dof1_mask])
        labels[augment_dof2_mask] = (a * labels[augment_dof2_mask]) + (b * labels[original_dof1_mask])

        return inputs, labels


class EMGDataset(Dataset):
    def __init__(self, windows, labels = None, transform = None):
        self.windows = windows
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        windows = self.windows[idx]
        if self.labels is None:
            # Unsupervised
            labels = None
        else:
            labels = np.array(self.labels[idx]) # cast to array b/c if it's 1D it'll return a float
        if self.transform:
            windows, labels = self.transform((windows, labels))
        return windows, labels


def make_dimensionality_reduction_pipeline(reducer):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('dimensionality-reduction', reducer)
    ])

