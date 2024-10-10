from copy import deepcopy
from typing import Callable, Sequence
import time
import pickle
import random

import numpy as np
import torch
from torch.optim.sgd import SGD
from torch.optim.optimizer import Optimizer
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
import libemg
import matplotlib.pyplot as plt

from utils.logger import Logger
from utils.preprocessing import convert_to_heatmap


class EarlyStopper:
    def __init__(self, patience = 1, delta = 0., delta_type = 'absolute'):
        self.patience = patience
        self.delta = delta
        self.delta_type = delta_type
        self.counter = 0
        self.min_loss = None
        self.best_weights = None

    def _check_relative_delta(self, loss):
        return ((self.min_loss - loss) / self.min_loss) > self.delta

    def _check_absolute_delta(self, loss):
        return (self.min_loss - loss) > self.delta

    def early_stop(self, loss, model):
        if self.delta_type == 'absolute':
            stop_method = self._check_absolute_delta
        elif self.delta_type == 'relative':
            stop_method = self._check_relative_delta
        else:
            raise ValueError(f"Unexpected value for delta_type ({self.delta_type}).")

        if self.min_loss is None:
            self.min_loss = loss
            self.best_weights = deepcopy(model.state_dict())
        elif stop_method(loss):
            # Store best model weights...
            self.min_loss = loss
            self.counter = 0
            self.best_weights = deepcopy(model.state_dict())
        else:
            self.counter += 1

        return self.counter > self.patience


def get_device():
    try:
        device = torch.get_default_device()
    except AttributeError:
        device = 'cpu'
    return device


def train_model(dataloader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: Optimizer, num_epochs: int,
                feature_extractor: nn.Module | None = None, logger: Logger | None = None, log_name: str = '', validation_dataloader: DataLoader | None = None,
                verbose: bool = False, early_stopper: EarlyStopper | None = None, scheduler: optim.lr_scheduler.LRScheduler | None = None):
    device = get_device()
    def calculate_loss(inputs, targets):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if feature_extractor is not None:
            # Extract features
            with torch.no_grad():
                representations = feature_extractor(inputs)
        else:
            representations = inputs
        outputs = model(representations)
        loss = criterion(outputs, targets)
        return loss

    if isinstance(scheduler, ReduceLROnPlateau) and validation_dataloader is None:
        raise ValueError('LR scheduler requires validation dataloader, but None was passed.')

    model.to(device)
    for epoch_idx in range(num_epochs):
        if verbose:
            print(epoch_idx)
        model.train()
        training_loss = 0.
        for inputs, targets in dataloader:
            # Compute prediction and loss
            loss = calculate_loss(inputs, targets)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()

        epoch_training_loss = training_loss / len(dataloader)
        if logger is not None:
            logger.log_value({'epoch': epoch_idx, f"training_loss/{log_name}": epoch_training_loss})

        if validation_dataloader is not None:
            model.eval()
            validation_loss = 0.
            for inputs, targets in validation_dataloader:
                loss = calculate_loss(inputs, targets)
                validation_loss += loss.item()

            epoch_validation_loss = validation_loss / len(validation_dataloader)

            if logger is not None:
                logger.log_value({'epoch': epoch_idx, f"validation_loss/{log_name}": epoch_validation_loss})

            if early_stopper is not None and early_stopper.early_stop(epoch_validation_loss, model):
                print('Early stop condition met.')
                model.load_state_dict(early_stopper.best_weights)
                return

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(epoch_validation_loss)
        elif scheduler is not None:
            scheduler.step()

            
    model.eval()


class ContrastiveBackbone(nn.Module):
    def __init__(self, encoder: nn.Module, criterion: nn.Module, lr: float = 1e-3, projection_network: nn.Module | None = None,
                 early_stopper: EarlyStopper | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection_network = projection_network
        self.criterion = criterion
        self.early_stopper = early_stopper
        self.lr = lr

    def make_optimizer(self, parameters = None):
        if parameters is None:
            parameters = self.parameters()
        return SGD(parameters, lr=self.lr, momentum=0.9, weight_decay=0.5)

    @staticmethod
    def make_lr_scheduler(optimizer):
        return ReduceLROnPlateau(optimizer, mode='min', patience=2)

    def forward(self, x):
        representations = self.encoder(x)
        if self.training and self.projection_network is not None:
            # Only apply projection network during training
            representations = self.projection_network(representations)

        return representations


class PredictionHead(nn.Module):
    def __init__(self, input_size: int, output_size: int, criterion: nn.Module, lr: float = 1e-3, early_stopper: EarlyStopper | None = None):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.criterion = criterion
        self.early_stopper = early_stopper
        self.lr = lr

    def make_optimizer(self, parameters = None):
        if parameters is None:
            parameters = self.parameters()
        return SGD(parameters, lr=self.lr, momentum=0.9, weight_decay=0)

    @staticmethod
    def make_lr_scheduler(optimizer):
        return ReduceLROnPlateau(optimizer, mode='min', patience=10)
        
    def forward(self, x):
        x = self.fc(x)
        return x


class BaselineModel(nn.Module):
    def __init__(self, encoder: nn.Module, prediction_head: PredictionHead):
        super().__init__()
        self.encoder = encoder
        self.prediction_head = prediction_head

    def forward(self, x):
        x = self.encoder(x)
        x = self.prediction_head(x)
        return x

    def fit(self, train_dataloader: DataLoader, num_epochs: int = 100, logger: Logger | None = None, validation_dataloader: DataLoader | None = None):
        optimizer = self.prediction_head.make_optimizer(self.parameters())
        lr_scheduler = self.prediction_head.make_lr_scheduler(optimizer)
        train_model(train_dataloader, self, self.prediction_head.criterion, optimizer, num_epochs,
                    logger=logger, validation_dataloader=validation_dataloader, verbose=True, early_stopper=self.prediction_head.early_stopper,
                    scheduler=lr_scheduler)

    def predict(self, inputs):
        is_numpy = isinstance(inputs, np.ndarray)
        if is_numpy:
            inputs = torch.from_numpy(inputs).type(torch.float32)

        self.eval()
        predictions = self(inputs)
        if is_numpy:
            predictions = predictions.detach().cpu().numpy()
        return predictions


class ContrastiveModel(nn.Module):
    def __init__(self, clr_backbone: ContrastiveBackbone, predictor: PredictionHead) -> None:
        super().__init__()
        self._clr_backbone_copy = deepcopy(clr_backbone)
        self._predictor_copy = deepcopy(predictor)
        self.clr_backbone = clr_backbone
        self.predictor = predictor

    def forward(self, x):
        x = self.clr_backbone(x)
        logits = self.predictor(x)
        return logits

    def _reset_weights(self):
        # Reset weights so you don't start training with pretrained weights
        self.clr_backbone = deepcopy(self._clr_backbone_copy)
        self.predictor = deepcopy(self._predictor_copy)

    def fit(self, train_dataloader: DataLoader, num_epochs: int = 100, logger: Logger | None = None, validation_dataloader: DataLoader | None = None,
            backbone_train_dataloader: DataLoader | None = None, backbone_validation_dataloader: DataLoader | None = None):
        if backbone_train_dataloader is not None or backbone_validation_dataloader is not None:
            assert backbone_train_dataloader is not None and backbone_validation_dataloader is not None, "Only one backbone dataloader was passed. Both should be passed."

        if backbone_train_dataloader is None:
            backbone_train_dataloader = train_dataloader
        if backbone_validation_dataloader is None:
            backbone_validation_dataloader = validation_dataloader

        # Pretrain encoder + projection network
        self._reset_weights()
        backbone_optimizer = self.clr_backbone.make_optimizer()
        backbone_lr_scheduler = self.clr_backbone.make_lr_scheduler(backbone_optimizer)
        train_model(backbone_train_dataloader, self.clr_backbone, self.clr_backbone.criterion, backbone_optimizer, num_epochs, logger=logger, log_name='backbone',
                    verbose=True, validation_dataloader=backbone_validation_dataloader, early_stopper=self.clr_backbone.early_stopper, scheduler=backbone_lr_scheduler)
        self.clr_backbone.requires_grad_(False) # freeze backbone
        self.clr_backbone.eval()

        # Train prediction head
        predictor_optimizer = self.predictor.make_optimizer()
        predictor_lr_scheduler = self.predictor.make_lr_scheduler(predictor_optimizer)
        train_model(train_dataloader, self.predictor, self.predictor.criterion, predictor_optimizer, num_epochs, feature_extractor=self.clr_backbone,
                    logger=logger, log_name='predictor', verbose=True, validation_dataloader=validation_dataloader, early_stopper=self.predictor.early_stopper,
                    scheduler=predictor_lr_scheduler)

    def predict(self, inputs):
        is_numpy = isinstance(inputs, np.ndarray)
        if is_numpy:
            inputs = torch.from_numpy(inputs).type(torch.float32)

        self.eval()
        predictions = self(inputs)
        if is_numpy:
            predictions = predictions.detach().cpu().numpy()
        return predictions


# class MLP(nn.Module):
#     def __init__(self, num_input_channels: int, hidden_channels: Sequence[int], last_layer_activation: bool = False):
#         # torchvision has this as a nn.Sequential, but this makes every field something it tries to iterate through
#         super().__init__()
#         self.input_size = num_input_channels
#         layers = []
#         in_dim = num_input_channels
#         for hidden_dim in hidden_channels[:-1]:
#             layers.append(nn.Linear(in_dim, hidden_dim))
#             layers.append(nn.ReLU())
#             in_dim = hidden_dim

#         # Add final layer separately because it may not have an activation layer
#         layers.append(nn.Linear(in_dim, hidden_channels[-1]))
#         if last_layer_activation:
#             layers.append(nn.ReLU())
#         self.layers = nn.ModuleList(layers)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x


class ProjectionNetwork(nn.Module):
    def __init__(self, num_in_features: int, num_out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(num_in_features, 50)
        self.fc2 = nn.Linear(50, num_out_features)
        self.num_out_features = num_out_features

    def forward(self, x):
        # SimCLR paper says that this should have another linear layer and the output SHOULDN'T be wrapped in a ReLU (output of projection network should be linear according to Shri)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_input_channels: int, hidden_channels: list[int], output_size = 100):
        super().__init__()
        # make HD heatmap and use it
        # compare HTD and wavelet
        # using conv2d in PyTorch
        self.padding = 2
        self.kernel_size = 3
        self.pool_kernel_size = 2
        self.pool_stride = 2

        cnn_blocks = []
        in_channels = num_input_channels
        for channels in hidden_channels:
            block = self._make_cnn_block(in_channels, channels)
            in_channels = channels
            cnn_blocks.append(block)

        self.cnn_blocks = nn.ModuleList(cnn_blocks)
        
        input_height = 4
        input_width = 16
        for block in self.cnn_blocks:
            input_height, input_width = self._calculate_output_size(block, input_height, input_width)
        flattened_size = int(input_height * input_width * hidden_channels[-1])
        self.fc = nn.Linear(flattened_size, output_size)


    def _make_cnn_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        )

    def _calculate_output_size(self, block, input_height, input_width):
        conv_layer = block[0]
        conv_output_height = ((input_height + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0]) + 1
        conv_output_width = ((input_width + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1]) + 1

        pool_layer = block[-1]
        output_height = ((conv_output_height + 2 * pool_layer.padding - pool_layer.padding) / pool_layer.stride)
        output_width = ((conv_output_width + 2 * pool_layer.padding - pool_layer.padding) / pool_layer.stride)
        return int(output_height), int(output_width)

    def forward(self, x):
        x = convert_to_heatmap(x, (4, 16))
        for block in self.cnn_blocks:
            x = block(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_size: int, mlp: MLP, num_lstm_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, mlp.input_size, num_layers=num_lstm_layers, batch_first=True)
        self.mlp = mlp

        
    def forward(self, x):
        # Need to sort out how I'll pass data in. Pass in sequence of features or raw data?
        # Since batch_first=True, the input should be (batch, seq, feature)
        # So it would be (batch, samples, channels) for raw data and (batch, queue of windows, features) for features
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)  # can pass initial hidden/cell states here if persisting states across windows is desired
        x = x[:, 0, :]  # returns the hidden state for each time point, so only use the hidden state from the last time point
        x = self.mlp(x)
        return x



class Transformer(nn.Module):
    def __init__(self, num_features: int, sequence_length: int, output_size = 100, nhead = 4, num_layers = 4, d_model = 100, dim_feedforward = 256):
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding = nn.Linear(num_features, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, sequence_length, d_model), requires_grad=True)  # learnable positional encoding (see https://dl.acm.org/doi/pdf/10.1145/3447548.3467401)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.5, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model * self.sequence_length, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Try with raw data? Or FFT idea from Shri? Or CNN idea from Evan?
        # Expects (batch, sequence, features)
        x = self.embedding(x)
        x = x + self.positional_encoding
        x = self.encoder(x) # will automatically generate a causal mask (to tell model that a future value can't predict a current one)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.relu(x)
        return x



# class RCNN(nn.Module):
#     def __init__(self, num_cnn_blocks: int = 2, num_lstm_layers: int = 1):
#         super().__init__()
#         # make a CNN block that's just a couple convolutional layers then some pooling. then maybe have a 'num_cnn_blocks' parameters
#         self.cnn = self._make_cnn_block()
#         self.lstm = nn.LSTM(x, 100, num_layers=num_lstm_layers, batch_first=True)
#         self.fc = nn.Linear(100, 100)

#     def _make_cnn_block(self):
#         return nn.Sequential(
#             nn.Conv2d(),
#             nn.BatchNorm1d(),
#             nn.ReLU(),
#             nn.AvgPool2d()
#         )

#     def forward(self, x):
#         x = x.reshape(x.shape[0], 4, 16, x.shape[-1])
#         # CNN layers -> expects (N, C, H, W) input
#         # LSTM
#         # FC layers
#         return x


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


from sklearn.decomposition import PCA
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.input_shape = config.input_shape
        self.net = None

        self.foreground_device = "cpu"
        self.train_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.setup_net()
        self.setup_optimizer()

        self.batch_size = config.batch_size
        self.config = config
        self.frames_saved = 0

    def setup_net(self):
        self.net = nn.Sequential(
            nn.Linear(self.input_shape, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10,2),
            nn.Tanh()
        )
    
    def setup_optimizer(self):
        # set optimizer
        self.optimizer_classifier = optim.Adam(self.net.parameters(), lr=self.config.learning_rate)
        if self.config.loss_function == "MSELoss":
            self.loss_function = nn.MSELoss()
        
    def forward(self,x):
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        x.requires_grad=False
        return self.net(x)
    
    def predict(self, data):
        self.predict_proba(data)
        # probs = self.predict_proba(data)
        # return np.array([np.where(p==np.max(p))[0][0] for p in probs])

    def predict_proba(self,data):
        if type(data) == np.ndarray:
            data = torch.tensor(data, dtype=torch.float32)
        data = data.to("cpu").clone()
        output = self.forward(data)
        return output.detach().numpy()


    def fit(self, shuffle_every_epoch=True, memory=None):
        self.net.to(self.train_device)
        num_batch = len(memory) // self.batch_size
        losses = []
        for e in range(self.config.DC_epochs):
            t = time.time()
            if shuffle_every_epoch:
                memory.shuffle()
            loss = []
            batch_start = 0
            if num_batch > 0:
                for b in range(num_batch):
                    batch_end = batch_start + self.batch_size
                    input = memory.experience_data[batch_start:batch_end,:].to(self.train_device)
                    targets = memory.experience_targets[batch_start:batch_end].to(self.train_device)
                    predictions = self.forward(input)
                    loss_value = self.loss_function(predictions, targets)
                    self.optimizer_classifier.zero_grad()
                    loss_value.backward()
                    self.optimizer_classifier.step()
                    loss.append(loss_value.item())
                    batch_start = batch_end
                losses.append(sum(loss)/len(loss))
                with open(f"Data/subject{self.config.subjectID}/{self.config.stage}/trial_{self.config.model}/loss.csv", 'a') as f:
                    f.writelines([str(t) + "," + str(i) + "\n" for i in loss])
        self.net.to(self.foreground_device)

    def get_pc_thresholds(self, memory):
        PC_Vals = memory.experience_data[:,:].mean(axis=1)
        probabilities = self.forward(memory.experience_data)
        probabilities = torch.tensor(probabilities)
        predictions   = torch.argmax(probabilities,1)

        lower_t = {}
        upper_t = {}
        for c in range(5):
            valid_ids = predictions == c
            if sum(valid_ids) == 0:
                lower_t[c] = self.config.WENG_SPEED_MIN
                upper_t[c] = self.config.WENG_SPEED_MAX
            else:
                sorted, _ = torch.sort(PC_Vals[valid_ids])
                lower_t[c] = sorted[int(len(sorted)*self.config.lower_PC_percentile)].item()
                upper_t[c] = sorted[int(len(sorted)*self.config.upper_PC_percentile)].item()
        return lower_t, upper_t
        
    def adapt(self, memory):
        self.net.to(self.train_device)
        self.train()
        self.fit(memory=memory)
        if self.config.visualize_training:
            self.visualize(memory=memory)
        self.net.to(self.foreground_device)
        self.eval()

    def visualize(self, memory):
        predictions = self.forward(memory.experience_data.to(self.train_device))
        predictions = predictions.detach().cpu().numpy()
        def create_frame(t):
            pca = PCA(2)
            transformed_data = pca.fit_transform(memory.experience_data.detach().numpy())
            fig = plt.figure(figsize=(12,6))
            gs = fig.add_gridspec(2,5)
            ax2 = fig.add_subplot(gs[0,0])
            ax3 = fig.add_subplot(gs[0,1])

            ax21 = fig.add_subplot(gs[1,0])
            ax31 = fig.add_subplot(gs[1,1])
            ax41 = fig.add_subplot(gs[1,2])
            ax51 = fig.add_subplot(gs[1,3])
            ax61 = fig.add_subplot(gs[1,4])


            ax2.scatter(transformed_data[:,0], transformed_data[:,1], c=memory.experience_targets[:,0], alpha=0.3, cmap="inferno", vmin=-1, vmax=1,s=2)
            ax2.set_title("SGT - F-E")
            points = ax3.scatter(transformed_data[:,0], transformed_data[:,1], c=memory.experience_targets[:,1], alpha=0.3, cmap="inferno", vmin=-1, vmax=1,s=2)
            ax2.set_title("SGT - UD-RD")
            fig.colorbar(points)

            plt.tight_layout()

            plt.savefig(f"Figures/Animation/img_{str(self.frames_saved).zfill(3)}_S{self.config.subjectID}_T{self.config.trial}")
            plt.close()

        create_frame(self.frames_saved)
        self.frames_saved += 1


def fix_random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
