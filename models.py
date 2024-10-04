import pickle
import libemg
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch import optim
import numpy as np
import random
from config import Config
import time
config = Config()



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
    def __init__(self, input_shape):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.net = None

        self.foreground_device = "cpu"
        self.train_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.setup_net()
        self.setup_optimizer()

        self.batch_size = config.batch_size
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
        self.optimizer_classifier = optim.Adam(self.net.parameters(), lr=config.learning_rate)
        if config.loss_function == "MSELoss":
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


    def fit(self, epochs, shuffle_every_epoch=True, memory=None):
        self.net.to(self.train_device)
        num_batch = len(memory) // self.batch_size
        losses = []
        for e in range(epochs):
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
                with open(f"Data/subject{config.subjectID}/{config.stage}/trial_{config.model}/loss.csv", 'a') as f:
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
                lower_t[c] = config.WENG_SPEED_MIN
                upper_t[c] = config.WENG_SPEED_MAX
            else:
                sorted, _ = torch.sort(PC_Vals[valid_ids])
                lower_t[c] = sorted[int(len(sorted)*config.lower_PC_percentile)].item()
                upper_t[c] = sorted[int(len(sorted)*config.upper_PC_percentile)].item()
        return lower_t, upper_t

        
    def adapt(self, memory):
        self.net.to(self.train_device)
        self.train()
        self.fit(epochs=config.adaptation_epochs, memory=memory)
        if config.visualize_training:
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

            plt.savefig(f"Figures/Animation/img_{str(self.frames_saved).zfill(3)}_S{config.subjectID}_T{config.trial}")
            plt.close()

        create_frame(self.frames_saved)
        self.frames_saved += 1



import pickle
def setup_classifier(odh,
                    save_dir,
                    smi):
    if config.stage == "fitts":
        model_to_load = f"Data/subject{config.subjectID}/sgt/trial_{config.model}/sgt_mdl.pkl"
    with open(model_to_load, 'rb') as handle:
        loaded_mdl = pickle.load(handle)

   
    # offline_classifier.__setattr__("feature_params", loaded_mdl.feature_params)
    feature_list = config.features


    if smi is None:
        smm = False
    else:
        smm = True
    classifier = libemg.emg_predictor.OnlineEMGRegressor(offline_regressor=loaded_mdl,
                                                            window_size=config.window_length,
                                                            window_increment=config.window_increment,
                                                            online_data_handler=odh,
                                                            features=feature_list,
                                                            file_path = save_dir,
                                                            file=True,
                                                            smm=smm,
                                                            smm_items=smi,
                                                            std_out=False)
    classifier.predictor.model.net.eval()
    classifier.run(block=False)
    return classifier
