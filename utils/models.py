import time
import random

import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class MLP(nn.Module):
    def __init__(self, input_shape, batch_size, lr, loss_type, loss_file):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.learning_rate = lr
        self.loss_type = loss_type
        self.loss_file = loss_file

        self.foreground_device = "cpu"
        self.train_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.frames_saved = 0

        self.setup_net()
        self.setup_optimizer()


    def setup_net(self):
        self.net = nn.Sequential(
            nn.Linear(self.input_shape, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10,2)
        )
    
    def setup_optimizer(self):
        # set optimizer
        self.optimizer_classifier = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        if self.loss_type == "MSELoss":
            self.loss_function = nn.MSELoss()
        elif self.loss_type == 'L1':
            self.loss_function = nn.L1Loss()
        else:
            raise ValueError(f"Unexpected value for loss_function. Got: {self.loss_function}.")
        
        
    def forward(self,x):
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        x.requires_grad=False
        return self.net(x)
    
    def predict(self, data):
        if type(data) == np.ndarray:
            data = torch.tensor(data, dtype=torch.float32)
        data = data.to("cpu").clone()
        output = self.forward(data)
        return output.detach().numpy()


    def fit(self, num_epochs, shuffle_every_epoch=True, memory=None):
        self.net.to(self.train_device)
        num_batch = len(memory) // self.batch_size
        losses = []
        for e in range(num_epochs):
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
                with open(self.loss_file, 'a') as f:
                    f.writelines([str(t) + "," + str(i) + "\n" for i in loss])
        self.net.to(self.foreground_device)

    def adapt(self, memory, num_epochs, filename = None):
        self.net.to(self.train_device)
        self.train()
        self.fit(num_epochs, memory=memory)
        if filename is not None:
            self.visualize(memory=memory, filename=filename)
        self.net.to(self.foreground_device)
        self.eval()

    def visualize(self, memory, filename):
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

            plt.savefig(filename)
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
