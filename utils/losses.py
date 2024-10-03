import numbers

import torch
from torch import nn
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import triang
import numpy as np


class LDSLoss(nn.Module):
    # LDS didn't seem to help results much and just adds complexity. Also hard to justify for multiple DOFs (how do you handle weighting of each one?).
    # I could maybe try it when training with no combined DOFs and doing simultaneous motions. Maybe then the weighting of getting the simultaneous motions would be important (since it hasn't seen those in training)? 
    # Could probably get a similar effect with a custom loss that weights simultaneous motions higher. What about a loss that explicitly weights problem areas higher (e.g., low end and combined motions)? Might make other areas worse, but could be worth a shot.
    def __init__(self, labels, criterion, kernel = 'gaussian', ks = 9, sigma = 1, reweight = 'inverse'):
        super().__init__()

        self.criterion = criterion
        self.kernel = kernel
        self.ks = ks
        self.sigma = sigma
        self.reweight = reweight
        assert self.reweight in ['inverse', 'sqrt']
        
        # Code provided by DIR paper: https://github.com/YyzHarry/imbalanced-regression/blob/main/agedb-dir/utils.py
        assert self.kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (self.ks - 1) // 2
        if self.kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            kernel_window = gaussian_filter1d(base_kernel, sigma=self.sigma) / max(gaussian_filter1d(base_kernel, sigma=self.sigma))
        elif self.kernel == 'triang':
            kernel_window = triang(self.ks)
        else:
            laplace = lambda x: np.exp(-abs(x) / self.sigma) / (2. * self.sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

        # Calculate label distribution
        bins = int((labels.max() - labels.min()) / 0.1) # want bin width of ~ 0.1
        hist, bins = np.histogramdd(labels, bins=bins)
        if self.reweight == 'sqrt':
            hist = np.sqrt(hist)
        smoothed_distribution = convolve1d(hist, weights=kernel_window, mode='constant')
        self.smoothed_distribution = torch.tensor(smoothed_distribution)
        self.bins = list(map(torch.tensor, bins))

        kernel_weights = self._get_kernel_weights(labels)
        weights = 1 / kernel_weights
        self.scaling = len(weights) / torch.sum(weights)

        
    def _get_kernel_weights(self, labels):
        kernel_weights = []
        for label in labels:
            bin_mask = []
            for dof_label, dof_bins in zip(label, self.bins):
                bin_idx = torch.where(dof_label < dof_bins)[0]
                if len(bin_idx) == 0:
                    bin_idx = len(dof_bins) - 2
                else:
                    bin_idx = bin_idx[0] - 1

                bin_mask.append(int(bin_idx))
            kernel_weights.append(self.smoothed_distribution[tuple(bin_mask)])

        kernel_weights = torch.tensor(kernel_weights)
        return kernel_weights

    def forward(self, input, target):
        # Reweight based on kernel
        # Not sure if this will work yet... depends on if the encoder is learning to align so it just predicts the mean (not sure how it would)
        # Question is how do you weight each one individually?? Maybe calculate separately for each DOF and then average?
        # I'd assume that this would help with simultaneous contractions if it doesn't help with predicting the mean...
        kernel_weights = self._get_kernel_weights(target)

        # Scaling
        weights = 1 / kernel_weights
        inf_mask = torch.isinf(weights)
        weights[inf_mask] = weights[~inf_mask].max()   # this value wasn't seen in the training distribution - set it to max priority
        weights = weights * self.scaling
        if weights.ndim == 1:
            weights = weights.unsqueeze(1)
        
        loss = self.criterion(input, target)
        loss = loss * weights
        return torch.mean(loss)


class SingleDOFLoss(nn.MSELoss):
    def __init__(self) -> None:
        super().__init__(reduction='none')

    def forward(self, input, target):
        loss = super().forward(input, target)
        return torch.max(loss, dim=1)[0].mean()
        

class RnCLoss(nn.Module):
    def __init__(self, temperature = 0.1, label_distance_metric = 1, feature_similarity_metric = 2):
        # Note about this loss function - you need to standardize data. If you don't then the feature distance may be so large that the exponential will go to 0, making the log produce infinity
        super().__init__()
        self.temperature = temperature
        self.label_distance_metric = label_distance_metric
        self.feature_simiarity_metric = feature_similarity_metric

    def _calculate_label_difference_matrix(self, labels):
        # This is basically just a fancy way of calculating the difference between each label with respect to all others
        # torch.all(label_difference matrix[idx] == (labels[idx] - labels)).item() should be True (and it is b/c I checked it)
        x1 = labels.unsqueeze(1)
        x2 = labels.unsqueeze(0)
        norm_dim = -1
        if isinstance(self.label_distance_metric, numbers.Number):
            label_difference_matrix = (x1 - x2).norm(p=self.label_distance_metric, dim=norm_dim)
        elif self.label_distance_metric == 'cosine':
            label_difference_matrix = torch.tensordot(x1, x2, dims=([1, 2], [0, 2])) / (x1.norm(p=2, dim=norm_dim) * x2.norm(p=2, dim=norm_dim))
        else:
            raise ValueError(f"Unexpected value for label_distance_metric. Got: {self.label_distance_metric}.")
        # I think we should go with p=-1 or 2 cause it'll push simultaneous contractions very far away
        # NOT -1, 2^-1 (0.5). I was reading this image wrong. https://en.wikipedia.org/wiki/Minkowski_distance#:~:text=The%20Minkowski%20distance%20or%20Minkowski,the%20German%20mathematician%20Hermann%20Minkowski.
        # This won't be the case for self-supervised stuff though. The labels for that will be essentially single DOF since it's just neighbouring time points (or augmented view of the same point). If this approach works super well, you could probably do a 3 stage approach where you pretrain on unlabelled data on RnC loss, then fine-tune embedding on labelled data using RnC loss, then tune regression head on labels
        return label_difference_matrix

    def _calculate_feature_similarity_matrix(self, inputs):
        x1 = inputs.unsqueeze(1)
        x2 = inputs.unsqueeze(0)
        difference_matrix = (x1 - x2)
        num_extra_dimensions = difference_matrix.ndim - 2
        norm_dim = [-(idx + 1) for idx in range(num_extra_dimensions)]  # take norm across all dimensions except the first 2

        if isinstance(self.feature_simiarity_metric, numbers.Number):
            feature_similarity_matrix = - (difference_matrix).norm(p=self.feature_simiarity_metric, dim=norm_dim) # taking the negative value because we're looking at negative L2 distance
        elif self.feature_simiarity_metric == 'cosine':
            x1_dim = [1] + norm_dim
            x2_dim = [0] + norm_dim
            feature_similarity_matrix = torch.tensordot(x1, x2, dims=(x1_dim, x2_dim)) / (x1.norm(p=2, dim=norm_dim) * x2.norm(p=2, dim=norm_dim))
        else:
            raise ValueError(f"Unexpected value for feature_similarity_metric. Got: {self.feature_similarity_metric}.")

        return feature_similarity_matrix

    def _remove_diagonal(self, x):
        diagonal_matrix = torch.eye(x.shape[0])
        return x.masked_select((1 - diagonal_matrix).bool()).view(x.shape[0], x.shape[0] - 1)   # reshape so each row is a sample and each column is its value compared to another sample (n - 1 b/c it's different from anchor i.e., j != i)

    def forward(self, inputs, targets):
        label_difference_matrix = self._calculate_label_difference_matrix(targets)
        feature_similarity_matrix = self._calculate_feature_similarity_matrix(inputs).div(self.temperature)
        # Normalize similarity within each sample. We do this so now the sample with the smallest distance has a value of 0. This is also why we use negative distance. If we wanted to use positive, we'd have to take the min, but then exponential would likely get too large.
        feature_similarity_max, _ = torch.max(feature_similarity_matrix, dim=1, keepdim=True)
        feature_similarity_matrix -= feature_similarity_max
        exp_feature_similarity_matrix = feature_similarity_matrix.exp()

        if torch.any(exp_feature_similarity_matrix == 0):
            # If the distance between samples is too large, then the negative exponential can get so small that it is considered 0
            print(f"Detected {torch.sum(exp_feature_similarity_matrix == 0)} 0 values in feature similarity matrix. Adding epsilon to avoid infinite loss.")
            exp_feature_similarity_matrix = exp_feature_similarity_matrix + torch.finfo(exp_feature_similarity_matrix.dtype).eps   # add eps so values aren't considered 0 (can't do += cause that's inplace and throw an error during backprop)

        # Remove values along the diagonal (can't compare a sample to itself)
        logits = self._remove_diagonal(feature_similarity_matrix)
        exp_logits = self._remove_diagonal(exp_feature_similarity_matrix)
        label_differences = self._remove_diagonal(label_difference_matrix)

        # So now we have matrices with shape (n, n-1). Summing across rows is the outer sum (index i) and summing across columns is the inner sum (index j). This is a bit weird cause you're looping through the inner one, summing across outer, then repeating but it should work out either way.
        loss = 0.
        for j in range(inputs.shape[0] - 1):
            # Assume every other sample is a positive
            pos_logits = logits[:, j]
            pos_label_differences = label_differences[:, j]
            # Find negative samples
            negative_mask = (label_differences >= pos_label_differences.view(-1, 1)).float()
            # Calculate loss and sum across all samples (then repeat for each comparison)
            log_probabilities = pos_logits - torch.log((negative_mask * exp_logits).sum(dim=-1))    # we don't use the exponential here because we can take the log of numerator (cancels out exp) and subtract denominator
            loss += - (log_probabilities / (inputs.shape[0] * (inputs.shape[0] - 1))).sum() # negative here because higher probabilities implies positive logits are closer than negative samples, so this should decrease the loss, and vice versa

        return loss



"""
class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim] (the 2 here is because at the start they create two augmented views of each sample based on the transforms. i don't think i need to do that, should be able to do that outside the loss function.)
        # labels: [bs, label_dim]

        # HERE THEY ARE CONCATENATING THE SECOND DIMENSION DOWN (ESSENTIALLY CONCATENATING THE AUGMENTED VIEWS ON TOP OF EACH OTHER SO EACH ROW IS A SAMPLE)
        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]    (repeating tensor along 0th dim, so basically repeating the labels tensor. saying that the 0th sample has the same label as the Nth sample, 1st sample is same as N+1...)

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss
"""

