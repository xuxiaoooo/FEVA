import torch
import torch.nn as nn

class FuzzyDecisionLayer(nn.Module):
    def __init__(self, input_dim):
        super(FuzzyDecisionLayer, self).__init__()
        
        self.input_dim = input_dim

        # 传统的权重
        self.w_n = nn.Parameter(torch.randn(input_dim, 1))
        # 模糊的权重
        self.w_f = nn.Parameter(torch.randn(input_dim, 1))
        # 特征选择权重
        self.feature_weights = nn.Parameter(torch.ones(input_dim))

        # Gaussian membership function parameters
        self.mu = nn.Parameter(torch.randn(input_dim))
        self.sigma = nn.Parameter(torch.abs(torch.randn(input_dim)))
        self.fc_final = nn.Linear(1000, 1)

    def gaussian_membership(self, x):
        return torch.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))

    def forward(self, x):
        # Feature selection
        x_weighted = x * self.feature_weights

        # Combine fuzzy weights with traditional weights
        lambda_val = 0.5  # this can be a predefined constant or a learnable parameter
        w_combined = lambda_val * self.w_f + (1 - lambda_val) * self.w_n

        # Neural network output
        out_nn = torch.mm(x_weighted, w_combined)

        # Apply Gaussian membership function as activation
        out_fuzzy = self.gaussian_membership(out_nn)

        x = self.fc_final(out_fuzzy)
        
        return torch.sigmoid(x)

# Usage example
# input_dim = 64
# model = FuzzyDecisionLayer(input_dim)
# input_tensor = torch.randn(32, input_dim)
# output = model(input_tensor)
