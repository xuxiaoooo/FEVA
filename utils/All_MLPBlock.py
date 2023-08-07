import torch
import torch.nn as nn
import torch.nn.functional as F

class AddNorm(nn.Module):
    def __init__(self, size):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(size)

    def forward(self, x, sub):
        "Apply residual connection followed by a layer norm."
        return self.norm(x + sub)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))

class PredictionLayer(nn.Module):
    def __init__(self, d_model, hidden_dim1, hidden_dim2, output_dim):
        super(PredictionLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim1),  # Two times d_model because we concatenate x and ei
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim2, output_dim),
        )

    def forward(self, x, ei):
        combined = torch.cat((x, ei), dim=-1)
        return self.mlp(combined)
