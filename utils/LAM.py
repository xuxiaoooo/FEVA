import torch
import torch.nn as nn
from Activate import Activation
from All_MLPBlock import AddNorm, FeedForward, PredictionLayer

class MyModel(nn.Module):
    def __init__(self, d_model, d_ff, interval=None, percentage=None):
        super(MyModel, self).__init__()
        self.act = Activation(interval, percentage)  # Add the activation module
        self.addnorm1 = AddNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.addnorm2 = AddNorm(d_model)
        self.pred = PredictionLayer(d_model)
        self.final_layer = nn.Linear(d_model, 2)

    def forward(self, x, ei):
        results = []
        for segment in x:
            # Apply activation
            out = self.act(segment)

            # Apply Add & Norm, Feed-forward, Add & Norm
            out = self.addnorm1(out, self.ff(out))
            out = self.addnorm2(out, self.ff(out))

            out = self.pred(out, ei)
            out = self.final_layer(out)
            _, predicted = torch.max(out, 0)
            results.append(predicted)
        votes = torch.bincount(torch.tensor(results))
        final_prediction = torch.argmax(votes)
        return final_prediction
