import torch
import torch.nn as nn

class Activation(nn.Module):
    def __init__(self, interval=None, percentage=None):
        super(Activation, self).__init__()
        self.interval = interval
        self.percentage = percentage

    def forward(self, x):
        if self.interval is not None:
            x = self.interval_activation(x, self.interval)
        elif self.percentage is not None:
            x = self.percentage_activation(x, self.percentage)
        return x

    def interval_activation(self, x, interval):
        mask = torch.ones_like(x)
        mask[::interval] = 0  # set the mask to 0 at each 'interval' element
        return x * mask.to(x.device)

    def percentage_activation(self, x, percentage):
        total_elements = torch.prod(torch.tensor(x.shape))
        num_to_deactivate = int(total_elements * percentage)  # calculate the number of elements to deactivate
        indices = torch.randperm(total_elements)[:num_to_deactivate]  # randomly choose indices to deactivate
        mask = torch.ones(total_elements)
        mask[indices] = 0  # set the mask to 0 at the chosen indices
        mask = mask.view(x.shape)
        return x * mask.to(x.device)
