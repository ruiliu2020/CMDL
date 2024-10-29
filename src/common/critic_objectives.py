import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def mlp(dim, hidden_dim, output_dim, layers, activation):
    activation = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    # for _ in range(layers):
    #      seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)
    
# Concat critic with the InfoNCE (NCE) objective
class InfoNCECritic(nn.Module):
    def __init__(self, A_dim, B_dim, hidden_dim, layers, activation, **extra_kwargs):
          super(InfoNCECritic, self).__init__()
          # output is scalar score
          self._f = mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    def forward(self, x_samples, y_samples):
        """
        [bs,hs]
        eg: x=[[a],[b],[c]]=y
        x_tile = [[a][b][c][a][b][c][a][b][c]]
        y_title = [[a][a][a][b][b][b][c][c][c]]
        t1 = f([[aa][ba][ca]...])
        t0 = f([[aa][bb][cc]]) = 
        """
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        
        T0 = self._f(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self._f(torch.cat([x_tile, y_tile], dim = -1))  

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        return -lower_bound

# Concat critic with the CLUBInfoNCE (NCE-CLUB) objective
class CLUBInfoNCECritic(nn.Module):
    def __init__(self, A_dim, B_dim, hidden_dim, layers, activation, **extra_kwargs):
          super(CLUBInfoNCECritic, self).__init__()
 
          self._f = mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    # CLUB loss
    def forward(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([y_samples,x_samples], dim = -1)) 
        T1 = self._f(torch.cat([y_tile, x_tile], dim = -1))  

        return T0.mean() - T1.mean()

    # InfoNCE loss
    def learning_loss(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([y_samples,x_samples], dim = -1))
        T1 = self._f(torch.cat([y_tile, x_tile], dim = -1)) 

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        return -lower_bound

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss  
    
