import torch
import torch.nn as nn
import torch.nn.functional as F

class PSI(nn.Module):
    def __init__(self, config, dim_size, num_actions):
        super().__init__()
        model = []
        model.append(nn.Conv2d(config[0],config[1],kernel_size=3,stride=1))
        model.append(nn.LeakyReLU())
        model.append(nn.BatchNorm2d(config[1]))
        model.append(nn.Conv2d(config[1],config[2],kernel_size=3,stride=1))
        model.append(nn.LeakyReLU())
        model.append(nn.BatchNorm2d(config[2]))
        model.append(nn.MaxPool2d(kernel_size=2,stride=2))

        model.append(nn.Flatten())
        for i in range(3,len(config)-1):
            model.append(nn.Linear(config[i],config[i+1]))
            model.append(nn.LeakyReLU())
        model.append(nn.Linear(config[-1],num_actions*dim_size))
        self.model = nn.Sequential(*model)
        self.num_actions = num_actions
        self.dim_size = dim_size
    
    def forward(self,state):
        successor = self.model(state)
        successor = successor.view(-1,self.num_actions,self.dim_size)
        return successor


class FQN(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1, bias=False)
    
    def forward(self, successor_feature):
        return self.linear(successor_feature)