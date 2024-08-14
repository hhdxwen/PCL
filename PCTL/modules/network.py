import torch.nn as nn
import torch
from torch.nn.functional import normalize
from modules import attention

#Batchsize-attention-net
class BaNet(nn.Module):
    def __init__(self, resnet, selfAttention ,feature_dim, class_num):
        super(BaNet, self).__init__()
        self.resnet = resnet
        self.selfAttention=selfAttention
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.rep_dim = resnet.rep_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self.rep_dim, self.rep_dim),
            nn.ReLU(),
            nn.Linear(self.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.rep_dim, self.rep_dim),
            nn.ReLU(),
            nn.Linear(self.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j, flag):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)
        ha_i = self.selfAttention(h_i, flag)
        ha_j = self.selfAttention(h_j, flag)

        z_i = normalize(self.instance_projector(ha_i), dim=1)
        z_j = normalize(self.instance_projector(ha_j), dim=1)
        
        c_i = self.cluster_projector(ha_i)
        c_j = self.cluster_projector(ha_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x, device='cpu'):
        h = self.resnet(x)
        ha = self.selfAttention(h, torch.ones(h.size(0)).to(device))
        c = self.cluster_projector(ha)
        c = torch.argmax(c, dim=1)
        return c
    
    def get_cluster_weight(self):
        for name, param in self.cluster_projector.named_parameters():
            print(name," wieght is:",param)

    def forward_z(self, x):
        h = self.resnet(x)
        ha= self.selfAttention(h, torch.ones(h.size(0)))
        z = normalize(self.instance_projector(ha), dim=1)
        return z
