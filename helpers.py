import torch
import torch.nn as nn
from itertools import permutations, product
import math
import copy
from torch.nn import Parameter

def build_tva_node_feature(text_features, video_features, audio_features, lengths, no_cuda):
    text_nodes = []
    audio_nodes = []
    video_nodes = []

    batch_size = text_features.size(1)

    for idx in range(batch_size):
        text_nodes.append(text_features[:lengths[idx], idx, :])
        audio_nodes.append(audio_features[:lengths[idx], idx, :])
        video_nodes.append(video_features[:lengths[idx], idx, :])
        
    if not no_cuda:
        text_nodes = torch.cat(text_nodes, dim=0).cuda()
        audio_nodes = torch.cat(audio_nodes, dim=0).cuda()
        video_nodes = torch.cat(video_nodes, dim=0).cuda()

    return text_nodes, audio_nodes, video_nodes


class DynamicWeightedLoss(nn.Module):
    def __init__(self, sigma_count):
        super(DynamicWeightedLoss, self).__init__()
        sigma = torch.ones(sigma_count, requires_grad=True)
        self.sigma = torch.nn.Parameter(sigma)

    def forward(self, *losses):
        total_loss = 0
        for i in range(len(losses)):
            total_loss += (0.5 / (self.sigma[i] ** 2)) * losses[i] + torch.log(self.sigma[i] ** 2)
        return total_loss
    


def _get_clones(module, layer_num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(layer_num)])
















class EdgeWeightedHeterGCN(torch.nn.Module):
    def __init__(self, input_dim, graph_conv_layer, num_layers, dropout_rate, no_cuda):
        super(EdgeWeightedHeterGCN, self).__init__()
        self.num_layers = num_layers
        self.no_cuda = no_cuda
        
        self.edge_weights = None
        
        self.heterogeneous_gcn_layers = _get_clones(graph_conv_layer, num_layers)
        self.mlp_layers = _get_clones(
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            ),
            num_layers
        )


class Heterogeneous_GraphConvL(torch.nn.Module):
    def __init__(self, input_feature_size, dropout=0.3, no_cuda=False):
        super(Heterogeneous_GraphConvL, self).__init__()
        self.no_cuda = no_cuda
        self.heterogeneous_gcn = Simple_GCN(input_feature_size, input_feature_size)

    def forward(self, feature, modality_number, adjencency_matrix):
        if modality_number > 1:
            heterogeneous_feature = self.heterogeneous_gcn(feature, adjencency_matrix)
            return heterogeneous_feature
        else:
            print("No need for heterogeneous graph when there is only one modality")
            return feature      

class Simple_GCN(torch.nn.Module):
    def __init__(self, input_feature, output_feature, bias=True):
        super(Simple_GCN, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.weight = Parameter(torch.FloatTensor(input_feature, output_feature))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_feature))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_feature, adj):
        try:
            input_feature = input_feature.float()
        except:
            pass
        support = torch.mm(input_feature, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output