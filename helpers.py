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

    def _build_cross_modal_edges(self, features, num_modalities, 
                                                    sequence_lengths, past_window, 
                                                    future_window):
        """
        Build heterogeneous edges connecting nodes across different modalities.
        
        Args:
            features: Input features
            num_modalities: Number of modalities
            sequence_lengths: List of sequence lengths for each dialogue
            past_window: Context window size for past nodes (-1 for unlimited)
            future_window: Context window size for future nodes (-1 for unlimited)
            
        Returns:
            edge_index: Tensor of shape (2, num_edges) containing edge connections
        """
        edge_indices = []
        total_sequence_length = sum(sequence_lengths)
        all_node_indices = list(range(total_sequence_length * num_modalities))
        modality_nodes = [None] * num_modalities

        # Map nodes by modality
        for modality_idx in range(num_modalities):
            modality_nodes[modality_idx] = all_node_indices[
                modality_idx * total_sequence_length:(modality_idx + 1) * total_sequence_length
            ]

        # Build edges for each sequence and modality pair
        position = 0
        for sequence_length in sequence_lengths:
            for src_modality, tgt_modality in permutations(range(num_modalities), 2):
                for node_position, source_node in enumerate(
                    modality_nodes[src_modality][position:position + sequence_length]
                ):
                    # Determine target nodes based on window configuration
                    if past_window == -1 and future_window == -1:
                        # Connect to all nodes in sequence
                        target_nodes = modality_nodes[tgt_modality][
                            position:position + sequence_length
                        ]
                    elif past_window == -1:
                        # Connect to current and future nodes
                        target_nodes = modality_nodes[tgt_modality][
                            position:min(position + sequence_length, 
                                       position + node_position + future_window + 1)
                        ]
                    elif future_window == -1:
                        # Connect to past and current nodes
                        target_nodes = modality_nodes[tgt_modality][
                            max(position, position + node_position - past_window):
                            position + sequence_length
                        ]
                    else:
                        # Connect within temporal window
                        target_nodes = modality_nodes[tgt_modality][
                            max(position, position + node_position - past_window):
                            min(position + sequence_length, 
                                position + node_position + future_window + 1)
                        ]
                    
                    edge_indices.extend(list(product([source_node], target_nodes)))
            
            position += sequence_length

        # Convert to tensor
        edge_index = torch.tensor(edge_indices).permute(1, 0)
        if not self.no_cuda:
            edge_index = edge_index.cuda()

        return edge_index

    def _construct_gcn_normalized_adj(self, edge_index, edge_weights=None, 
                                      num_nodes=100, no_cuda=False):
        """
        Construct GCN-normalized adjacency matrix from edge indices and weights.
        
        Applies normalization: D^(-1/2) * A * D^(-1/2) where D is the degree matrix
        and A is the adjacency matrix. This normalization prevents exploding/vanishing
        gradients in graph convolutions.
        
        Args:
            edge_index: Tensor of shape (2, num_edges) with edge connections
            edge_weights: Optional tensor of edge weights. If None, defaults to ones
            num_nodes: Total number of nodes in the graph
            
        Returns:
            normalized_adjacency: Normalized adjacency matrix (dense)
        """
        # Initialize edge weights if not provided
        if edge_weights is not None:
            edge_weights = edge_weights.squeeze()
        else:
            edge_weights = torch.ones(edge_index.size(1))
            if not self.no_cuda:
                edge_weights = edge_weights.cuda()

        # Create sparse adjacency matrix from edge indices and weights
        sparse_adjacency = torch.sparse_coo_tensor(
            edge_index,
            edge_weights,
            size=(num_nodes, num_nodes)
        )
        
        # Convert to dense for normalization
        adjacency_matrix = sparse_adjacency.to_dense()
        
        # Compute degree normalization: D^(-1/2)
        degree_sum = torch.sum(adjacency_matrix, dim=1)
        inverse_sqrt_degree = torch.pow(degree_sum, -0.5)
        inverse_sqrt_degree[inverse_sqrt_degree == float('inf')] = 0
        
        # Create diagonal matrix from inverse square root degrees
        inverse_sqrt_degree_matrix = torch.diag_embed(inverse_sqrt_degree)
        
        # Apply normalization: D^(-1/2) * A * D^(-1/2)
        normalized_adjacency = torch.matmul(
            inverse_sqrt_degree_matrix,
            torch.matmul(adjacency_matrix, inverse_sqrt_degree_matrix)
        )
        
        # Ensure tensor is on correct device
        if not self.no_cuda and torch.cuda.is_available():
            normalized_adjacency = normalized_adjacency.cuda()

        return normalized_adjacency



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