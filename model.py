import torch.nn as nn
import torch
from helpers import (
    EdgeWeightedHeterGCN, 
    Heterogeneous_GraphConvL, 
    SentenceMatchingModule, 
    build_tva_node_feature
)


class GS_Model(nn.Module):
    """
    Graph Smile (GS) Model for multimodal emotion and sentiment analysis.
    Integrates text, video, and audio features using heterogeneous graph convolution.
    """

    def __init__(self, args, embedding_dims, n_classes_emo):
        """
        Initialize the GS_Model.
        
        Args:
            args: Configuration arguments containing model hyperparameters
            embedding_dims: Tuple of (text_dim, video_dim, audio_dim)
            n_classes_emo: Number of emotion classes
        """
        super(GS_Model, self).__init__()
        
        # Store configuration
        self.textf_mode = args.textf_mode
        self.no_cuda = args.no_cuda
        self.win_p = args.win[0]
        self.win_f = args.win[1]
        self.modals = args.modals
        self.shift_win = args.shift_win

        # ============ Setup text feature indices ============
        self._setup_text_feature_indices()
        
        # ============ Batch normalization for text features ============
        self.batchnorms_t = nn.ModuleList([
            nn.BatchNorm1d(embedding_dims[0]) for _ in self.used_t_indices
        ])

        # ============ Dimension projection layers ============
        self._setup_dimension_layers(args, embedding_dims)
        
        # ============ Heterogeneous Graph Convolution layers ============
        self._setup_heterogeneous_graph_convolutions(args)
        
        # ============ Fusion and output layers ============
        self.modal_fusion = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.LeakyReLU(),
        )
        
        self.emo_output = nn.Linear(args.hidden_dim, n_classes_emo)
        self.sen_output = nn.Linear(args.hidden_dim, 3)
        self.senshift = SentenceMatchingModule(
            args.hidden_dim, 
            args.drop,
            args.shift_win
        )

    def _setup_text_feature_indices(self):
        """Determine which text features to use based on textf_mode."""
        text_feature_mapping = {
            'concat4': [0, 1, 2, 3],
            'sum4': [0, 1, 2, 3],
            'concat2': [0, 1],
            'sum2': [0, 1],
            'textf0': [0],
            'textf1': [1],
            'textf2': [2],
            'textf3': [3],
        }
        
        if self.textf_mode not in text_feature_mapping:
            raise ValueError(f"Unsupported textf_mode: {self.textf_mode}")
        
        self.used_t_indices = text_feature_mapping[self.textf_mode]

    def _setup_dimension_layers(self, args, embedding_dims):
        """Setup linear projection layers for each modality."""
        # Text dimension layer
        text_input_dim = (
            len(self.used_t_indices) * embedding_dims[0]
            if self.textf_mode.startswith('concat')
            else embedding_dims[0]
        )
        self.dim_layer_t = nn.Sequential(
            nn.Linear(text_input_dim, args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop)
        )
        
        # Video dimension layer
        self.dim_layer_v = nn.Sequential(
            nn.Linear(embedding_dims[1], args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop),
        )
        
        # Audio dimension layer
        self.dim_layer_a = nn.Sequential(
            nn.Linear(embedding_dims[2], args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop),
        )

    def _setup_heterogeneous_graph_convolutions(self, args):
        """Setup heterogeneous graph convolution layers for modality pairs."""
        # Text-Video convolution
        text_video_conv_layer = Heterogeneous_GraphConvL(
            args.hidden_dim, 
            args.drop,
            args.no_cuda
        )
        self.hetergconv_tv = EdgeWeightedHeterGCN(
            args.hidden_dim,
            text_video_conv_layer,
            args.heter_n_layers[0],
            args.drop,
            args.no_cuda,
        )
        
        # Text-Audio convolution
        text_audio_conv_layer = Heterogeneous_GraphConvL(
            args.hidden_dim, 
            args.drop,
            args.no_cuda
        )
        self.hetergconv_ta = EdgeWeightedHeterGCN(
            args.hidden_dim,
            text_audio_conv_layer,
            args.heter_n_layers[1],
            args.drop,
            args.no_cuda,
        )
        
        # Video-Audio convolution
        video_audio_conv_layer = Heterogeneous_GraphConvL(
            args.hidden_dim, 
            args.drop,
            args.no_cuda
        )
        self.hetergconv_va = EdgeWeightedHeterGCN(
            args.hidden_dim,
            video_audio_conv_layer,
            args.heter_n_layers[2],
            args.drop,
            args.no_cuda,
        )

    def forward(self, feature_t0, feature_t1, feature_t2, feature_t3,
                feature_v, feature_a, umask, qmask, dia_lengths):
        """
        Forward pass of the GS_Model.
        
        Args:
            feature_t0-t3: Text feature variants
            feature_v: Video features
            feature_a: Audio features
            umask: User mask
            qmask: Query mask
            dia_lengths: Dialogue lengths
            
        Returns:
            Tuple of (emotion_logits, sentiment_logits, shift_logits, fused_features)
        """
        # ============ Process text features ============
        fused_text_features = self._process_text_features(
            feature_t0, feature_t1, feature_t2, feature_t3
        )
        
        # ============ Project to common hidden dimension ============
        fused_text_dim = self.dim_layer_t(fused_text_features)
        fused_video_dim = self.dim_layer_v(feature_v)
        fused_audio_dim = self.dim_layer_a(feature_a)

        # ============ Convert to node features for heterogeneous graphs ============
        emo_text, emo_video, emo_audio = build_tva_node_feature(
            fused_text_dim, 
            fused_video_dim, 
            fused_audio_dim,
            dia_lengths, 
            self.no_cuda
        )

        # ============ Apply heterogeneous graph convolutions ============
        hetergconv_text_video, heter_edge_idx = self.hetergconv_tv(
            (emo_text, emo_video), dia_lengths, self.win_p, self.win_f
        )
        hetergconv_text_audio, heter_edge_idx = self.hetergconv_ta(
            (emo_text, emo_audio), dia_lengths, self.win_p, self.win_f,
            heter_edge_idx
        )
        hetergconv_video_audio, heter_edge_idx = self.hetergconv_va(
            (emo_video, emo_audio), dia_lengths, self.win_p, self.win_f,
            heter_edge_idx
        )

        # ============ Fuse multimodal features ============
        fused_features = (
            self.modal_fusion(hetergconv_text_video[0]) + 
            self.modal_fusion(hetergconv_text_audio[0]) + 
            self.modal_fusion(hetergconv_text_video[1]) +
            self.modal_fusion(hetergconv_video_audio[0]) +
            self.modal_fusion(hetergconv_text_audio[1]) +
            self.modal_fusion(hetergconv_video_audio[1])
        ) / 6

        # ============ Generate predictions ============
        logit_emo = self.emo_output(fused_features)
        logit_sen = self.sen_output(fused_features)
        logit_shift = self.senshift(fused_features, fused_features, dia_lengths)

        return logit_emo, logit_sen, logit_shift, fused_features

    def _process_text_features(self, feature_t0, feature_t1, feature_t2, feature_t3):
        """
        Process and fuse multiple text feature variants.
        
        Handles batch normalization and either concatenation or averaging
        based on the configured textf_mode.
        
        Args:
            feature_t0-t3: Four text feature variants
            
        Returns:
            Fused text features
        """
        all_text_features = [feature_t0, feature_t1, feature_t2, feature_t3]
        seq_len, batch_size, feature_dim = feature_t0.shape
        normalized_text_features = []
        
        # Apply batch normalization to selected text features
        for feature_idx, norm_layer in zip(self.used_t_indices, self.batchnorms_t):
            feature = all_text_features[feature_idx]
            normalized_feature = norm_layer(
                feature.transpose(0, 1).reshape(-1, feature_dim)
            )
            normalized_feature = normalized_feature.reshape(
                batch_size, seq_len, feature_dim
            ).transpose(1, 0)
            normalized_text_features.append(normalized_feature)

        # Fuse text features based on mode
        if self.textf_mode in ['concat4', 'concat2']:
            fused_text = torch.cat(normalized_text_features, dim=-1)
        elif self.textf_mode in ['sum4', 'sum2']:
            fused_text = sum(normalized_text_features) / len(normalized_text_features)
        else:
            fused_text = normalized_text_features[0]
        
        return fused_text