import torch
from torch_geometric.nn import HeteroConv, SAGEConv, InnerProductDecoder

class HeteroGraphEncoder(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        # Define heterogeneous graph convolution
        self.conv1 = HeteroConv({
            ('account', 'initiates', 'transaction'): SAGEConv((-1, -1), 16),
            ('transaction', 'receives', 'account'): SAGEConv((-1, -1), 16),
        }, aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        # Perform graph convolution
        x_dict = self.conv1(x_dict, edge_index_dict)
        return x_dict

class HeteroGraphAutoencoder(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.encoder = HeteroGraphEncoder(metadata)  # Encoder for heterogeneous graphs
        self.decoder = InnerProductDecoder()  # Decoder to reconstruct edges

    def forward(self, x_dict, edge_index_dict, edge_index):
        # Encode the node features
        embeddings = self.encoder(x_dict, edge_index_dict)
        
        # Check if required embeddings exist
        assert 'account' in embeddings and 'transaction' in embeddings, \
            "Missing embeddings for required node types"
        
        # Decode using embeddings and edge index
        reconstructed = self.decoder(embeddings['account'], edge_index)
        return reconstructed
