import torch
from torch_geometric.nn import HeteroConv, GCNConv

class HeteroGraphEncoder(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.conv1 = HeteroConv({
            ('account', 'initiates', 'transaction'): GCNConv(-1, 16),
            ('transaction', 'receives', 'account'): GCNConv(-1, 16),
        }, aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        # Encode node features
        x_dict = self.conv1(x_dict, edge_index_dict)
        return x_dict


class GraphDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, account_embeddings, transaction_embeddings, edge_index):
        # Decode edge existence using dot product
        src, dst = edge_index
        scores = (account_embeddings[src] * transaction_embeddings[dst]).sum(dim=-1)
        return torch.sigmoid(scores) 


class GraphAutoencoder(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.encoder = HeteroGraphEncoder(metadata)
        self.decoder = GraphDecoder()

    def forward(self, x_dict, edge_index_dict, edge_index):
        # Encode the graph and decode edges
        embeddings = self.encoder(x_dict, edge_index_dict)
        return self.decoder(embeddings['account'], embeddings['transaction'], edge_index)
