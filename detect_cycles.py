import networkx as nx
import torch

def detect_cycles(model, data, threshold=0.5, min_cycle_length=2):
    """
    Detect cycles (multi-loops) in the reconstructed graph.

    Args:
        model (torch.nn.Module): Trained graph autoencoder.
        data (HeteroData): Heterogeneous graph data.
        threshold (float): Threshold for edge reconstruction probabilities.
        min_cycle_length (int): Minimum length of cycles to consider.

    Returns:
        list: Detected multi-loops in the graph.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model.encoder(data.x_dict, data.edge_index_dict)
        pos_edge_index = data['account', 'initiates', 'transaction'].edge_index
        predicted_probs = model.decoder(embeddings['account'], pos_edge_index, sigmoid=True)

    # Apply threshold to predicted edges
    predicted_edges = pos_edge_index[:, predicted_probs > threshold]

    # Convert to NetworkX graph
    G = nx.DiGraph()
    for src, dst in zip(predicted_edges[0], predicted_edges[1]):
        G.add_edge(src.item(), dst.item())

    # Detect all cycles (filter by length)
    cycles = [cycle for cycle in nx.simple_cycles(G) if len(cycle) >= min_cycle_length]
    print(f"Detected {len(cycles)} cycles (length >= {min_cycle_length}).")
    return cycles
