import networkx as nx
import torch

def detect_cycles(model, data, threshold=0.5):
    
    # Encode the graph
    model.eval()
    with torch.no_grad():
        embeddings = model.encoder(data.x_dict, data.edge_index_dict)
        pos_edge_index = data['account', 'initiates', 'transaction'].edge_index
        predicted_probs = model.decoder(embeddings['account'], pos_edge_index, sigmoid=True)

    # Apply threshold to determine predicted edges
    predicted_edges = pos_edge_index[:, predicted_probs > threshold]

    # Convert to NetworkX graph
    G = nx.DiGraph()
    for src, dst in zip(predicted_edges[0], predicted_edges[1]):
        G.add_edge(src.item(), dst.item())

    # Detect simple cycles
    cycles = list(nx.simple_cycles(G))
    print(f"Detected {len(cycles)} cycles:")
    for cycle in cycles:
        print(cycle)

    return cycles
