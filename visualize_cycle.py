import matplotlib.pyplot as plt
import networkx as nx

def visualize_cycles(cycles, graph):
    G = nx.DiGraph()
    for cycle in cycles:
        for i in range(len(cycle)):
            src = cycle[i]
            dst = cycle[(i + 1) % len(cycle)]
            G.add_edge(src, dst)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="black", node_size=2000, font_size=15)
    plt.title("Detected Cycles")
    plt.show()
