import torch
from graph_construction import create_hetero_graph_from_csv
from encoder_decoder import HeteroGraphAutoencoder
from detect_cycles import detect_cycles
from analyze_cycles import analyze_cycles


def map_internal_to_account_id(cycles, account_mapping):
    """
    Maps internal graph indices to original account IDs and removes duplicates for self-loops.

    Args:
        cycles (list): List of cycles detected in the graph.
        account_mapping (dict): Mapping of account IDs to internal indices.

    Returns:
        list: Cycles with unique original account IDs.
    """
    # Reverse the account_mapping dictionary
    reverse_mapping = {v: k for k, v in account_mapping.items()}

    # Map internal indices to account IDs and ensure uniqueness
    mapped_cycles = []
    for cycle in cycles:
        # Map internal indices to account IDs and remove duplicates
        unique_cycle = list({int(reverse_mapping[node]) for node in cycle})
        mapped_cycles.append(unique_cycle)

    return mapped_cycles

def main_detect_and_analyze(csv_file_path, model_path, threshold=0.5):
    # Load data
    hetero_data, account_mapping = create_hetero_graph_from_csv(csv_file_path)

    # Load model
    metadata = hetero_data.metadata()
    model = HeteroGraphAutoencoder(metadata)
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")

    # Detect cycles
    cycles = detect_cycles(model, hetero_data, threshold)

    # Analyze cycles
    suspicious_cycles = analyze_cycles(cycles, hetero_data, account_mapping)

    # Determine money laundering
    if len(suspicious_cycles) > 0:
        print("Potential Money Laundering Detected!")
    else:
        print("No Money Laundering Detected.")

    return suspicious_cycles


if __name__ == "__main__":
    csv_file_path = "/home/binit/Anti-Money-Laundrying/data/test.csv"  
    model_path = "trained_model.pt" 
    main_detect_and_analyze(csv_file_path, model_path, threshold=0.5)
