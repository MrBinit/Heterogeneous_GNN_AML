import torch
from graph_construction import create_hetero_graph_from_csv
from encoder_decoder import HeteroGraphAutoencoder
from detect_cycles import detect_cycles
from analyze_cycles import analyze_cycles

def main_detect_and_analyze(csv_file_path, model_path, threshold=0.5):
    # Load data
    hetero_data = create_hetero_graph_from_csv(csv_file_path)

    # Load model
    metadata = hetero_data.metadata()
    model = HeteroGraphAutoencoder(metadata)
    model.load_state_dict(torch.load(model_path))

    # Detect cycles
    cycles = detect_cycles(model, hetero_data, threshold)

    # Analyze cycles
    suspicious_cycles = analyze_cycles(cycles, hetero_data)

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
