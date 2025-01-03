import torch
from graph_construction import create_hetero_graph_from_csv
from encoder_decoder import HeteroGraphAutoencoder

# Step 1: Test the autoencoder with your CSV file
def test_autoencoder_with_csv(file_path):
    # Create HeteroData from the CSV
    hetero_data = create_hetero_graph_from_csv(file_path)

    # Initialize the model
    metadata = hetero_data.metadata()  # Extract metadata from the graph
    model = HeteroGraphAutoencoder(metadata)

    # Prepare inputs for the model
    x_dict = hetero_data.x_dict  # Node features for 'account' and 'transaction'
    edge_index_dict = hetero_data.edge_index_dict  # Edge indices for all relationships
    edge_index = hetero_data['account', 'initiates', 'transaction'].edge_index  # Edge index for 'initiates'

    # Perform a forward pass
    output = model(x_dict, edge_index_dict, edge_index)

    # Print results
    print("Reconstructed Edge Probabilities:")
    print(output)

    # Validate the output shape
    print("Output Shape:", output.shape)
    assert output.shape[0] == edge_index.shape[1], "Output shape mismatch with edge count"

# Run the test
if __name__ == "__main__":
    csv_file_path = "/home/binit/Anti-Money-Laundrying/data/test.csv"  # Replace with the actual path to your CSV
    test_autoencoder_with_csv(csv_file_path)
