import torch
import torch.optim as optim
from torch_geometric.utils import negative_sampling
from graph_construction import create_hetero_graph_from_csv
from encoder_decoder import HeteroGraphAutoencoder


# Define training function
def train(model, data, num_epochs=50, learning_rate=0.01):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare edge indices for positive and negative edges
    pos_edge_index = data['account', 'initiates', 'transaction'].edge_index

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Encode and reconstruct
        embeddings = model.encoder(data.x_dict, data.edge_index_dict)
        reconstructed = model.decoder(embeddings['account'], pos_edge_index)

        # Negative sampling
        neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data['account'].x.size(0))

        # Compute loss
        pos_loss = -torch.log(reconstructed + 1e-15).mean()  # BCE for positive edges
        neg_reconstructed = model.decoder(embeddings['account'], neg_edge_index)
        neg_loss = -torch.log(1 - neg_reconstructed + 1e-15).mean()

        loss = pos_loss + neg_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Log progress
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    print("Training complete.")
    return model


# Main function
def main(csv_file_path, num_epochs=50, learning_rate=0.01):
    # Load data
    hetero_data = create_hetero_graph_from_csv(csv_file_path)

    # Initialize model
    metadata = hetero_data.metadata()
    model = HeteroGraphAutoencoder(metadata)

    # Train the model
    trained_model = train(model, hetero_data, num_epochs=num_epochs, learning_rate=learning_rate)

    # Save the trained model
    torch.save(trained_model.state_dict(), "trained_model.pt")
    print("Model saved to 'trained_model.pt'")


if __name__ == "__main__":
    csv_file_path = "/home/binit/Anti-Money-Laundrying/data/test.csv" 
    main(csv_file_path, num_epochs=50, learning_rate=0.01)
