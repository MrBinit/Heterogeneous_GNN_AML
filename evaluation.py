import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from graph_construction import create_hetero_graph_from_csv
from encoder_decoder import HeteroGraphAutoencoder
from torch_geometric.utils import negative_sampling


# Define evaluation function
def evaluate(model, data):
    pos_edge_index = data['account', 'initiates', 'transaction'].edge_index
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data['account'].x.size(0))

    # Encode the graph
    model.eval()
    with torch.no_grad():
        embeddings = model.encoder(data.x_dict, data.edge_index_dict)

        # Predict probabilities
        pos_pred = model.decoder(embeddings['account'], pos_edge_index, sigmoid=True)
        neg_pred = model.decoder(embeddings['account'], neg_edge_index, sigmoid=True)

        # True labels
        pos_labels = torch.ones(pos_pred.size(0))
        neg_labels = torch.zeros(neg_pred.size(0))
        labels = torch.cat([pos_labels, neg_labels])

        # Predictions
        preds = torch.cat([pos_pred, neg_pred])

        # Compute metrics
        roc_auc = roc_auc_score(labels.detach().numpy(), preds.detach().numpy())
        ap = average_precision_score(labels.detach().numpy(), preds.detach().numpy())

    print(f"Evaluation Results -> ROC AUC: {roc_auc:.4f}, Average Precision: {ap:.4f}")
    return roc_auc, ap


# Main function
def main(csv_file_path):
    # Load data
    hetero_data = create_hetero_graph_from_csv(csv_file_path)

    # Initialize model
    metadata = hetero_data.metadata()
    model = HeteroGraphAutoencoder(metadata)

    # Load trained model
    model.load_state_dict(torch.load("trained_model.pt"))
    print("Model loaded from 'trained_model.pt'")

    # Evaluate the model
    evaluate(model, hetero_data)


if __name__ == "__main__":
    csv_file_path = "/home/binit/Anti-Money-Laundrying/data/test.csv"  # Replace with your dataset path
    main(csv_file_path)
