import pandas as pd
import torch
from torch_geometric.data import HeteroData

def create_hetero_graph_from_csv(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Initialize the HeteroData object
    hetero_data = HeteroData()

    # Map unique IDs for 'account' and 'transaction'
    account_ids = pd.unique(data[['From_Account_id', 'To_Account_id']].values.ravel('K'))
    transaction_ids = data['Transaction_ID'].unique()

    account_mapping = {id: idx for idx, id in enumerate(account_ids)}
    transaction_mapping = {id: idx for idx, id in enumerate(transaction_ids)}

    # Add account nodes (placeholder features)
    hetero_data['account'].x = torch.zeros((len(account_ids), 1), dtype=torch.float)

    # Add transaction nodes with features
    transaction_features = data[['amount', 'Year', 'Month', 'Day', 'Hour', 'Minute',
                                  'Transaction_Count', 'Transaction_Sum', 'Average_Transaction_Amount']].fillna(0).values
    hetero_data['transaction'].x = torch.tensor(transaction_features, dtype=torch.float)

    # Add edges for 'initiates' relationship (account -> transaction)
    initiates_src = [account_mapping[id] for id in data['From_Account_id']]
    initiates_dst = [transaction_mapping[id] for id in data['Transaction_ID']]
    hetero_data['account', 'initiates', 'transaction'].edge_index = torch.tensor(
        [initiates_src, initiates_dst], dtype=torch.long
    )

    # Add edges for 'receives' relationship (transaction -> account)
    receives_src = [transaction_mapping[id] for id in data['Transaction_ID']]
    receives_dst = [account_mapping[id] for id in data['To_Account_id']]
    hetero_data['transaction', 'receives', 'account'].edge_index = torch.tensor(
        [receives_src, receives_dst], dtype=torch.long
    )

    # Debugging Outputs
    print("Initiates edge index dtype:", hetero_data['account', 'initiates', 'transaction'].edge_index.dtype)
    print("Receives edge index dtype:", hetero_data['transaction', 'receives', 'account'].edge_index.dtype)
    print("Account node features dtype:", hetero_data['account'].x.dtype)
    print("Transaction node features dtype:", hetero_data['transaction'].x.dtype)
    print(hetero_data)

    return hetero_data
