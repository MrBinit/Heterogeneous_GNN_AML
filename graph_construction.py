import pandas as pd
import torch
from torch_geometric.data import HeteroData

def create_hetero_graph_from_csv(file_path):
    data = pd.read_csv(file_path)
    hetero_data = HeteroData()

    # Extract unique account and transaction IDs
    account_ids = pd.unique(data[['From_Account_id', 'To_Account_id']].values.ravel('K'))
    transaction_ids = data['Transaction_ID'].unique()
    # mappings for accounts and transactions
    account_mapping = {id: idx for idx, id in enumerate(account_ids)}
    transaction_mapping = {id: idx for idx, id in enumerate(transaction_ids)}

    # Adding 'account' and 'transcation' nodes with placeholder features
    hetero_data['account'].x = torch.zeros((len(account_ids), 1))
    transaction_features = data[['amount', 'Year', 'Month', 'Day', 'Hour', 'Minute',
                                 'Transaction_Count', 'Transaction_Sum', 'Average_Transaction_Amount']].values
    hetero_data['transaction'].x = torch.tensor(transaction_features, dtype=torch.float)

    # transaction initiated and received edges.
    initiates_src = [account_mapping[id] for id in data['From_Account_id']]
    initiates_dst = [transaction_mapping[id] for id in data['Transaction_ID']]
    hetero_data['account', 'initiates', 'transaction'].edge_index = torch.tensor([initiates_src, initiates_dst], dtype=torch.long)

    receives_src = [transaction_mapping[id] for id in data['Transaction_ID']]
    receives_dst = [account_mapping[id] for id in data['To_Account_id']]
    hetero_data['transaction', 'receives', 'account'].edge_index = torch.tensor([receives_src, receives_dst], dtype=torch.long)
    print(hetero_data)

    return hetero_data


if __name__ == "__main__":
    hetero_data = create_hetero_graph_from_csv('/home/binit/Anti-Money-Laundrying/data/test.csv')
    print(hetero_data)
    print("Nodes of accounts both receiver and sender: ", hetero_data['account'].x.shape)
    print("Nodes of transcation boths receivers and sender:  ", hetero_data['transaction'].x.shape)
    print("edge indexes for initiation: ", hetero_data['account', 'initiates', 'transaction'].edge_index.shape)
    print("edge index for receive: ",hetero_data['transaction', 'receives', 'account'].edge_index.shape)
