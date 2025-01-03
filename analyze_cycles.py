def analyze_cycles(cycles, data, account_mapping):
    """
    Analyze detected cycles and filter suspicious cycles based on transaction amounts.

    Args:
        cycles (list): List of detected cycles.
        data (HeteroData): Heterogeneous graph data.
        account_mapping (dict): Mapping from account IDs to internal indices.

    Returns:
        list: Suspicious cycles with account IDs and transaction details.
    """
    reverse_mapping = {v: k for k, v in account_mapping.items()}
    suspicious_cycles = []

    for cycle in cycles:
        # Check transaction amounts in the cycle
        amounts = []
        is_self_loop = len(cycle) == 1 or all(node == cycle[0] for node in cycle)

        if is_self_loop:
            # Handle self-loops specifically
            node = cycle[0]
            original_account = reverse_mapping[node]

            # Find the maximum transaction amount involving the account
            transactions = data['account', 'initiates', 'transaction'].edge_index
            mask = (transactions[0] == node) & (transactions[1] == node)
            amounts.extend(data['transaction'].x[mask][:, 0].tolist())

            max_amount = max(amounts) if amounts else 0
            if max_amount > 10000:  # Example threshold
                suspicious_cycles.append((original_account, max_amount))
        else:
            # Handle other cycles if needed
            pass

    print(f"Suspicious Cycles (High-Value Transactions): {len(suspicious_cycles)}")
    for account, max_amount in suspicious_cycles:
        print(f"Account: {account}, Max Transaction Amount: {max_amount}")

    return suspicious_cycles
