def analyze_cycles(cycles, data, account_mapping):
    """
    Analyze detected cycles and filter suspicious cycles based on transaction amounts.

    Args:
        cycles (list): List of detected cycles.
        data (HeteroData): Heterogeneous graph data.
        account_mapping (dict): Mapping from account IDs to internal indices.

    Returns:
        list: Suspicious cycles with account IDs, transaction details, and cycle counts.
    """
    reverse_mapping = {v: k for k, v in account_mapping.items()}
    account_cycle_counts = {}
    suspicious_cycles = []

    for cycle in cycles:
        # Check if it's a self-loop
        is_self_loop = len(cycle) == 1 or all(node == cycle[0] for node in cycle)

        if is_self_loop:
            node = cycle[0]
            original_account = reverse_mapping[node]
            if original_account not in account_cycle_counts:
                account_cycle_counts[original_account] = 0
            account_cycle_counts[original_account] += 1

            # Find the maximum transaction amount for the account
            transactions = data['account', 'initiates', 'transaction'].edge_index
            mask = (transactions[0] == node) & (transactions[1] == node)
            amounts = data['transaction'].x[mask][:, 0].tolist()  
            max_amount = max(amounts) if amounts else 0

            # Add to suspicious cycles if amount exceeds threshold
            if max_amount > 10000:  
                suspicious_cycles.append((original_account, max_amount))
        else:
            pass
    print(f"Suspicious Cycles (High-Value Transactions): {len(suspicious_cycles)}")
    for account, max_amount in suspicious_cycles:
        total_cycles = account_cycle_counts[account]
        print(f"Account: {account}, Max Transaction Amount: {max_amount}, Total Cycle Count: {total_cycles}")

    return suspicious_cycles
