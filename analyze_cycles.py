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
    suspicious_cycles = []

    for cycle in cycles:
        # Map internal indices to original account IDs
        original_accounts = [reverse_mapping[node] for node in cycle]

        # Check transaction amounts in the cycle
        amounts = []
        for i in range(len(cycle)):
            src = cycle[i]
            dst = cycle[(i + 1) % len(cycle)]  
            transactions = data['account', 'initiates', 'transaction'].edge_index
            mask = (transactions[0] == src) & (transactions[1] == dst)
            amounts.extend(data['transaction'].x[mask][:, 0].tolist())  

        max_amount = max(amounts) if amounts else 0

        # Flag suspicious cycles
        if max_amount > 1000000:  # if the amount is higher than 1000000 it would detect. 
            suspicious_cycles.append((original_accounts, max_amount, len(cycle)))

    # Print results
    print(f"Suspicious Cycles (High-Value Transactions): {len(suspicious_cycles)}")
    for accounts, max_amount, cycle_length in suspicious_cycles:
        print(f"Cycle: {accounts}, Max Transaction Amount: {max_amount}, Cycle Length: {cycle_length}")

    return suspicious_cycles
