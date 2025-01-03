def analyze_cycles(cycles, data):
    suspicious_cycles = []
    for cycle in cycles:
        # Check transaction amounts in the cycle
        amounts = []
        for i in range(len(cycle)):
            src = cycle[i]
            dst = cycle[(i + 1) % len(cycle)]  # Next node in the cycle
            # Find transactions between src and dst
            transactions = data['account', 'initiates', 'transaction'].edge_index
            mask = (transactions[0] == src) & (transactions[1] == dst)
            amounts.extend(data['transaction'].x[mask][:, 0].tolist())  # Assuming 'amount' is the first feature

        if len(amounts) > 0 and max(amounts) > 10000:  # Example threshold
            suspicious_cycles.append((cycle, max(amounts)))

    print(f"Suspicious Cycles (High-Value Transactions): {len(suspicious_cycles)}")
    for cycle, max_amount in suspicious_cycles:
        print(f"Cycle: {cycle}, Max Transaction Amount: {max_amount}")

    return suspicious_cycles
