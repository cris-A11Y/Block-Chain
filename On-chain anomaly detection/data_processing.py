import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch_geometric.data import Data

def load_and_process_data(filepath):
    """
    Load Ethereum transaction CSV, encode addresses into node indices, create edge list,
    generate node features and labels, and return a PyG Data object.

    Args:
        filepath (str): Path to the CSV file containing transaction data with 'from_address', 'to_address',
                        and optional scam labels 'from_scam' and 'to_scam'.

    Returns:
        data (torch_geometric.data.Data): PyG Data object with graph structure.
        num_nodes (int): Total number of unique addresses (nodes).
    """
    # Load transaction data
    df = pd.read_csv(filepath)

    # Encode all addresses to unique integer node IDs
    all_addresses = pd.concat([df['from_address'], df['to_address']])
    le = LabelEncoder()
    le.fit(all_addresses)

    # Map addresses to integer node IDs for edges
    df['from_id'] = le.transform(df['from_address'])
    df['to_id'] = le.transform(df['to_address'])
    edges = df[['from_id', 'to_id']].values

    # Prepare address-wise scam labels
    addr_labels = pd.concat([
        df[['from_address', 'from_scam']].rename(columns={'from_address': 'address', 'from_scam': 'label'}),
        df[['to_address', 'to_scam']].rename(columns={'to_address': 'address', 'to_scam': 'label'})
    ])

    # Remove missing and duplicate entries
    addr_labels = addr_labels.dropna().drop_duplicates('address')
    addr_labels['node_id'] = le.transform(addr_labels['address'])
    labels_dict = dict(zip(addr_labels['node_id'], addr_labels['label']))

    # Prepare labels array, default to -1 (unknown)
    num_nodes = len(le.classes_)
    labels = np.full(num_nodes, -1)
    for idx, label in labels_dict.items():
        labels[idx] = int(label)

    # Build edge index tensor [2, num_edges]
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    # Initialize node features (here using dummy 1-feature per node)
    x = torch.ones((num_nodes, 1), dtype=torch.float)

    # Prepare labels tensor
    y = torch.tensor(labels, dtype=torch.long)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    # Save address to node_id mapping in the Data object (optional, for reference)
    address_mapping = dict(zip(range(num_nodes), le.classes_))
    data.address_mapping = address_mapping

    return data, num_nodes
