import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def load_transaction_data(file_path):
    """
    Load Ethereum transaction data from a CSV file.
    """
    df = pd.read_csv(file_path)
    return df[['from_address', 'to_address', 'value']]

def build_transaction_graph(df):
    """
    Build a directed transaction graph from Ethereum transaction data.
    Nodes represent addresses and edges represent transactions with associated value.
    """
    G = nx.DiGraph()

    for _, row in df.iterrows():
        from_node = row['from_address']
        to_node = row['to_address']
        value = row['value']

        if G.has_edge(from_node, to_node):
            G[from_node][to_node]['weight'] += value
        else:
            G.add_edge(from_node, to_node, weight=value)

    return G

def filter_isolated_nodes(G, min_degree=1):
    """
    Filter out nodes with degree lower than a specified threshold to remove isolated or low-activity nodes.
    """
    degree_dict = dict(G.degree())
    non_isolated_nodes = [node for node, degree in degree_dict.items() if degree >= min_degree]
    return G.subgraph(non_isolated_nodes)

def simplify_node_labels(G):
    """
    Simplify node labels by displaying only the first few characters to reduce clutter in visualization.
    """
    label_mapping = {node: node[:5] for node in G.nodes()}
    return label_mapping

def visualize_filtered_graph(G, df):
    """
    Visualize the filtered transaction graph with log-scaled edge colors and thickness based on transaction value.
    """
    # Calculate total transaction volume per address (outgoing transactions only)
    node_transaction_value = df.groupby('from_address')['value'].sum()

    # Select the top 500 nodes by transaction volume
    top_nodes = node_transaction_value.nlargest(500).index
    subgraph = G.subgraph(top_nodes)

    # Further filter to nodes with degree > 1
    filtered_subgraph = filter_isolated_nodes(subgraph, min_degree=2)

    # Extract and log-scale the edge weights
    raw_weights = [filtered_subgraph[u][v]['weight'] for u, v in filtered_subgraph.edges()]
    log_weights = [np.log10(w + 1) for w in raw_weights]  # Add 1 to avoid log(0)

    # Normalize and map edge colors using a color map
    norm = plt.Normalize(min(log_weights), max(log_weights))
    cmap = plt.cm.Reds
    edge_colors = [cmap(norm(w)) for w in log_weights]

    # Set edge widths proportional to log-scaled weights
    edge_widths = [0.25 * w for w in log_weights]

    # Generate node positions using spring layout
    pos = nx.spring_layout(filtered_subgraph, k=0.15, iterations=20)

    # Simplify node labels
    label_mapping = simplify_node_labels(filtered_subgraph)

    # Plot the graph
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(filtered_subgraph, pos, node_size=50, node_color='blue', alpha=0.6)
    nx.draw_networkx_edges(filtered_subgraph, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    nx.draw_networkx_labels(filtered_subgraph, pos, labels=label_mapping, font_size=10, font_color='black')

    plt.title("Ethereum Transaction Flow Graph (Top 500 Nodes with >1 Connections, Log-Scaled Colors)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Load transaction data
    df = load_transaction_data('Dataset.csv')

    # Build the transaction graph
    G = build_transaction_graph(df)

    # Visualize the transaction graph
    visualize_filtered_graph(G, df)
