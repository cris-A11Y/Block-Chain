import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_processing import load_and_process_data

# Define the encoder architecture for the Graph Autoencoder
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# Define the unsupervised Graph Autoencoder model
class FlashloanGAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FlashloanGAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.gae = GAE(self.encoder)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_index):
        return self.gae.decode(z, edge_index)

# Load the processed data (see data_processing.py for implementation)
def load_data(filepath):
    data, num_nodes = load_and_process_data(filepath)
    return data

# Train the unsupervised GAE model
def train_model(data):
    model = FlashloanGAE(input_dim=1, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.gae.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    return model

# Perform clustering analysis using KMeans on node embeddings to detect flashloan-like behavior
def detect_flashloans(model, data):
    model.eval()
    # Get node embeddings
    z = model.encode(data.x, data.edge_index).detach().cpu().numpy()

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=42).fit(z)
    labels = kmeans.labels_

    # Visualize embeddings using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(z)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='coolwarm', s=50)
    plt.title("PCA of Node Embeddings (KMeans Clustering)")
    plt.show()

    return labels

# Main function: load data, train model, perform clustering, and identify suspicious addresses
def main(filepath):
    data = load_data(filepath)
    model = train_model(data)
    labels = detect_flashloans(model, data)

    # Print potentially flashloan-related addresses (assuming Cluster 1 indicates suspicious nodes)
    print("Potential Flashloan Addresses (Cluster 1):")
    address_mapping = data.address_mapping

    for i, label in enumerate(labels):
        if label == 1:
            address = address_mapping[i]
            print(f"Address {address} might be a flashloan-related address.")

if __name__ == "__main__":
    filepath = 'Dataset.csv'  # Your dataset file path
    main(filepath)
