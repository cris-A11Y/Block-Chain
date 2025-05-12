import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from data_processing import load_and_process_data
import os


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def train_and_evaluate(filepath, save_path='fraud_address_detection_model_checkpoint/gcn_model.pt'):
    """
    Train a simple 2-layer GCN on the processed Ethereum transaction graph and evaluate.

    Args:
        filepath (str): Path to the CSV file.
        save_path (str): Path to save the model checkpoint.
    """
    # Load and process data
    data, num_nodes = load_and_process_data(filepath)

    # Create masks (use only labeled nodes for supervised learning)
    mask = data.y != -1
    train_mask = torch.rand(len(data.y)) < 0.8
    train_mask = train_mask & mask
    test_mask = ~train_mask & mask

    # Initialize GCN model
    model = GCN(input_dim=1, hidden_dim=16, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # Save model checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Evaluation
    model.eval()
    out = model(data)
    pred = out[test_mask].argmax(dim=1)

    # Map predictions back to original Ethereum addresses
    node_ids = torch.where(test_mask)[0]
    address_mapping = data.address_mapping

    predicted_addresses = []
    for node_id, label in zip(node_ids, pred):
        address = address_mapping[node_id.item()]
        predicted_addresses.append((address, label.item()))

    # Display some predictions
    print("\nSample Predictions:")
    for address, label in predicted_addresses[:20]:  # Limit to first 20
        print(f"Address: {address}, Predicted Label: {label}")

    # Calculate accuracy
    acc = (pred == data.y[test_mask]).sum().item() / test_mask.sum().item()
    print(f'Test Accuracy: {acc:.4f}')


if __name__ == "__main__":
    filepath = 'Dataset.csv'
    train_and_evaluate(filepath)
