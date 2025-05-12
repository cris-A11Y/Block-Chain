import torch
import torch.nn.functional as F
from fraud_address_detection_model import GCN
from data_processing import load_and_process_data


def predict_address(filepath, model_path, address):
    # Load data
    data, _ = load_and_process_data(filepath)

    # Get address mapping dictionary
    address_mapping = data.address_mapping

    # Check if the address exists
    if address not in address_mapping.values():
        print(f"Address {address} not found in the dataset.")
        return

    # Get the corresponding node index
    node_idx = [key for key, value in address_mapping.items() if value == address][0]

    # Initialize the model and load parameters
    model = GCN(input_dim=1, hidden_dim=16, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Model inference
    with torch.no_grad():
        output = model(data)
        logits = output[node_idx]
        probs = F.softmax(logits, dim=0)
        predicted_class = torch.argmax(probs).item()

        print(f"Address {address} predicted class: {predicted_class}")
        print(f"Class probabilities: {probs.tolist()}")


if __name__ == "__main__":
    filepath = "Dataset.csv"
    model_path = "fraud_address_detection_model_checkpoint/gcn_model.pt"

    try:
        address = input("Please enter the address to predict (e.g., '0x1234abcd'): ")
        predict_address(filepath, model_path, address)
    except ValueError:
        print("Invalid input. Please enter a valid address.")
