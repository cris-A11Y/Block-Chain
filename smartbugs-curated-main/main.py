import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizerFast
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data_processing import load_smartbugs_with_line_mask

# Dataset wrapper for SmartBugs with line mask support
class SmartBugsDataset(Dataset):
    def __init__(self, input_ids, attention_masks, line_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.line_masks = line_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'line_mask': self.line_masks[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# CodeBERT-based classifier
class CodeBERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(CodeBERTClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token representation
        return self.classifier(cls_output)

# Training function
def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    return acc

# Main execution
if __name__ == "__main__":
    dataset_root = os.path.join(".", "dataset")
    input_ids, attention_masks, line_masks, labels, label2id, id2label = load_smartbugs_with_line_mask(dataset_root)

    # Split into training and validation sets
    total_samples = len(labels)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    dataset = SmartBugsDataset(input_ids, attention_masks, line_masks, labels)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CodeBERTClassifier(num_labels=len(label2id)).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(100):
        print(f"\nEpoch {epoch + 1}")
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

    # Save the trained model
    os.makedirs("checkpoint", exist_ok=True)
    save_path = os.path.join("checkpoint", "codebert_classifier.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'label2id': label2id,
        'id2label': id2label
    }, save_path)
    print(f"Model saved to {save_path}")
