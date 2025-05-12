import os
import json
import torch
from transformers import RobertaTokenizerFast

def lines_to_token_mask(code: str, lines: list[int], tokenizer: RobertaTokenizerFast, max_length: int = 256):
    """
    Generate a binary mask for each token indicating whether the token belongs to the vulnerability lines.
    """
    # Split by lines and record the line number for each character
    src_lines = code.splitlines(keepends=True)
    char2line = []
    for idx, ln in enumerate(src_lines, start=1):
        char2line += [idx] * len(ln)

    # Tokenize and get offset mapping
    encoding = tokenizer(
        code,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_offsets_mapping=True
    )
    offsets = encoding.pop("offset_mapping")
    line_mask = []
    line_set = set(lines)
    for (st, ed) in offsets:
        if st < len(char2line):
            token_lines = set(char2line[st:min(ed, len(char2line))])
            # Mark 1 if there is overlap with vulnerability lines
            line_mask.append(int(bool(token_lines & line_set)))
        else:
            line_mask.append(0)
    return line_mask

def load_smartbugs_with_line_mask(dataset_root: str, max_length: int = 256):
    """
    Load SmartBugs Curated dataset and generate:
      - input_ids: List[torch.Tensor]
      - attention_masks: List[torch.Tensor]
      - line_masks: List[torch.Tensor]
      - labels: List[int]
      - label2id: dict
      - id2label: dict
    """
    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
    vuln_path = os.path.join(dataset_root, "vulnerabilities.json")
    with open(vuln_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    data = []
    for rec in records:
        rel_path = rec['path'].split("dataset/", 1)[1]
        sol_path = os.path.join(dataset_root, rel_path)
        if not os.path.isfile(sol_path):
            continue
        with open(sol_path, 'r', encoding='utf-8') as sf:
            code = sf.read()
        vuln_info = rec['vulnerabilities'][0]
        lines = vuln_info['lines']
        category = vuln_info['category']
        data.append((code, lines, category))

    categories = sorted({cat for _, _, cat in data})
    label2id = {cat: idx for idx, cat in enumerate(categories)}
    id2label = {idx: cat for cat, idx in label2id.items()}

    input_ids, attention_masks, line_masks, labels = [], [], [], []
    for code, lines, category in data:
        enc = tokenizer(
            code,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        lm = lines_to_token_mask(code, lines, tokenizer, max_length=max_length)
        input_ids.append(enc['input_ids'].squeeze(0))
        attention_masks.append(enc['attention_mask'].squeeze(0))
        line_masks.append(torch.tensor(lm, dtype=torch.long))
        labels.append(label2id[category])

    return input_ids, attention_masks, line_masks, labels, label2id, id2label

def preprocess_single_file(filepath, tokenizer=None, max_length=512):
    """
    Process a single .sol file into input_ids and attention_mask for CodeBERT.
    """
    if tokenizer is None:
        tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    tokens = tokenizer(
        code,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    return tokens['input_ids'][0], tokens['attention_mask'][0]

if __name__ == "__main__":
    dataset_root = os.path.join(".", "dataset")
    X_ids, X_attn, X_line_masks, y, label2id, id2label = load_smartbugs_with_line_mask(dataset_root)

    print(f"Loaded {len(y)} samples")
    print("Label2ID:", label2id)

    for i in range(min(3, len(y))):
        print(f"\nSample {i}:")
        print("Label ID:", y[i], "Category:", id2label[y[i]])
        print("Input IDs shape:", X_ids[i].shape)
        print("Line mask sum:", X_line_masks[i].sum().item(), "marked tokens")
