import os
import sys

# Reconfigure stdout to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import time

print("1. Imports completed")
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, set_seed
from train_classifier import JobLevelDataset, collate_fn, OrdinalLoss, evaluate

print("2. Loading config...")
cfg = load_config("config.yaml")

print("3. Model and Tokenizer initialization (FIRST)...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6,
    ignore_mismatched_sizes=True,
)
model = model.to("cpu")
print("4. Model loaded and moved to CPU successfully!")

print("5. Loading dataset (SECOND)...")
dataset_path = "cleaned_dataset.json"  # local path
with open(dataset_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
print(f"   Loaded {len(raw_data)} raw samples")

# Map levels
level_labels = cfg["classifier"]["level_labels"]
valid_levels = set(lv.upper() for lv in level_labels)
data = []
for d in raw_data:
    if not d.get("text", "").strip():
        continue
    lvl = str(d.get("level", "")).upper().strip()
    if lvl in ["LEAD", "PRINCIPAL", "ARCHITECT", "DIRECTOR", "MANAGER"]:
        mapped_lvl = "LEAD_PLUS"
    elif lvl in ["INTERN", "FRESHER", "JUNIOR", "MIDDLE", "SENIOR"]:
        mapped_lvl = lvl
    else:
        continue
    if mapped_lvl in valid_levels:
        d["level"] = mapped_lvl
        data.append(d)

print(f"   Filtered to {len(data)} valid samples")
val_ratio = cfg["data"].get("val_ratio", 0.2)
n_val = max(1, int(len(data) * val_ratio))
import random
random.seed(42)
random.shuffle(data)
val_data = data[:n_val]
train_data = data[n_val:]
print(f"   Split train: {len(train_data)} | val: {len(val_data)}")

print("6. Dataset initialization...")
train_dataset = JobLevelDataset(
    train_data, tokenizer, level_labels, 512, "head+tail"
)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
)

print("7. Calculating class weights...")
from sklearn.utils.class_weight import compute_class_weight
train_labels = [s[1] for s in train_dataset.samples]
unique_train_labels = np.unique(train_labels)
computed_weights = compute_class_weight(
    class_weight='balanced',
    classes=unique_train_labels,
    y=train_labels
)
class_weights = torch.tensor(computed_weights, dtype=torch.float).to("cpu")

print("8. Optimizer and scheduler setup...")
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 1  # 1 epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

print("9. Loss function setup...")
loss_fct = OrdinalLoss(class_weights=class_weights, level_labels=level_labels, lambda_penalty=1.0).to("cpu")

print("10. Starting 1 batch training step...")
model.train()
global_step = 0
for batch in train_loader:
    global_step += 1
    input_ids = batch["input_ids"].to("cpu")
    attention_mask = batch["attention_mask"].to("cpu")
    labels = batch["labels"].to("cpu")
    
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits
    loss = loss_fct(logits, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    print("11. Step completed. Loss:", loss.item())
    break
        
print("12. Debug finished successfully!")
