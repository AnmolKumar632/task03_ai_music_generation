import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

from training.data_loader import create_dataloaders
from training.model import MusicLSTM


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, criterion, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0.0
    count = 0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for idx, (inputs, targets) in enumerate(progress):
        if max_batches and idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / count if count > 0 else 0.0


def validate(model, dataloader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            if max_batches and idx >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            count += 1

    return total_loss / count if count > 0 else 0.0


def run_training(config_path: str = "training/config.yaml", max_batches: int = None, override_epochs: int = None):
    # Load config
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    data_dir = Path(cfg.get("data_dir", "data/processed"))
    model_dir = Path(cfg.get("model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    seq_len = cfg.get("seq_len", 64)
    batch_size = cfg.get("batch_size", 32)
    embed_dim = cfg.get("embed_dim", 128)
    hidden_dim = cfg.get("hidden_dim", 256)
    num_layers = cfg.get("num_layers", 2)
    dropout = cfg.get("dropout", 0.2)
    learning_rate = cfg.get("learning_rate", 0.001)
    epochs = override_epochs if override_epochs is not None else cfg.get("epochs", 30)
    seed = cfg.get("seed", 42)
    use_gpu = cfg.get("use_gpu", True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Using device: {device}")

    # Prepare dataset & vocabulary
    print(f"Loading data from {data_dir}...")
    train_loader, val_loader, token_to_idx, idx_to_token = create_dataloaders(
        data_dir=data_dir,
        seq_len=seq_len,
        batch_size=batch_size,
        train_split=cfg.get("train_split", 0.85)
    )

    vocab_size = len(token_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")

    # Model instantiation
    model = MusicLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        padding_idx=0
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model_path = model_dir / "best_music_model.pt"
    vocab_path = model_dir / "vocab.json"

    # Save vocabulary dictionary for inference/generation
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({
            "token_to_idx": token_to_idx,
            "idx_to_token": {str(k): v for k, v in idx_to_token.items()},
            "seq_len": seq_len
        }, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}")

    # Training Loop
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, max_batches=max_batches)
        val_loss = validate(model, val_loader, criterion, device, max_batches=max_batches)

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss and val_loss > 0:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
                "config": {
                    "vocab_size": vocab_size,
                    "embed_dim": embed_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "seq_len": seq_len
                }
            }, best_model_path)
            print(f"==> Saved new best model checkpoint to {best_model_path}")

    print("Training process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Music Generation PyTorch Model")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit batches per epoch for testing")
    args = parser.parse_args()

    run_training(config_path=args.config, max_batches=args.max_batches, override_epochs=args.epochs)
