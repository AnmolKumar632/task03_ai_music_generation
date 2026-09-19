import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


def token_from_event(event: dict) -> str:
    """Converts a parsed event dict into a consistent string token."""
    pitch = event.get("pitch")
    if isinstance(pitch, list):
        return ".".join(str(p) for p in sorted(pitch))
    return str(pitch)


def load_processed_sequences(data_dir: Path) -> List[List[str]]:
    """Loads all JSON files from the processed directory and extracts token sequences."""
    sequences = []
    json_files = list(data_dir.rglob("*.json"))
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data:
                continue
            seq = [token_from_event(evt) for evt in data if "pitch" in evt]
            if seq:
                sequences.append(seq)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    return sequences


def build_vocab(sequences: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Builds token to ID and ID to token mappings from sequence data."""
    unique_tokens = sorted(list(set(token for seq in sequences for token in seq)))
    # Reserve 0 for padding if needed, start tokens from 1 or 0
    # Here 0 is <PAD>
    token_to_idx = {"<PAD>": 0}
    for idx, token in enumerate(unique_tokens, start=1):
        token_to_idx[token] = idx
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    return token_to_idx, idx_to_token


class MusicDataset(Dataset):
    """Dataset creating input-target window pairs for next-token prediction."""

    def __init__(self, sequences: List[List[str]], token_to_idx: Dict[str, int], seq_len: int = 16):
        self.seq_len = seq_len
        self.samples: List[Tuple[List[int], int]] = []

        for seq in sequences:
            indices = [token_to_idx.get(tok, 0) for tok in seq]
            if not indices:
                continue

            # Prepend padding so early tokens and short sequences also produce training targets
            padded = [0] * seq_len + indices
            for i in range(len(indices)):
                input_chunk = padded[i : i + seq_len]
                target_token = indices[i]
                self.samples.append((input_chunk, target_token))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp, tgt = self.samples[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def create_dataloaders(
    data_dir: Path,
    seq_len: int = 64,
    batch_size: int = 32,
    train_split: float = 0.85
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    sequences = load_processed_sequences(data_dir)
    if not sequences:
        raise ValueError(f"No valid note sequences found in {data_dir}. Ensure MIDI files were parsed into JSON.")

    token_to_idx, idx_to_token = build_vocab(sequences)

    # Train/val split by sequences
    split_idx = max(1, int(len(sequences) * train_split))
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:] if split_idx < len(sequences) else sequences[:split_idx]

    train_ds = MusicDataset(train_seqs, token_to_idx, seq_len=seq_len)
    val_ds = MusicDataset(val_seqs, token_to_idx, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, token_to_idx, idx_to_token
