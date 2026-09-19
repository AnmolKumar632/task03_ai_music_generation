import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from music21 import stream, note, chord, instrument

import sys

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.model import MusicLSTM


def sample_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """Samples next token from logits using temperature scaling."""
    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    predicted_idx = torch.multinomial(probabilities, num_samples=1).item()
    return int(predicted_idx)


def tokens_to_midi(tokens, output_file: Path, step_duration: float = 0.5):
    """Converts token strings into a music21 stream and writes a MIDI file."""
    midi_stream = stream.Stream()
    midi_stream.append(instrument.Piano())

    current_offset = 0.0

    for tok in tokens:
        if tok == "<PAD>":
            current_offset += step_duration
            continue

        # Check if it's a chord (represented as pitches joined by dots, e.g. "60.64.67")
        if "." in tok:
            try:
                chord_pitches = [int(p) for p in tok.split(".") if p.isdigit()]
                if chord_pitches:
                    new_chord = chord.Chord(chord_pitches)
                    new_chord.quarterLength = step_duration
                    new_chord.offset = current_offset
                    midi_stream.append(new_chord)
            except Exception as e:
                print(f"Skipping chord token {tok}: {e}")
        else:
            try:
                pitch_val = int(tok)
                new_note = note.Note(pitch_val)
                new_note.quarterLength = step_duration
                new_note.offset = current_offset
                midi_stream.append(new_note)
            except Exception as e:
                print(f"Skipping note token {tok}: {e}")

        current_offset += step_duration

    output_file.parent.mkdir(parents=True, exist_ok=True)
    midi_stream.write("midi", fp=str(output_file))
    print(f"Generated MIDI saved successfully to: {output_file}")


def generate(
    model_path: str = "models/best_music_model.pt",
    vocab_path: str = "models/vocab.json",
    output_path: str = "generated/output.mid",
    num_notes: int = 100,
    temperature: float = 0.9,
    seed_token: str = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_meta = json.load(f)

    token_to_idx = vocab_meta["token_to_idx"]
    idx_to_token = {int(k): v for k, v in vocab_meta["idx_to_token"].items()}
    seq_len = vocab_meta.get("seq_len", 64)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint.get("config", {})

    model = MusicLSTM(
        vocab_size=cfg.get("vocab_size", len(token_to_idx)),
        embed_dim=cfg.get("embed_dim", 128),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
        padding_idx=0
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Initialise sequence seed
    available_tokens = [t for t in token_to_idx.keys() if t != "<PAD>"]
    if seed_token and seed_token in token_to_idx:
        start_token = seed_token
    else:
        start_token = random.choice(available_tokens)

    current_sequence = [token_to_idx[start_token]]
    # Pad to seq_len
    if len(current_sequence) < seq_len:
        current_sequence = [0] * (seq_len - len(current_sequence)) + current_sequence

    generated_tokens = [start_token]

    print(f"Generating {num_notes} notes with temperature {temperature}...")

    with torch.no_grad():
        for _ in range(num_notes):
            input_tensor = torch.tensor([current_sequence[-seq_len:]], dtype=torch.long, device=device)
            logits, _ = model(input_tensor)
            # Mask out <PAD> (index 0) to avoid generating silence
            logits[0, 0] = -1e9

            next_idx = sample_with_temperature(logits[0], temperature=temperature)
            next_token = idx_to_token.get(next_idx, start_token)

            generated_tokens.append(next_token)
            current_sequence.append(next_idx)

    tokens_to_midi(generated_tokens, Path(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MIDI music using trained PyTorch model")
    parser.add_argument("--model", type=str, default="models/best_music_model.pt", help="Path to trained model")
    parser.add_argument("--vocab", type=str, default="models/vocab.json", help="Path to vocab JSON")
    parser.add_argument("--output", type=str, default="generated/sample_music.mid", help="Output MIDI file path")
    parser.add_argument("--num_notes", type=int, default=120, help="Number of notes to generate")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature (0.5=conservative, 1.2=creative)")
    parser.add_argument("--seed_token", type=str, default=None, help="Optional initial seed token/pitch")
    args = parser.parse_args()

    generate(
        model_path=args.model,
        vocab_path=args.vocab,
        output_path=args.output,
        num_notes=args.num_notes,
        temperature=args.temperature,
        seed_token=args.seed_token
    )
