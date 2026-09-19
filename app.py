import json
import os
import random
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
import torch
import torch.nn.functional as F
from music21 import stream, note, chord, instrument

# Ensure root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.model import MusicLSTM

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

MODEL_PATH = PROJECT_ROOT / "models" / "best_music_model.pt"
VOCAB_PATH = PROJECT_ROOT / "models" / "vocab.json"
GENERATED_DIR = PROJECT_ROOT / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Global cached model and vocab
cached_model = None
cached_vocab = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_vocab():
    global cached_model, cached_vocab
    if cached_model is not None and cached_vocab is not None:
        return cached_model, cached_vocab

    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        return None, None

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        cached_vocab = json.load(f)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    cfg = checkpoint.get("config", {})

    token_to_idx = cached_vocab["token_to_idx"]
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

    cached_model = model
    return cached_model, cached_vocab


def sample_token(logits: torch.Tensor, temperature: float = 0.85) -> int:
    if temperature <= 0.05:
        return int(torch.argmax(logits, dim=-1).item())
    scaled = logits / max(0.1, temperature)
    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    model, vocab = load_model_and_vocab()
    has_model = model is not None and vocab is not None
    vocab_size = len(vocab["token_to_idx"]) if vocab else 0

    return jsonify({
        "status": "ready" if has_model else "model_missing",
        "device": str(device),
        "vocab_size": vocab_size,
        "model_file": str(MODEL_PATH.name) if MODEL_PATH.exists() else None,
        "available_tokens": [k for k in (vocab["token_to_idx"].keys() if vocab else []) if k != "<PAD>"][:20]
    })


@app.route("/api/generate", methods=["POST"])
def generate_music():
    model, vocab = load_model_and_vocab()
    if model is None or vocab is None:
        return jsonify({"error": "Model or vocabulary file not found. Please train the model first."}), 400

    data = request.get_json() or {}
    num_notes = min(max(int(data.get("num_notes", 60)), 10), 300)
    temperature = max(0.1, min(float(data.get("temperature", 0.85)), 1.8))
    seed_token = data.get("seed_token", None)
    seed_notes = data.get("seed_notes", None)  # Optional array from virtual piano
    step_duration = float(data.get("step_duration", 0.5))
    instrument_type = data.get("instrument", "piano")

    token_to_idx = vocab["token_to_idx"]
    idx_to_token = {int(k): v for k, v in vocab["idx_to_token"].items()}
    seq_len = vocab.get("seq_len", 16)

    # Initial tokens handling
    available = [t for t in token_to_idx.keys() if t != "<PAD>"]
    if not available:
        return jsonify({"error": "Vocabulary contains no valid pitch tokens."}), 500

    current_seq_tokens = []
    if seed_notes and isinstance(seed_notes, list) and len(seed_notes) > 0:
        for s in seed_notes:
            s_str = str(s)
            if s_str in token_to_idx and s_str != "<PAD>":
                current_seq_tokens.append(s_str)
            else:
                # Fallback to nearest or random available
                current_seq_tokens.append(random.choice(available))
    elif seed_token and str(seed_token) in token_to_idx and str(seed_token) != "<PAD>":
        current_seq_tokens.append(str(seed_token))
    else:
        current_seq_tokens.append(random.choice(available))

    current_seq = [token_to_idx[t] for t in current_seq_tokens]
    if len(current_seq) < seq_len:
        current_seq = [0] * (seq_len - len(current_seq)) + current_seq

    generated_tokens = list(current_seq_tokens)

    with torch.no_grad():
        for _ in range(num_notes):
            inp = torch.tensor([current_seq[-seq_len:]], dtype=torch.long, device=device)
            logits, _ = model(inp)
            logits[0, 0] = -1e9  # Suppress <PAD>

            next_idx = sample_token(logits[0], temperature=temperature)
            next_token = idx_to_token.get(next_idx, random.choice(available))

            generated_tokens.append(next_token)
            current_seq.append(next_idx)

    # Build music21 stream and note timeline for client-side audio synth
    midi_stream = stream.Stream()
    if instrument_type == "electric_piano":
        midi_stream.append(instrument.ElectricPiano())
    elif instrument_type == "synth_lead":
        inst = instrument.Instrument()
        inst.midiProgram = 80  # GM Lead 1 (Square)
        inst.instrumentName = "Synth Lead"
        midi_stream.append(inst)
    elif instrument_type == "strings":
        midi_stream.append(instrument.StringInstrument())
    else:
        midi_stream.append(instrument.Piano())

    client_notes = []
    current_time = 0.0

    for tok in generated_tokens:
        if tok == "<PAD>":
            current_time += step_duration
            continue

        if "." in tok:
            # Chord
            pitches = [int(p) for p in tok.split(".") if p.isdigit()]
            if pitches:
                ch = chord.Chord(pitches)
                ch.quarterLength = step_duration
                ch.offset = current_time
                midi_stream.append(ch)

                client_notes.append({
                    "pitches": pitches,
                    "time": current_time,
                    "duration": step_duration,
                    "is_chord": True,
                    "label": f"Chord ({', '.join(str(p) for p in pitches)})"
                })
        else:
            # Single note
            try:
                pitch_val = int(tok)
                n = note.Note(pitch_val)
                n.quarterLength = step_duration
                n.offset = current_time
                midi_stream.append(n)

                client_notes.append({
                    "pitches": [pitch_val],
                    "time": current_time,
                    "duration": step_duration,
                    "is_chord": False,
                    "label": n.nameWithOctave
                })
            except Exception:
                pass

        current_time += step_duration

    # Save to generated directory with timestamp
    timestamp = int(time.time())
    filename = f"composition_{timestamp}.mid"
    output_path = GENERATED_DIR / filename
    midi_stream.write("midi", fp=str(output_path))

    return jsonify({
        "success": True,
        "filename": filename,
        "download_url": f"/download/{filename}",
        "total_notes": len(client_notes),
        "total_duration": round(current_time, 2),
        "notes": client_notes
    })


@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)


@app.route("/api/history", methods=["GET"])
def get_history():
    files = list(GENERATED_DIR.glob("*.mid"))
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    results = []
    for f in files[:10]:
        results.append({
            "filename": f.name,
            "download_url": f"/download/{f.name}",
            "size_kb": round(f.stat().st_size / 1024, 2),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime))
        })
    return jsonify({"history": results})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
