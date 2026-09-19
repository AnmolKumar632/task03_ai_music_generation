# NeuralHarmony — AI Music Studio

<p align="center">
  <img src="assets/neuralharmony-banner.svg" alt="NeuralHarmony banner" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Flask-Web%20API-000000?logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/music21-MIDI%20Parsing-6A5ACD" alt="music21" />
  <img src="https://img.shields.io/badge/WebAudio-Synth-FF6B6B" alt="Web Audio" />
</p>

NeuralHarmony is an end-to-end AI music generation project that converts MIDI data into new symbolic compositions using a PyTorch LSTM model and presents the output in a modern interactive browser-based music studio.

## Highlights

- Symbolic music generation with LSTM-based sequence learning
- MIDI parsing and processing using `music21`
- Composition generation with temperature-based creativity control
- Browser UI with virtual piano seed prompts
- Real-time piano-roll visualization
- Web Audio synthesis with reverb and delay effects
- MIDI and WAV export
- Automated tests for pipeline and API validation

## Architecture

```mermaid
flowchart LR
    A[Raw MIDI Files] --> B[Music21 Parsing]
    B --> C[Tokenized Sequences]
    C --> D[Dataset + Vocab]
    D --> E[MusicLSTM Training]
    E --> F[Generated Tokens]
    F --> G[Flask API]
    G --> H[Interactive Studio]
    H --> I[Web Audio Playback]
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/parse_midi.py
python -m training.train_model
python app.py
```

Open: http://127.0.0.1:5000

## Project Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── data/
├── generated/
├── models/
├── music/
├── scripts/
├── static/
├── templates/
├── tests/
├── training/
└── assets/
```

## Tech Stack

- Python
- PyTorch
- Flask
- music21
- HTML / CSS / JavaScript
- Web Audio API

## Status

Project is working end-to-end with model generation, web studio playback, and export flow validated.
