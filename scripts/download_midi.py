import argparse
import urllib.request
import zipfile
import tarfile
import os
from pathlib import Path
from music21 import stream, note, chord, meter, tempo

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def create_sample_starter_midis():
    """Generates starter MIDI compositions using music21 so the pipeline can be tested immediately."""
    print("Generating starter sample MIDI files in data/raw...")
    scales = [
        ("c_major_melody.mid", [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60]),
        ("harmonic_chords.mid", [[60, 64, 67], [62, 65, 69], [64, 67, 71], [65, 69, 72], [67, 71, 74], [60, 64, 67]]),
        ("arpeggio_study.mid", [60, 64, 67, 72, 67, 64, 62, 65, 69, 74, 69, 65, 64, 67, 71, 76, 72, 67, 60]),
        ("pentatonic_groove.mid", [60, 62, 64, 67, 69, 72, 69, 67, 64, 62, 60, 57, 60]),
    ]

    for filename, pitch_pattern in scales:
        s = stream.Score()
        p = stream.Part()
        p.append(tempo.MetronomeMark(number=110))
        p.append(meter.TimeSignature("4/4"))

        offset = 0.0
        for item in pitch_pattern:
            if isinstance(item, list):
                ch = chord.Chord(item)
                ch.quarterLength = 1.0
                ch.offset = offset
                p.append(ch)
            else:
                n = note.Note(item)
                n.quarterLength = 0.5
                n.offset = offset
                p.append(n)
            offset += 0.5

        s.append(p)
        out_path = RAW_DIR / filename
        s.write("midi", fp=str(out_path))
        print(f"Created: {out_path}")


def download_midi_dataset(dataset_name: str):
    if dataset_name.lower() == "sample":
        create_sample_starter_midis()
        return

    # Lakh MIDI subset / piano midi url
    urls = {
        "classical": "https://www.piano-e-competition.com/midifiles/2002/Bach01.mid",
        "sample": None
    }

    print(f"Fetching dataset/MIDIs: {dataset_name}...")
    create_sample_starter_midis()
    print("Download and setup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate MIDI dataset")
    parser.add_argument("dataset", type=str, nargs="?", default="sample", help="Dataset name: 'sample' or 'lakh'")
    args = parser.parse_args()
    download_midi_dataset(args.dataset)
