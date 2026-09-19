import os
import json
from pathlib import Path

from music21 import converter, instrument, note, chord

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'raw'
PROCESSED_DIR = Path(__file__).resolve().parents[1] / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def _extract_notes(midi_stream):
    """Convert a music21 stream into a list of note dictionaries.
    Each dict contains:
        - pitch (MIDI number)
        - offset (beat offset from start)
        - duration (quarterLength)
        - is_chord (bool)
    """
    notes = []
    for element in midi_stream.recurse():
        if isinstance(element, note.Note):
            notes.append({
                "pitch": element.pitch.midi,
                "offset": float(element.offset),
                "duration": float(element.quarterLength),
                "is_chord": False,
            })
        elif isinstance(element, chord.Chord):
            # Represent each chord as a list of pitches; keep same offset/duration
            notes.append({
                "pitch": [p.midi for p in element.pitches],
                "offset": float(element.offset),
                "duration": float(element.quarterLength),
                "is_chord": True,
            })
    return notes

def parse_and_save():
    midi_files = list(RAW_DATA_DIR.rglob('*.mid'))
    if not midi_files:
        print('No MIDI files found in', RAW_DATA_DIR)
        return
    for midi_path in midi_files:
        try:
            stream = converter.parse(midi_path)
            # Keep only piano parts if multiple instruments
            parts = instrument.partitionByInstrument(stream)
            if parts:  # select piano or first part
                stream = parts.parts[0]
            notes = _extract_notes(stream)
            out_path = PROCESSED_DIR / (midi_path.stem + '.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(notes, f, ensure_ascii=False, indent=2)
            print(f'Processed {midi_path.name} -> {out_path.name}')
        except Exception as e:
            print(f'Failed to process {midi_path.name}: {e}')

if __name__ == '__main__':
    parse_and_save()
