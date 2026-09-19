import pathlib
import json
from music21 import converter, note, chord


def parse_midi_file(midi_path: str):
    """Parse a single MIDI file into a list of musical events.

    This function now includes comprehensive error handling:
    * Checks that the file exists.
    * Catches exceptions raised by ``music21.converter.parse``.
    * Wraps any parsing error in a ``RuntimeError`` with a clear message.
    The returned value is a list of event dictionaries as before.
    """
    # Validate file existence
    midi_file = pathlib.Path(midi_path)
    if not midi_file.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    try:
        score = converter.parse(str(midi_file))
    except Exception as e:
        raise RuntimeError(f"Failed to parse MIDI file '{midi_path}': {e}")

    events = []
    for elem in score.flat.notesAndRests:
        if isinstance(elem, note.Note):
            events.append({
                "type": "note",
                "pitches": [elem.pitch.nameWithOctave],
                "offset": float(elem.offset),
                "duration": float(elem.quarterLength)
            })
        elif isinstance(elem, chord.Chord):
            events.append({
                "type": "chord",
                "pitches": [p.nameWithOctave for p in elem.pitches],
                "offset": float(elem.offset),
                "duration": float(elem.quarterLength)
            })
    return events


def parse_all(raw_dir: str, processed_dir: str, manifest_path: str):
    """Parse all MIDI files in raw_dir, write JSON events to processed_dir, and record a manifest.
    """
    raw_path = pathlib.Path(raw_dir)
    processed_path = pathlib.Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    files_processed = 0
    files_skipped = 0
    for midi_file in raw_path.rglob('*.mid'):
        try:
            events = parse_midi_file(str(midi_file))
            status = "success"
            error_msg = None
        except Exception as e:
            events = []
            status = "error"
            error_msg = str(e)
        out_dict = {
            "source_file": midi_file.name,
            "status": status,
            "events": events
        }
        if error_msg:
            out_dict["error"] = error_msg
        out_file = processed_path / (midi_file.stem + '.json')
        out_file.write_text(json.dumps(out_dict, indent=2))
        manifest_entries.append({
            "midi": str(midi_file),
            "json": str(out_file),
            "status": status,
            "error": error_msg
        })
        if status == "success":
            files_processed += 1
        else:
            files_skipped += 1
    manifest = {
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "entries": manifest_entries
    }
    pathlib.Path(manifest_path).write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print('Usage: python parser.py <raw_dir> <processed_dir> <manifest_path>')
        sys.exit(1)
    parse_all(sys.argv[1], sys.argv[2], sys.argv[3])
