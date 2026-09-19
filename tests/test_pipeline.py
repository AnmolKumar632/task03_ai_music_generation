import unittest
import tempfile
import shutil
import json
from pathlib import Path
import torch

from training.model import MusicLSTM
from training.data_loader import build_vocab, MusicDataset, token_from_event
from scripts.generate_midi import tokens_to_midi


class TestMusicGenerationPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_token_conversion(self):
        single_note_evt = {"pitch": 60, "offset": 0.0, "duration": 1.0, "is_chord": False}
        chord_evt = {"pitch": [64, 60, 67], "offset": 1.0, "duration": 1.0, "is_chord": True}

        self.assertEqual(token_from_event(single_note_evt), "60")
        self.assertEqual(token_from_event(chord_evt), "60.64.67")

    def test_vocab_and_dataset(self):
        sequences = [
            ["60", "62", "64", "65", "67"],
            ["60.64.67", "62.65.69", "60"]
        ]
        token_to_idx, idx_to_token = build_vocab(sequences)

        self.assertIn("<PAD>", token_to_idx)
        self.assertEqual(token_to_idx["<PAD>"], 0)
        self.assertIn("60", token_to_idx)
        self.assertIn("60.64.67", token_to_idx)

        dataset = MusicDataset(sequences, token_to_idx, seq_len=4)
        self.assertGreater(len(dataset), 0)

        inp, tgt = dataset[0]
        self.assertEqual(inp.shape[0], 4)
        self.assertIsInstance(tgt.item(), int)

    def test_model_forward_pass(self):
        vocab_size = 20
        seq_len = 8
        batch_size = 4

        model = MusicLSTM(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_layers=1)
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits, _ = model(dummy_input)
        self.assertEqual(logits.shape, (batch_size, vocab_size))

    def test_tokens_to_midi_export(self):
        test_tokens = ["60", "62", "60.64.67", "65", "67"]
        output_file = self.test_dir / "test.mid"

        tokens_to_midi(test_tokens, output_file, step_duration=0.5)
        self.assertTrue(output_file.exists())
        self.assertGreater(output_file.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
