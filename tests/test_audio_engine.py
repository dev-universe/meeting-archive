import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from meeting_stt.audio import SAMPLE_RATE, audio_chunks
from meeting_stt.engine import Transcriber


class AudioEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "회의.wav"
        self.samples = (np.sin(np.arange(SAMPLE_RATE * 4 + 123) * 0.08) * 2000).astype(np.int16)
        with wave.open(str(self.source), "wb") as output:
            output.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            output.writeframes(self.samples.tobytes())
        self.job = {"source": str(self.source), "output": str(self.root / "output"), "model": "large-v3"}

    def tearDown(self):
        self.temp.cleanup()

    def test_chunking_preserves_every_sample_once_and_offsets(self):
        chunks = list(audio_chunks(self.source, chunk_seconds=1))
        actual = np.concatenate([samples for _, samples in chunks])
        np.testing.assert_array_equal(actual, self.samples.astype(np.float32) / 32768)
        position = 0
        for offset, samples in chunks:
            self.assertEqual(offset, position / SAMPLE_RATE)
            position += len(samples)
            self.assertLessEqual(len(samples), SAMPLE_RATE)

    def test_success_atomically_publishes_text_and_manifest(self):
        model = SimpleNamespace(transcribe=lambda *a, **kw: (iter([SimpleNamespace(start=0, end=1.5, text=" 다음 회의는 월요일입니다.")]), None))
        events = []
        with patch.object(Transcriber, "load", return_value=model):
            Transcriber().run(self.job, lambda kind, **data: events.append((kind, data)))
        folder = next((self.root / "output").iterdir())
        manifest = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertIn("다음 회의는 월요일", (folder / manifest["transcript_file"]).read_text(encoding="utf-8-sig"))
        self.assertFalse(list(folder.glob("*.partial.txt")))
        self.assertEqual((folder / self.source.name).read_bytes(), self.source.read_bytes())
        self.assertEqual(events[-1][0], "done")

    def test_failure_retains_partial_but_never_labels_it_completed(self):
        def fail(*args, **kwargs):
            def segments():
                yield SimpleNamespace(start=0, end=1, text="보관할 내용")
                raise RuntimeError("simulated decode failure")
            return segments(), None
        with patch.object(Transcriber, "load", return_value=SimpleNamespace(transcribe=fail)):
            with self.assertRaises(RuntimeError):
                Transcriber().run(self.job, lambda *a, **kw: None)
        folder = next((self.root / "output").iterdir())
        manifest = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue(list(folder.glob("*.partial.txt")))
        self.assertFalse((folder / "회의.txt").exists())
        self.assertTrue(self.source.exists())


if __name__ == "__main__":
    unittest.main()
