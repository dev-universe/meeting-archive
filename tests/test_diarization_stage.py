"""Run with .venv-diarization; fixture labels test plumbing, not model accuracy."""
import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from meeting_stt.diarization import run_diarization


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires the optional diarization environment")
class DiarizationStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        (self.folder / "metadata.json").write_text('{"status":"awaiting_diarization"}')
        data = {"header": "테스트 녹음", "segments": [
            {"start": 0, "end": 1, "text": "첫 발언", "words": [{"start": 0, "end": 1, "word": "첫 발언"}]},
            {"start": 3598, "end": 3599, "text": "마지막 발언", "words": [{"start": 3598, "end": 3599, "word": "마지막 발언"}]},
        ], "audio_file": "audio.wav", "text_file": "meeting.txt"}
        (self.folder / "transcript.json").write_text(json.dumps(data), encoding="utf-8")
        (self.folder / "meeting.txt").write_text("existing ASR", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_hour_audio_uses_one_global_pipeline_and_exports(self):
        from pyannote.core import Annotation, Segment
        with wave.open(str(self.folder / "audio.wav"), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            silence = b"\0\0" * 16000 * 10
            for _ in range(360):
                stream.writeframesraw(silence)
        annotation = Annotation()
        annotation[Segment(0, 1)] = "A"
        annotation[Segment(30, 31)] = "B"
        annotation[Segment(3598, 3599)] = "A"
        calls = []

        class FixturePipeline:
            def to(self, device):
                return self

            def __call__(self, audio, **options):
                calls.append((tuple(audio["waveform"].shape), options["num_speakers"]))
                return SimpleNamespace(speaker_diarization=annotation, exclusive_speaker_diarization=annotation)

        events = []
        run_diarization({"folder": str(self.folder), "diarization_device": "cpu", "num_speakers": 2},
                        lambda event, **data: events.append(event), pipeline_factory=FixturePipeline)
        self.assertEqual(calls, [((1, 3600 * 16000), 2)])
        self.assertEqual(events[-1], "done")
        self.assertFalse((self.folder / ".diarization.f32").exists())
        result = json.loads((self.folder / "speakers.json").read_text(encoding="utf-8"))
        self.assertEqual([u["speaker"] for u in result["utterances"]], ["A", "A"])
        self.assertIn("화자 1", (self.folder / "meeting.txt").read_text(encoding="utf-8-sig"))

    def test_model_failure_preserves_plain_text_and_marks_only_diarization_failed(self):
        with wave.open(str(self.folder / "audio.wav"), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\0\0" * 16000)
        def broken():
            raise RuntimeError("CUDA out of memory")
        with self.assertRaises(RuntimeError):
            run_diarization({"folder": str(self.folder)}, lambda *a, **kw: None, pipeline_factory=broken)
        self.assertEqual((self.folder / "meeting.txt").read_text(encoding="utf-8"), "existing ASR")
        manifest = json.loads((self.folder / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["diarization_status"], "failed")
        self.assertFalse((self.folder / ".diarization.f32").exists())


if __name__ == "__main__":
    unittest.main()
