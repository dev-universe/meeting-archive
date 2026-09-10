import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from meeting_stt.diarization import model_ready, run_diarization, safe_error
from meeting_stt.speakers import local_file, merge_speakers, rename_speakers, save_speakers


def segment(start, text, words):
    return {"start": start, "end": start + len(words), "text": text,
            "words": [{"start": start + i, "end": start + i + 1, "word": word} for i, word in enumerate(words)]}


class SpeakerTests(unittest.TestCase):
    def test_speaker_changes_inside_sentence_and_across_hour(self):
        transcript = {"segments": [segment(0, "안녕 하세요", ["안녕", " 하세요"]), segment(3600, "다시 만나요", ["다시", " 만나요"])]}
        turns = [{"start": 0, "end": 1, "speaker": "A"}, {"start": 1, "end": 2, "speaker": "B"}, {"start": 3600, "end": 3602, "speaker": "A"}]
        result = merge_speakers(transcript, turns, turns)
        self.assertEqual([u["speaker"] for u in result["utterances"]], ["A", "B", "A"])
        self.assertEqual(result["speaker_names"], {"A": "화자 1", "B": "화자 2"})
        self.assertEqual("".join(u["text"] for u in result["utterances"]), "안녕 하세요다시 만나요")

    def test_overlap_is_simultaneous_speech_not_adjacent_turns(self):
        transcript = {"segments": [segment(0, "발언", ["발언"])]}
        turns = [{"start": 0, "end": 0.5, "speaker": "A"}, {"start": 0.5, "end": 1, "speaker": "B"}]
        result = merge_speakers(transcript, turns, turns)
        self.assertFalse(result["utterances"][0]["overlap"])
        self.assertTrue(result["utterances"][0]["uncertain"])
        turns[0]["end"] = 1
        result = merge_speakers(transcript, turns, [turns[0]])
        self.assertTrue(result["utterances"][0]["overlap"])

    def test_unmatched_words_stay_unknown(self):
        result = merge_speakers({"segments": [segment(10, "네", ["네"])]}, [], [])
        self.assertIsNone(result["utterances"][0]["speaker"])
        self.assertEqual(result["utterances"][0]["text"], "네")

    def test_incomplete_alignment_keeps_all_original_text(self):
        result = merge_speakers({"segments": [segment(0, "예산 200만 원입니다.", ["예산"])]}, [], [])
        self.assertEqual(result["utterances"][0]["text"], "예산 200만 원입니다.")
        self.assertTrue(result["utterances"][0]["uncertain"])

    def test_rename_regenerates_txt_and_preserves_plain_text(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            transcript = {"header": "녹음: 회의.wav\n", "segments": [segment(0, "안녕하세요", ["안녕하세요"])], "text_file": "회의.txt"}
            (folder / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
            turns = [{"start": 0, "end": 1, "speaker": "A"}]
            result = merge_speakers(transcript, turns, turns)
            save_speakers(folder, result)
            plain = (folder / "회의.plain.txt").read_bytes()
            rename_speakers(folder, {"A": "김팀장"})
            self.assertIn("김팀장: 안녕하세요", (folder / "회의.txt").read_text(encoding="utf-8-sig"))
            self.assertEqual((folder / "회의.plain.txt").read_bytes(), plain)
            with self.assertRaises(ValueError):
                rename_speakers(folder, {"A": " "})

    def test_paths_cannot_escape_result_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                local_file(Path(directory), "../other.txt")

    def test_missing_model_does_not_load_torch_or_change_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "metadata.json").write_text('{"status":"completed"}')
            (folder / "transcript.json").write_text('{"text_file":"test.txt"}')
            (folder / "test.txt").write_text("plain transcription")
            events = []
            with patch("meeting_stt.diarization.model_ready", return_value=False):
                run_diarization({"folder": directory}, lambda event, **data: events.append(event))
            self.assertEqual(events, ["needs_setup"])
            self.assertEqual((folder / "test.txt").read_text(), "plain transcription")

    def test_partial_download_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / ".ready.json").write_text('{"files":{"config.yaml":100}}')
            with patch("meeting_stt.diarization.MODEL_DIR", folder):
                self.assertFalse(model_ready())

    def test_token_is_redacted(self):
        self.assertNotIn("hf_privateToken123", safe_error(RuntimeError("hf_privateToken123 network failure")))
        self.assertIn("접근 권한", safe_error(RuntimeError("403 gated repo")))


if __name__ == "__main__":
    unittest.main()
