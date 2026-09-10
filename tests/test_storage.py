import hashlib
import tempfile
import unittest
import wave
from pathlib import Path

from meeting_stt.storage import choose_date, copy_recording, create_folder, inspect_recording, parse_date, safe_stem


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio = self.root / "한국어 회의.wav"
        with wave.open(str(self.audio), "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            output.writeframes(b"\0\0" * 16000)

    def tearDown(self):
        self.temp.cleanup()

    def test_embedded_utc_crosses_midnight_in_korea(self):
        stamp, source, precision, warning = choose_date({"container": {"creation_time": "2026-09-07T18:30:45.000000Z"}}, self.audio, "Asia/Seoul")
        self.assertEqual(stamp, "2026-09-08T03:30:45+09:00")
        self.assertEqual(source, "container.creation_time")
        self.assertFalse(warning)

    def test_offset_exif_date_only_and_invalid_tags(self):
        self.assertEqual(parse_date("2026:09:07 18:00:00", "Asia/Seoul")[0].hour, 18)
        self.assertEqual(parse_date("2026-09-07T18:00:00+09:00", "Asia/Seoul")[0].hour, 18)
        self.assertEqual(parse_date("2026-09-07", "Asia/Seoul")[1], "day")
        self.assertIsNone(parse_date("2026", "Asia/Seoul"))
        self.assertIsNone(parse_date("0000-00-00T00:00:00Z", "Asia/Seoul"))
        stamp, source, _, warning = choose_date({"container": {"creation_time": "invalid"}, "stream0": {"creation_time": "2026-09-07T12:00:00Z"}}, self.audio, "Asia/Seoul")
        self.assertEqual(source, "stream0.creation_time")

    def test_bwf_and_day_precision(self):
        date, source, _, warning = choose_date({"container": {"origination_date": "2026-09-07", "origination_time": "15:30:00"}}, self.audio, "Asia/Seoul")
        self.assertIn("15:30:00", date)
        _, _, precision, warning = choose_date({"container": {"origination_date": "2026-09-07"}}, self.audio, "Asia/Seoul")
        self.assertEqual(precision, "day")
        self.assertTrue(warning)
        _, _, precision, warning = choose_date({"audio_tags": {"TDRC": "2026-09-07"}}, self.audio, "Asia/Seoul")
        self.assertEqual(precision, "day")
        self.assertTrue(warning)

    def test_missing_metadata_is_explicit_and_copy_is_byte_identical(self):
        recording = inspect_recording(self.audio)
        self.assertTrue(recording.date_source.startswith("filesystem."))
        self.assertTrue(recording.date_warning)
        folder = create_folder(self.root / "output", recording)
        target = folder / self.audio.name
        original = self.audio.read_bytes()
        digest = copy_recording(recording, target, lambda _: None)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(self.audio.read_bytes(), original)
        self.assertEqual(digest, hashlib.sha256(original).hexdigest())
        self.assertFalse(target.with_name(target.name + ".copying").exists())

    def test_duplicate_folder_does_not_overwrite(self):
        recording = inspect_recording(self.audio)
        first = create_folder(self.root / "output", recording)
        (first / "keep.txt").write_text("existing", encoding="utf-8")
        second = create_folder(self.root / "output", recording)
        self.assertNotEqual(first, second)
        self.assertEqual((first / "keep.txt").read_text(), "existing")
        self.assertEqual(safe_stem("CON"), "_CON")
        self.assertNotIn(":", safe_stem("meeting: review"))

    def test_changed_source_does_not_become_final_copy(self):
        recording = inspect_recording(self.audio)
        with self.audio.open("ab") as output:
            output.write(b"changed")
        target = self.root / "copy.wav"
        with self.assertRaises(RuntimeError):
            copy_recording(recording, target, lambda _: None)
        self.assertFalse(target.exists())

    def test_invalid_audio_fails_before_output_is_created(self):
        self.audio.write_bytes(b"not audio")
        with self.assertRaises(Exception):
            inspect_recording(self.audio)


if __name__ == "__main__":
    unittest.main()
