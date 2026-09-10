import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QMimeData, QPoint, QPointF, QProcess, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import meeting_stt.app as gui


class QueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        state = self.root / "state"
        state.mkdir()
        (state / "session.json").write_text(json.dumps({"output": str(self.root / "output"), "autostart": False}))
        self.state_patch = patch.object(gui, "STATE", state)
        self.state_patch.start()
        original_set = QProcess.setArguments
        fixture = Path(__file__).with_name("queue_worker_fixture.py")
        self.arguments_patch = patch.object(QProcess, "setArguments", lambda process, args: original_set(process, ["-u", str(fixture)]))
        self.arguments_patch.start()
        self.window = gui.MainWindow()

    def tearDown(self):
        self.window.close()
        self.application.processEvents()
        self.arguments_patch.stop()
        self.state_patch.stop()
        self.temp.cleanup()

    def file(self, name):
        path = self.root / f"{name}.wav"
        path.write_bytes(b"fixture")
        return str(path)

    def wait_until(self, predicate, seconds=5):
        deadline = time.monotonic() + seconds
        while not predicate() and time.monotonic() < deadline:
            QTest.qWait(20)
        self.assertTrue(predicate(), "Queue did not reach expected state")

    def test_file_drop_and_duplicate_pending_deduplication(self):
        path = self.file("한국어 회의")
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(path), QUrl.fromLocalFile(path)])
        drag = QDragEnterEvent(QPoint(20, 20), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.application.sendEvent(self.window.drop, drag)
        self.assertTrue(drag.isAccepted())
        drop = QDropEvent(QPointF(20, 20), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.application.sendEvent(self.window.drop, drop)
        self.assertEqual(len(self.window.jobs), 1)
        self.assertEqual(self.window.jobs[0]["source"], path)
        self.assertEqual(self.window.jobs[0]["status"], "pending")

    def test_queue_recovers_after_error_and_native_crash(self):
        self.window.add_files([self.file(name) for name in ("error", "first", "crash", "last")])
        self.window.autostart.setChecked(True)
        self.wait_until(lambda: [job["status"] for job in self.window.jobs] == ["failed", "completed", "failed", "completed"])
        self.wait_until(lambda: self.window.process is None)
        self.assertTrue(Path(self.window.jobs[-1]["transcript"]).exists())

    def test_cancel_stops_current_job_and_keeps_queue_paused(self):
        self.window.add_files([self.file("slow"), self.file("next")])
        self.window.autostart.setChecked(True)
        self.wait_until(lambda: bool(self.window.jobs[0].get("folder")))
        self.window.cancel_current()
        self.wait_until(lambda: self.window.jobs[0]["status"] == "cancelled")
        self.assertEqual(self.window.jobs[1]["status"], "pending")
        manifest = json.loads((Path(self.window.jobs[0]["folder"]) / "metadata.json").read_text())
        self.assertEqual(manifest["status"], "cancelled")
        self.window.toggle_pause()
        self.wait_until(lambda: self.window.jobs[1]["status"] == "completed")

    def test_pausing_finishes_active_job_before_next(self):
        self.window.add_files([self.file("first"), self.file("next")])
        self.window.autostart.setChecked(True)
        self.window.toggle_pause()
        self.wait_until(lambda: self.window.jobs[0]["status"] == "completed")
        self.assertEqual(self.window.jobs[1]["status"], "pending")
        self.window.toggle_pause()
        self.wait_until(lambda: self.window.jobs[1]["status"] == "completed")

    def test_asr_and_diarization_use_different_processes(self):
        with patch.object(gui, "runtime_ready", return_value=True), patch.object(gui, "model_ready", return_value=True), patch.object(gui, "DIARIZATION_PYTHON", gui.python_console()):
            self.window.add_files([self.file("staged")])
            self.window.autostart.setChecked(True)
            self.wait_until(lambda: self.window.jobs[0]["status"] == "completed")
            metadata = json.loads((Path(self.window.jobs[0]["folder"]) / "metadata.json").read_text())
            self.assertNotEqual(metadata["asr_pid"], metadata["diarization_pid"])
            self.assertEqual(self.window.jobs[0]["speaker_count"], 2)

    def test_no_account_keeps_plain_transcript_and_allows_stage_only_retry(self):
        with patch.object(gui, "model_ready", return_value=False):
            self.window.add_files([self.file("staged")])
            self.window.autostart.setChecked(True)
            self.wait_until(lambda: self.window.jobs[0]["status"] == "needs_setup")
        job = self.window.jobs[0]
        folder = Path(job["folder"])
        before = (folder / "result.txt").read_bytes()
        self.assertEqual(len(self.window.jobs), 1)
        with patch.object(gui, "runtime_ready", return_value=True), patch.object(gui, "model_ready", return_value=True), patch.object(gui, "DIARIZATION_PYTHON", gui.python_console()):
            self.window.table.selectRow(0)
            self.window.queue_diarization()
            self.wait_until(lambda: job["status"] == "completed")
        self.assertEqual(job["folder"], str(folder))
        self.assertEqual((folder / "result.txt").read_bytes(), before)
        self.assertEqual(len(list((self.root / "output").iterdir())), 1)

    def test_diarization_crash_keeps_asr_result(self):
        with patch.object(gui, "runtime_ready", return_value=True), patch.object(gui, "model_ready", return_value=True), patch.object(gui, "DIARIZATION_PYTHON", gui.python_console()):
            self.window.hotwords.setText("diarization_crash")
            self.window.add_files([self.file("staged")])
            self.window.autostart.setChecked(True)
            self.wait_until(lambda: self.window.jobs[0]["status"] == "failed")
            job = self.window.jobs[0]
            self.assertEqual(job["stage"], "diarization")
            self.assertTrue(Path(job["transcript"]).exists())

    def test_cancel_diarization_preserves_transcript(self):
        with patch.object(gui, "runtime_ready", return_value=True), patch.object(gui, "model_ready", return_value=True), patch.object(gui, "DIARIZATION_PYTHON", gui.python_console()):
            self.window.hotwords.setText("diarization_slow")
            self.window.add_files([self.file("staged")])
            self.window.autostart.setChecked(True)
            self.wait_until(lambda: self.window.active is not None and self.window.active.get("stage") == "diarization" and not self.window.phase_complete)
            self.window.cancel_current()
            self.wait_until(lambda: self.window.jobs[0]["status"] == "cancelled")
            self.assertTrue(Path(self.window.jobs[0]["transcript"]).exists())

    def test_recover_completed_file_when_final_event_was_lost(self):
        folder = self.root / "committed"
        folder.mkdir()
        (folder / "metadata.json").write_text('{"status":"completed","diarization_status":"completed"}')
        (folder / "transcript.json").write_text('{"text_file":"result.txt"}')
        (folder / "result.txt").write_text("saved")
        job = {"folder": str(folder), "status": "running", "stage": "diarization", "diarize": True}
        self.assertTrue(self.window.recover_committed_result(job))
        self.assertEqual(job["status"], "completed")

    def test_recover_asr_checkpoint_before_handoff_event(self):
        folder = self.root / "asr_committed"
        folder.mkdir()
        (folder / "metadata.json").write_text('{"status":"awaiting_diarization"}')
        (folder / "transcript.json").write_text('{"text_file":"result.txt"}')
        (folder / "result.txt").write_text("saved")
        job = {"folder": str(folder), "status": "running", "stage": "asr", "diarize": True}
        self.assertFalse(self.window.recover_committed_result(job))
        self.assertEqual(job["stage"], "diarization")
        self.assertEqual(Path(job["transcript"]).read_text(), "saved")


if __name__ == "__main__":
    unittest.main()
