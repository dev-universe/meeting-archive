from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QLockFile, QProcess, QProcessEnvironment, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .runtime import ROOT, STATE, python_console
from .storage import SUPPORTED, atomic_json
from .diarization import DIARIZATION_PYTHON, model_ready, runtime_ready
from .speaker_dialogs import SpeakerNamesDialog, SpeakerSetupDialog
from .speakers import local_file

STATUS = {"pending": "대기", "running": "처리 중", "completed": "완료", "failed": "실패", "cancelled": "중단", "needs_setup": "준비 필요"}

STYLE = """
QWidget { font-family: 'Malgun Gothic'; font-size: 13px; color: #25334A; }
QMainWindow, #canvas { background: #F3F5F8; }
QLabel#eyebrow { color: #008578; font-size: 11px; font-weight: 700; }
QLabel#heading { font-size: 28px; font-weight: 700; color: #15243C; }
QLabel#muted { color: #67758A; }
QLabel#badge { background: #E0F1EB; color: #146E5B; border-radius: 12px; padding: 7px 12px; font-size: 12px; }
QFrame#card { background: white; border: 1px solid #DEE4ED; border-radius: 12px; }
QFrame#drop { background: #EAF4F1; border: 2px dashed #8ABFAF; border-radius: 12px; }
QFrame#drop QLabel { background: transparent; border: none; }
QLabel#dropTitle { font-size: 19px; font-weight: 700; color: #175E50; }
QPushButton { background: white; border: 1px solid #CFD7E2; border-radius: 7px; padding: 7px 13px; }
QPushButton:hover { background: #EEF4F6; border-color: #8CAFA8; }
QPushButton:disabled { color: #A2ACBA; background: #F3F5F7; }
QPushButton#primary { background: #087F70; color: white; border-color: #087F70; font-weight: 700; }
QPushButton#primary:hover { background: #06695D; }
QLineEdit, QComboBox, QSpinBox { border: 1px solid #D4DCE6; border-radius: 6px; padding: 7px 10px; background: #FAFBFD; }
QComboBox QAbstractItemView { background: white; selection-background-color: #DCF0E9; selection-color: #164B41; }
QCheckBox { spacing: 6px; }
QTableWidget { border: none; background: white; gridline-color: #EDF0F5; selection-background-color: #E2F2EC; selection-color: #174E43; }
QHeaderView::section { background: #F7F9FB; color: #748094; border: none; border-bottom: 1px solid #E5EAF0; padding: 9px; font-weight: 600; }
QPlainTextEdit { background: white; border: none; padding: 9px; selection-background-color: #CBEADF; }
QSplitter::handle { background: #F3F5F8; height: 10px; }
QStatusBar { background: #E8EEF3; color: #59697D; }
"""


class DropArea(QFrame):
    files_dropped = Signal(list)
    browse = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("drop")
        self.setAcceptDrops(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        words = QVBoxLayout()
        title = QLabel("녹음 파일을 여기에 놓으세요")
        title.setObjectName("dropTitle")
        words.addWidget(title)
        sub = QLabel("여러 파일을 한 번에 · 녹음 날짜로 정리 · TXT로 저장")
        sub.setObjectName("muted")
        words.addWidget(sub)
        layout.addLayout(words, 1)
        button = QPushButton("파일 선택")
        button.setObjectName("primary")
        button.clicked.connect(self.browse.emit)
        layout.addWidget(button)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        self.files_dropped.emit([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("회의 보관함 · 로컬 한국어 STT")
        self.resize(1120, 860)
        self.setMinimumSize(920, 740)
        self.setAcceptDrops(True)
        self.jobs = []
        self.active = None
        self.process = None
        self.stdout_buffer = b""
        self.stderr_buffer = ""
        self.worker_closing = False
        self.phase_complete = False
        self.closing = False
        self.cancelling = False
        self.paused = False
        STATE.mkdir(parents=True, exist_ok=True)
        config = self.read_state()
        self.jobs = config.get("jobs", [])
        for job in self.jobs:
            if job.get("status") == "running":
                if self.recover_committed_result(job):
                    continue
                job.update(status="cancelled", message="이전 실행에서 중단됨 · 선택 후 다시 시도")
                self.mark_manifest(job, "interrupted")
        self.build_ui(config)
        self.refresh_table()
        self.save()
        QTimer.singleShot(300, self.start_next)

    def read_state(self):
        try:
            return json.loads((STATE / "session.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def build_ui(self, config):
        canvas = QWidget()
        canvas.setObjectName("canvas")
        self.setCentralWidget(canvas)
        layout = QVBoxLayout(canvas)
        layout.setContentsMargins(28, 22, 28, 20)
        layout.setSpacing(15)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(3)
        eyebrow = QLabel("MEETING ARCHIVE")
        eyebrow.setObjectName("eyebrow")
        titles.addWidget(eyebrow)
        heading = QLabel("회의는 끝났고, 기록은 남습니다.")
        heading.setObjectName("heading")
        titles.addWidget(heading)
        header.addLayout(titles, 1)
        badge = QLabel("내 PC에서 처리  ·  한국어")
        badge.setObjectName("badge")
        header.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header)
        self.drop = DropArea()
        self.drop.files_dropped.connect(self.add_files)
        self.drop.browse.connect(self.browse_files)
        layout.addWidget(self.drop)

        card = QFrame()
        card.setObjectName("card")
        settings = QVBoxLayout(card)
        settings.setContentsMargins(16, 13, 16, 13)
        row = QHBoxLayout()
        row.addWidget(QLabel("저장 위치"))
        self.output = QLineEdit(config.get("output", str(ROOT / "output")))
        self.output.setReadOnly(True)
        row.addWidget(self.output, 1)
        choose = QPushButton("변경")
        choose.clicked.connect(self.choose_output)
        row.addWidget(choose)
        open_root = QPushButton("폴더 열기")
        open_root.clicked.connect(self.open_root)
        row.addWidget(open_root)
        settings.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("전사 모델"))
        self.model = QComboBox()
        self.model.addItem("Large v3  ·  정확도 우선", "large-v3")
        self.model.addItem("Large v3 Turbo  ·  속도 우선", "turbo")
        self.model.setCurrentIndex(max(0, self.model.findData(config.get("model", "large-v3"))))
        row.addWidget(self.model, 1)
        self.timestamps = QCheckBox("TXT에 시간 표시")
        self.timestamps.setChecked(config.get("timestamps", True))
        row.addWidget(self.timestamps)
        self.autostart = QCheckBox("추가하면 바로 시작")
        self.autostart.setChecked(config.get("autostart", True))
        self.autostart.toggled.connect(self.autostart_changed)
        row.addWidget(self.autostart)
        settings.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("참고 용어"))
        self.hotwords = QLineEdit(config.get("hotwords", ""))
        self.hotwords.setPlaceholderText("선택 사항 · 참석자 이름, 회사명, 제품명 등")
        self.hotwords.setMaxLength(800)
        row.addWidget(self.hotwords, 1)
        settings.addLayout(row)
        row = QHBoxLayout()
        self.diarize = QCheckBox("화자 구분 사용")
        self.diarize.setChecked(config.get("diarize", True))
        row.addWidget(self.diarize)
        row.addWidget(QLabel("참석자 수"))
        self.num_speakers = QSpinBox()
        self.num_speakers.setRange(0, 30)
        self.num_speakers.setSpecialValueText("자동")
        self.num_speakers.setValue(config.get("num_speakers", 0))
        row.addWidget(self.num_speakers)
        self.diarization_device = QComboBox()
        self.diarization_device.addItem("화자 분석 · GPU", "cuda")
        self.diarization_device.addItem("화자 분석 · CPU", "cpu")
        self.diarization_device.setCurrentIndex(max(0, self.diarization_device.findData(config.get("diarization_device", "cuda"))))
        row.addWidget(self.diarization_device)
        row.addStretch()
        self.model_state = QLabel("화자 모델 준비됨" if model_ready() else "화자 모델 준비 필요")
        self.model_state.setObjectName("muted")
        row.addWidget(self.model_state)
        setup_button = QPushButton("모델 준비")
        setup_button.clicked.connect(self.open_speaker_setup)
        row.addWidget(setup_button)
        settings.addLayout(row)
        note = QLabel("녹음 메타데이터 우선 · 한국 시간(UTC+9) · 원본은 결과 폴더에 복사됩니다")
        note.setObjectName("muted")
        settings.addWidget(note)
        layout.addWidget(card)

        row = QHBoxLayout()
        self.summary = QLabel("작업 목록")
        row.addWidget(self.summary, 1)
        self.pause_button = QPushButton("대기열 일시정지" if self.autostart.isChecked() else "대기열 시작")
        self.pause_button.clicked.connect(self.toggle_pause)
        row.addWidget(self.pause_button)
        self.cancel_button = QPushButton("현재 작업 중단")
        self.cancel_button.clicked.connect(self.cancel_current)
        self.cancel_button.setEnabled(False)
        row.addWidget(self.cancel_button)
        self.retry_button = QPushButton("선택 항목 다시 시도")
        self.retry_button.clicked.connect(self.retry_selected)
        row.addWidget(self.retry_button)
        layout.addLayout(row)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["녹음 파일", "녹음 시점 · 한국 시간", "상태", "진행 상황"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(43)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(1, 180)
        self.table.setColumnWidth(2, 82)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        self.table.cellDoubleClicked.connect(lambda *_: self.open_selected_folder())
        splitter.addWidget(self.table)
        bottom = QFrame()
        bottom.setObjectName("card")
        box = QVBoxLayout(bottom)
        box.setContentsMargins(12, 9, 12, 9)
        row = QHBoxLayout()
        row.addWidget(QLabel("전사 미리보기"), 1)
        self.analyze_speakers = QPushButton("화자 분석 / 재시도")
        self.analyze_speakers.clicked.connect(self.queue_diarization)
        row.addWidget(self.analyze_speakers)
        self.edit_speakers = QPushButton("화자 이름 변경")
        self.edit_speakers.clicked.connect(self.edit_speaker_names)
        row.addWidget(self.edit_speakers)
        self.open_result = QPushButton("결과 폴더")
        self.open_result.clicked.connect(self.open_selected_folder)
        row.addWidget(self.open_result)
        self.open_text = QPushButton("TXT 열기")
        self.open_text.clicked.connect(self.open_selected_text)
        row.addWidget(self.open_text)
        box.addLayout(row)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumBlockCount(600)
        self.preview.setPlaceholderText("녹음을 추가하면 전사 내용이 여기에 표시됩니다.\n모델 다운로드 이후에는 인터넷 연결 없이 사용할 수 있습니다.")
        box.addWidget(self.preview)
        splitter.addWidget(bottom)
        splitter.setSizes([140, 250])
        layout.addWidget(splitter, 1)
        self.statusBar().showMessage("준비됨 · GPU를 사용하며, 녹음 파일은 외부로 전송하지 않습니다.")

    def save(self):
        config = {"output": self.output.text(), "model": self.model.currentData(),
                  "timestamps": self.timestamps.isChecked(), "hotwords": self.hotwords.text(),
                  "autostart": self.autostart.isChecked(), "jobs": self.jobs,
                  "diarize": self.diarize.isChecked(), "num_speakers": self.num_speakers.value(),
                  "diarization_device": self.diarization_device.currentData()}
        try:
            atomic_json(STATE / "session.json", config)
        except OSError as error:
            self.statusBar().showMessage(f"작업 기록 저장 실패: {error}")

    def add_files(self, paths):
        skipped = []
        known = {str(Path(job["source"]).resolve()).casefold() for job in self.jobs if job["status"] in ("pending", "running")}
        for raw in paths:
            path = Path(raw).resolve()
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                skipped.append(path.name)
                continue
            key = str(path).casefold()
            if key in known:
                continue
            known.add(key)
            self.jobs.append({"id": uuid.uuid4().hex, "source": str(path), "output": self.output.text(),
                              "model": self.model.currentData(), "timezone": "Asia/Seoul",
                              "timestamps": self.timestamps.isChecked(), "hotwords": self.hotwords.text().strip(),
                              "diarize": self.diarize.isChecked(), "num_speakers": self.num_speakers.value(),
                              "diarization_device": self.diarization_device.currentData(), "stage": "asr",
                              "status": "pending", "message": "순서대로 처리합니다", "progress": 0})
        self.refresh_table()
        self.save()
        if skipped:
            self.statusBar().showMessage("지원하지 않는 파일·폴더: " + ", ".join(skipped[:5]))
        if self.autostart.isChecked():
            self.start_next()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.add_files([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()

    def browse_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "녹음 파일 추가", "", "녹음·동영상 (" + " ".join("*" + ext for ext in sorted(SUPPORTED)) + ")")
        self.add_files(paths)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "전사 결과를 저장할 폴더", self.output.text())
        if path:
            self.output.setText(path)
            self.save()

    def open_root(self):
        path = Path(self.output.text())
        try:
            path.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except OSError as error:
            self.statusBar().showMessage(str(error))

    def selected_job(self):
        row = self.table.currentRow()
        return self.jobs[row] if 0 <= row < len(self.jobs) else None

    def selection_changed(self):
        job = self.selected_job()
        self.selection_buttons()
        if not job:
            return
        self.preview.clear()
        if job.get("transcript"):
            try:
                self.preview.setPlainText(Path(job["transcript"]).read_text(encoding="utf-8-sig"))
            except OSError as error:
                self.preview.setPlainText(str(error))
        elif job.get("folder"):
            partials = list(Path(job["folder"]).glob("*.partial.txt"))
            if partials:
                try:
                    self.preview.setPlainText(partials[0].read_text(encoding="utf-8-sig"))
                except (OSError, UnicodeError):
                    pass
        if job["status"] in ("failed", "cancelled", "needs_setup"):
            self.preview.appendPlainText("\n" + job.get("message", ""))

    def open_selected_folder(self):
        job = self.selected_job()
        if job and job.get("folder"):
            QDesktopServices.openUrl(QUrl.fromLocalFile(job["folder"]))

    def open_selected_text(self):
        job = self.selected_job()
        if job and job.get("transcript"):
            QDesktopServices.openUrl(QUrl.fromLocalFile(job["transcript"]))

    def refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.jobs))
        for row, job in enumerate(self.jobs):
            recording = job.get("recording", {})
            when = recording.get("recorded_at", "")
            date = datetime.fromisoformat(when).strftime("%Y-%m-%d %H:%M:%S") if when else "처리 시 확인"
            if recording.get("date_warning"):
                date = "※ " + date
            for column, text in enumerate((Path(job["source"]).name, date, STATUS[job["status"]], job.get("message", ""))):
                item = QTableWidgetItem(text)
                item.setToolTip(job["source"] if column == 0 else (recording.get("date_source", "") + "\n" + recording.get("date_warning", "")) if column == 1 else job.get("message", ""))
                self.table.setItem(row, column, item)
        self.table.blockSignals(False)
        counts = {status: sum(job["status"] == status for job in self.jobs) for status in STATUS}
        self.summary.setText(f"작업 {len(self.jobs)}개  ·  대기 {counts['pending']}  ·  완료 {counts['completed']}")
        self.cancel_button.setEnabled(self.active is not None)
        self.selection_buttons()

    def selection_buttons(self):
        job = self.selected_job()
        self.open_result.setEnabled(bool(job and job.get("folder")))
        self.open_text.setEnabled(bool(job and job.get("transcript")))
        self.retry_button.setEnabled(bool(job and job["status"] in ("failed", "cancelled", "needs_setup")))
        idle = job and job["status"] not in ("pending", "running")
        folder = Path(job["folder"]) if job and job.get("folder") else None
        self.analyze_speakers.setEnabled(bool(idle and folder and (folder / "transcript.json").is_file()))
        self.edit_speakers.setEnabled(bool(idle and folder and (folder / "speakers.json").is_file()))

    def open_speaker_setup(self):
        if self.active:
            QMessageBox.information(self, "화자 모델 준비", "현재 작업이 끝난 뒤 모델 준비를 열어 주세요. 대기열을 일시정지하면 다음 작업 전에 멈춥니다.")
            return
        was_paused = self.paused
        self.paused = True
        SpeakerSetupDialog(self).exec()
        self.model_state.setText("화자 모델 준비됨" if model_ready() else "화자 모델 준비 필요")
        self.paused = was_paused
        self.start_next()

    def edit_speaker_names(self):
        job = self.selected_job()
        if job and job.get("folder") and job["status"] not in ("running", "pending"):
            try:
                SpeakerNamesDialog(Path(job["folder"]), self).exec()
                self.selection_changed()
            except (ValueError, OSError, KeyError) as error:
                QMessageBox.warning(self, "화자 이름", str(error))

    def queue_diarization(self):
        job = self.selected_job()
        if not job or job["status"] in ("pending", "running") or not job.get("folder"):
            return
        if not (Path(job["folder"]) / "transcript.json").is_file():
            self.statusBar().showMessage("이전 버전의 결과에는 단어 시간 정보가 없습니다. 파일을 새로 추가해 전사해 주세요.")
            return
        job.update(status="pending", stage="diarization", diarize=True,
                   num_speakers=self.num_speakers.value(), diarization_device=self.diarization_device.currentData(),
                   message="일반 전사본을 사용해 화자 분석 대기 중")
        self.save()
        self.refresh_table()
        self.start_next()

    def autostart_changed(self):
        self.pause_button.setText("대기열 시작" if not self.autostart.isChecked() else "대기열 일시정지")
        self.save()
        if self.autostart.isChecked():
            self.start_next()

    def toggle_pause(self):
        if not self.autostart.isChecked():
            self.paused = False
            self.autostart.setChecked(True)
        else:
            self.paused = not self.paused
        self.pause_button.setText("대기열 계속" if self.paused else "대기열 일시정지")
        if self.paused:
            self.statusBar().showMessage("현재 작업을 마친 뒤 대기열을 멈춥니다.")
        else:
            self.start_next()

    def start_next(self):
        if self.active or self.closing or self.worker_closing:
            return
        pending = next((job for job in self.jobs if job["status"] == "pending"), None)
        if not pending or self.paused or not self.autostart.isChecked():
            if self.process:
                self.worker_closing = True
                self.process.closeWriteChannel()
            return
        self.active = pending
        self.cancelling = False
        pending.update(status="running", message="작업 시작 중")
        self.save()
        self.refresh_table()
        self.table.selectRow(self.jobs.index(pending))
        self.preview.clear()
        self.launch_stage()

    def launch_stage(self):
        job = self.active
        if not job:
            return
        self.phase_complete = False
        if job.get("stage") == "diarization" and (not runtime_ready() or not model_ready()):
            job.update(status="needs_setup", message="일반 전사 완료 · 모델 준비 후 화자 분석을 실행해 주세요.")
            self.mark_manifest(job, "awaiting_diarization")
            self.active = None
            self.save()
            self.refresh_table()
            self.selection_changed()
            QTimer.singleShot(0, self.start_next)
            return
        # A pending diarization-only job may follow a reusable Whisper worker.
        if self.process is not None and job.get("stage") == "diarization":
            self.phase_complete = True
            self.worker_closing = True
            self.process.closeWriteChannel()
            return
        if self.process is None:
            self.stdout_buffer = b""
            self.stderr_buffer = ""
            process = QProcess(self)
            self.process = process
            environment = QProcessEnvironment.systemEnvironment()
            environment.insert("PYTHONUTF8", "1")
            environment.insert("HF_HUB_DISABLE_TELEMETRY", "1")
            environment.insert("PYANNOTE_METRICS_ENABLED", "0")
            environment.insert("OTEL_SDK_DISABLED", "true")
            process.setProcessEnvironment(environment)
            process.setWorkingDirectory(str(ROOT))
            is_diarization = job.get("stage") == "diarization"
            process.setProgram(str(DIARIZATION_PYTHON) if is_diarization else python_console())
            process.setArguments(["-u", "-m", "meeting_stt.diarization_worker" if is_diarization else "meeting_stt.worker"])
            process.started.connect(self.send_job)
            process.readyReadStandardOutput.connect(self.read_stdout)
            process.readyReadStandardError.connect(self.read_stderr)
            process.finished.connect(self.process_finished)
            process.errorOccurred.connect(self.process_error)
            process.start()
        else:
            self.send_job()

    def send_job(self):
        if self.active and self.process:
            self.process.write((json.dumps(self.active, ensure_ascii=False) + "\n").encode("utf-8"))
            if self.active.get("stage") == "diarization":
                self.process.closeWriteChannel()

    def read_stderr(self):
        if self.process:
            self.stderr_buffer = (self.stderr_buffer + bytes(self.process.readAllStandardError()).decode("utf-8", "replace"))[-24000:]

    def read_stdout(self):
        if not self.process:
            return
        self.stdout_buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split(b"\n", 1)
            try:
                self.on_event(json.loads(line))
            except (ValueError, KeyError):
                self.stderr_buffer = (self.stderr_buffer + line.decode("utf-8", "replace"))[-24000:]

    def on_event(self, event):
        job = self.active
        if not job or event.get("id") != job["id"] or self.cancelling:
            return
        kind = event["event"]
        if kind == "prepared":
            job.update(folder=event["folder"], recording=event["recording"])
            self.save()
        elif kind == "status":
            job["message"] = event["message"]
            self.statusBar().showMessage(event["message"])
        elif kind == "copy_progress":
            job["message"] = f"녹음본 복사 {event['progress']:.0%}"
        elif kind in ("segment", "progress"):
            duration = event.get("duration", 0)
            percentage = min(99, int(event["seconds"] / duration * 100)) if duration > 0 else 0
            job["message"] = f"전사 중 {percentage}%" if duration > 0 else f"전사 중 · {int(event['seconds'])}초 처리"
            if kind == "segment" and self.selected_job() is job:
                self.preview.appendPlainText(event["text"])
        elif kind == "asr_done":
            job.update(transcript=event["transcript"], folder=event["folder"], stage="diarization",
                       message="일반 전사 완료 · GPU 메모리 반환 중")
            self.phase_complete = True
            self.worker_closing = True
            self.save()
            self.selection_changed()
        elif kind == "needs_setup":
            job.update(status="needs_setup", transcript=event["transcript"], folder=event["folder"], message=event["message"])
            self.active = None
            self.worker_closing = True
            self.save()
            self.selection_changed()
        elif kind == "done":
            job.update(status="completed", transcript=event["transcript"], folder=event["folder"],
                       message=f"완료 · {event['elapsed'] / 60:.1f}분 소요" if event["segments"] else "완료 · 인식된 발화 없음")
            if event.get("speaker_count") is not None:
                job["message"] = f"완료 · 화자 {event['speaker_count']}명 · 분석 {event['elapsed'] / 60:.1f}분"
                job["speaker_count"] = event["speaker_count"]
            if job.get("stage") == "diarization":
                self.worker_closing = True
            self.active = None
            self.save()
            self.selection_changed()
            self.statusBar().showMessage("전사 완료 · " + event["folder"])
            QTimer.singleShot(0, self.start_next)
        elif kind == "error":
            job.update(status="failed", message=self.friendly_error(event["message"]))
            self.active = None
            self.save()
            self.selection_changed()
            self.statusBar().showMessage(job["message"])
            # The worker exits after an error; finished starts the next file.
        self.refresh_table()

    @staticmethod
    def friendly_error(message):
        lowered = message.lower()
        if "out of memory" in lowered:
            return "GPU 메모리 부족 · GPU를 쓰는 다른 앱을 닫거나 Turbo 선택 후 다시 시도. " + message
        if any(word in lowered for word in ("cublas", "cudnn", "dll", "cuda driver")):
            return "GPU 실행 오류 · setup.ps1로 설치를 확인해 주세요. " + message
        return message

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.process_finished(-1, QProcess.ExitStatus.CrashExit)

    def process_finished(self, code, status):
        self.read_stdout()
        self.read_stderr()
        handoff = self.active is not None and self.phase_complete and code == 0 and not self.cancelling and not self.closing
        if self.active and not handoff:
            job = self.active
            if self.recover_committed_result(job):
                pass
            elif self.cancelling or self.closing:
                job.update(status="cancelled", message="중단됨 · 복사본과 임시 전사본은 결과 폴더에 보관")
            else:
                detail = self.stderr_buffer.strip()[-1500:] or f"전사 프로세스가 종료되었습니다 (코드 {code})."
                job.update(status="failed", message=self.friendly_error(detail))
            if job["status"] != "completed":
                self.mark_manifest(job, job["status"])
            self.active = None
        if self.stderr_buffer:
            try:
                (STATE / "worker.log").write_text(self.stderr_buffer, encoding="utf-8")
            except OSError:
                pass
        if self.process:
            self.process.deleteLater()
        self.process = None
        self.worker_closing = False
        self.phase_complete = False
        self.cancelling = False
        self.save()
        self.refresh_table()
        self.selection_changed()
        if not self.closing:
            QTimer.singleShot(0, self.launch_stage if handoff else self.start_next)

    @staticmethod
    def recover_committed_result(job):
        """Recover a final file written just before a process died or its event was lost."""
        if not job.get("folder"):
            return False
        folder = Path(job["folder"])
        try:
            manifest = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            transcript = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
            target = local_file(folder, transcript["text_file"])
            if not target.is_file():
                return False
            job["transcript"] = str(target)
            if manifest.get("status") == "completed" and (not job.get("diarize") or manifest.get("diarization_status") == "completed"):
                job.update(status="completed", message="저장 완료된 결과를 복구했습니다.")
                return True
            if job.get("diarize"):
                job["stage"] = "diarization"
        except (OSError, ValueError, KeyError):
            pass
        return False

    @staticmethod
    def mark_manifest(job, status):
        if not job.get("folder"):
            return
        path = Path(job["folder"]) / "metadata.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed" or job.get("stage") == "diarization":
                manifest.update(status=status, error=job.get("message", ""))
                if job.get("stage") == "diarization":
                    manifest["diarization_status"] = status
                atomic_json(path, manifest)
        except (OSError, ValueError):
            pass

    def cancel_current(self):
        if self.process and self.active:
            self.cancelling = True
            self.paused = True
            self.pause_button.setText("대기열 계속")
            self.process.kill()

    def retry_selected(self):
        job = self.selected_job()
        if job and job["status"] in ("failed", "cancelled", "needs_setup"):
            if job.get("stage") == "diarization" and job.get("folder") and (Path(job["folder"]) / "transcript.json").is_file():
                self.queue_diarization()
            else:
                self.add_files([job["source"]])

    def closeEvent(self, event: QCloseEvent):
        self.closing = True
        self.save()
        if self.process:
            self.process.kill()
            self.process.waitForFinished(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    STATE.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(STATE / "app.lock"))
    if not lock.tryLock(100):
        QMessageBox.information(None, "회의 보관함", "회의 보관함이 이미 실행 중입니다. 작업 표시줄의 기존 창을 열어 주세요.")
        return 0
    app.setApplicationName("회의 보관함")
    app.setOrganizationName("LocalMeetingSTT")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
