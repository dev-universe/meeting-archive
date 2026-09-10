from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QVBoxLayout)

from .diarization import DIARIZATION_PYTHON, model_ready, runtime_ready, safe_error
from .runtime import ROOT
from .speakers import rename_speakers


class SpeakerSetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("화자 구분 준비")
        self.setMinimumWidth(590)
        self.process = None
        self.buffer = b""
        self.outcome = False
        self.payload = None
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        intro = QLabel("처음 한 번 모델을 받으면, 이후 화자 분석은 이 PC에서 오프라인으로 실행됩니다.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        steps = [
            ("1. Hugging Face 가입 · 이메일 확인", "가입 페이지", "https://huggingface.co/join"),
            ("2. Community-1 모델 이용 조건 동의", "모델 페이지", "https://huggingface.co/pyannote/speaker-diarization-community-1"),
            ("3. 읽기(Read) 권한 토큰 만들기", "토큰 만들기", "https://huggingface.co/settings/tokens"),
        ]
        for text, label, url in steps:
            row = QHBoxLayout()
            row.addWidget(QLabel(text), 1)
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, address=url: QDesktopServices.openUrl(QUrl(address)))
            row.addWidget(button)
            layout.addLayout(row)
        terms = QLabel("모델 접근 조건에는 연락처 공유와 안내 메일 동의가 포함됩니다. 모델 페이지에서 직접 확인해 주세요.")
        terms.setWordWrap(True)
        terms.setObjectName("muted")
        layout.addWidget(terms)
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("hf_로 시작하는 읽기 권한 토큰")
        self.token.setMaxLength(500)
        layout.addWidget(self.token)
        info = QLabel("토큰은 다운로드에만 사용합니다. 앱 설정·로그에 저장하지 않습니다.")
        info.setObjectName("muted")
        layout.addWidget(info)
        self.state_label = QLabel("모델 준비 완료" if model_ready() else "화자 모델이 아직 준비되지 않았습니다. 일반 전사는 바로 사용할 수 있습니다.")
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.download = QPushButton("모델 다운로드 및 확인")
        self.download.setObjectName("primary")
        self.download.clicked.connect(self.start_download)
        layout.addWidget(self.download)
        if not runtime_ready():
            self.state_label.setText("화자 분석 실행 환경이 없습니다. 프로젝트의 setup-diarization.ps1을 먼저 실행해 주세요.")
            self.download.setEnabled(False)
        close = QPushButton("닫기")
        close.clicked.connect(self.reject)
        layout.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)

    def start_download(self):
        token = self.token.text().strip()
        if not token.startswith("hf_"):
            self.state_label.setText("hf_로 시작하는 읽기 권한 토큰을 입력해 주세요.")
            return
        self.payload = {"mode": "setup", "token": token}
        self.token.clear()
        self.token.setEnabled(False)
        self.download.setEnabled(False)
        self.progress.show()
        self.outcome = False
        self.buffer = b""
        self.state_label.setText("모델 다운로드 시작 중")
        self.process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYANNOTE_METRICS_ENABLED", "0")
        environment.insert("HF_HUB_DISABLE_TELEMETRY", "1")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setProgram(str(DIARIZATION_PYTHON))
        self.process.setArguments(["-u", "-m", "meeting_stt.diarization_worker"])
        self.process.started.connect(self.send_payload)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(lambda: self.process.readAllStandardError() if self.process else None)
        self.process.finished.connect(self.finished_download)
        self.process.errorOccurred.connect(self.process_error)
        self.process.start()

    def send_payload(self):
        self.process.write((json.dumps(self.payload) + "\n").encode("utf-8"))
        self.payload = None
        self.process.closeWriteChannel()

    def read_output(self):
        if not self.process:
            return
        self.buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            try:
                event = json.loads(line)
                if "message" in event:
                    self.state_label.setText(safe_error(event["message"]))
                if event.get("event") in ("error", "setup_done"):
                    self.outcome = True
            except (ValueError, KeyError):
                pass

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.state_label.setText("화자 분석 실행 환경을 시작할 수 없습니다. setup-diarization.ps1을 확인해 주세요.")
            self.outcome = True
            self.finished_download(-1)

    def finished_download(self, code, *args):
        self.read_output()
        if not self.outcome:
            self.state_label.setText("모델 준비가 중단되었습니다. 토큰과 네트워크를 확인한 뒤 다시 시도해 주세요.")
        self.progress.hide()
        self.download.setEnabled(True)
        self.token.setEnabled(True)
        self.payload = None
        if self.process:
            self.process.deleteLater()
        self.process = None

    def reject(self):
        if self.process:
            self.process.kill()
            self.process.waitForFinished(3000)
        self.token.clear()
        self.payload = None
        super().reject()


class SpeakerNamesDialog(QDialog):
    def __init__(self, folder: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("화자 이름 변경")
        self.folder = folder
        self.setMinimumWidth(460)
        data = json.loads((folder / "speakers.json").read_text(encoding="utf-8"))
        layout = QVBoxLayout(self)
        description = QLabel("이 녹음의 화자 이름을 변경합니다. 저장하면 TXT에 즉시 반영됩니다.")
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        self.fields = {}
        for index, (speaker, name) in enumerate(data["speaker_names"].items(), 1):
            edit = QLineEdit(name)
            edit.setMaxLength(60)
            form.addRow(f"화자 {index}", edit)
            self.fields[speaker] = edit
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("이름 저장")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def save(self):
        try:
            rename_speakers(self.folder, {key: edit.text() for key, edit in self.fields.items()})
            self.accept()
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "이름 저장 실패", str(error))
