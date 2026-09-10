from __future__ import annotations

import gc
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .audio import SAMPLE_RATE, audio_chunks
from .runtime import MODELS, configure_runtime
from .storage import atomic_json, copy_recording, create_folder, inspect_recording, safe_stem


def clock_text(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{seconds % 3600 // 60:02}:{seconds % 60:02}"


class Transcriber:
    def __init__(self):
        self.model = None
        self.model_name = None

    def load(self, name, emit):
        if self.model is not None and self.model_name == name:
            return self.model
        configure_runtime()
        from faster_whisper import WhisperModel, download_model

        self.model = None
        gc.collect()
        emit("status", message="모델 준비 중 · 최초 사용 시 다운로드가 필요합니다.")
        directory = MODELS / name
        # A completed download is loaded directly, so subsequent runs work offline.
        marker = directory / ".ready"
        if not marker.exists():
            directory.mkdir(parents=True, exist_ok=True)
            download_model(name, output_dir=str(directory))
        emit("status", message="GPU에 모델을 불러오는 중")
        self.model = WhisperModel(str(directory), device="cuda", compute_type="int8_float16", cpu_threads=4, num_workers=1)
        marker.write_text("ready", encoding="ascii")
        self.model_name = name
        return self.model

    def run(self, job, emit):
        started = time.monotonic()
        manifest = None
        folder = None
        try:
            emit("status", message="녹음 날짜와 오디오 확인 중")
            recording = inspect_recording(Path(job["source"]), job.get("timezone", "Asia/Seoul"))
            folder = create_folder(Path(job["output"]).resolve(), recording)
            emit("prepared", folder=str(folder), recording=asdict(recording))
            manifest = {
                "schema": 1, "status": "copying", "recording": asdict(recording),
                "model": job["model"], "language": "ko", "device": "cuda", "compute_type": "int8_float16",
                "timezone": job.get("timezone", "Asia/Seoul"), "timestamps": job.get("timestamps", True),
                "hotwords": job.get("hotwords", ""), "created_at": datetime.now().astimezone().isoformat(),
            }
            atomic_json(folder / "metadata.json", manifest)
            source = Path(recording.path)
            saved_audio = folder / source.name
            emit("status", message="결과 폴더에 녹음본 복사 중")
            manifest["sha256"] = copy_recording(recording, saved_audio, lambda p: emit("copy_progress", progress=p))
            manifest["audio_file"] = saved_audio.name
            manifest["status"] = "loading"
            atomic_json(folder / "metadata.json", manifest)
            model = self.load(job["model"], emit)
            manifest["status"] = "transcribing"
            atomic_json(folder / "metadata.json", manifest)
            stem = safe_stem(source.stem)
            partial = folder / f"{stem}.partial.txt"
            transcript = folder / f"{stem}.txt"
            emit("status", message="한국어 전사 중")
            segment_count = 0
            duration = 0
            structured = []
            header = f"녹음: {source.name}\n녹음 시점: {recording.recorded_at}\n날짜 출처: {recording.date_source}\n"
            if recording.date_warning:
                header += f"날짜 확인: {recording.date_warning}\n"
            header += f"모델: {job['model']} / 한국어\n"
            with partial.open("w", encoding="utf-8-sig") as writer:
                writer.write(header + "\n")
                writer.flush()
                for offset, samples in audio_chunks(saved_audio):
                    segments, _ = model.transcribe(
                        samples, language="ko", task="transcribe", beam_size=5,
                        vad_filter=True, vad_parameters={"min_silence_duration_ms": 700},
                        condition_on_previous_text=False,
                        word_timestamps=True,
                        hotwords=job.get("hotwords") or None,
                    )
                    for segment in segments:
                        text = segment.text.strip()
                        if not text:
                            continue
                        begin, end = offset + segment.start, offset + segment.end
                        words = [{"start": offset + word.start, "end": offset + word.end,
                                  "word": word.word, "probability": word.probability}
                                 for word in (getattr(segment, "words", None) or [])]
                        structured.append({"start": begin, "end": end, "text": segment.text, "words": words})
                        line = f"[{clock_text(begin)} → {clock_text(end)}] {text}" if job.get("timestamps", True) else text
                        writer.write(line + "\n")
                        writer.flush()
                        segment_count += 1
                        emit("segment", text=line, seconds=end, duration=recording.duration)
                    duration = offset + len(samples) / SAMPLE_RATE
                    emit("progress", seconds=duration, duration=recording.duration)
                if not segment_count:
                    writer.write("[인식된 발화가 없습니다. 녹음의 음량과 내용을 확인해 주세요.]\n")
            partial.replace(transcript)
            atomic_json(folder / "transcript.json", {"schema": 1, "header": header, "segments": structured,
                        "timestamps": job.get("timestamps", True), "text_file": transcript.name,
                        "duration": duration, "audio_file": saved_audio.name})
            manifest.update(status="completed", transcript_file=transcript.name, segments=segment_count,
                            decoded_seconds=round(duration, 3), elapsed_seconds=round(time.monotonic() - started, 2))
            if job.get("diarize"):
                manifest.update(status="awaiting_diarization", diarization_status="pending")
            atomic_json(folder / "metadata.json", manifest)
            emit("asr_done" if job.get("diarize") else "done", folder=str(folder), transcript=str(transcript), segments=segment_count,
                 elapsed=manifest["elapsed_seconds"], duration=duration)
        except Exception as error:
            if manifest is not None and folder is not None:
                manifest.update(status="failed", error=str(error))
                try:
                    atomic_json(folder / "metadata.json", manifest)
                except OSError:
                    pass
            raise
