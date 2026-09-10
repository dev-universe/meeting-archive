"""Optional pyannote stage; imported by the dedicated PyTorch worker only."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from .runtime import MODELS, ROOT, STATE
from .speakers import local_file, merge_speakers, save_speakers
from .storage import atomic_json

MODEL_ID = "pyannote/speaker-diarization-community-1"
MODEL_DIR = MODELS / "speaker-diarization-community-1"
DIARIZATION_PYTHON = ROOT / ".venv-diarization" / "Scripts" / "python.exe"


def runtime_ready():
    return DIARIZATION_PYTHON.is_file() and (STATE / "diarization-runtime.json").is_file()


def model_ready():
    try:
        marker = json.loads((MODEL_DIR / ".ready.json").read_text(encoding="utf-8"))
        return bool(marker.get("files")) and all((MODEL_DIR / name).is_file() and (MODEL_DIR / name).stat().st_size == size for name, size in marker["files"].items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def offline_environment():
    # Set before importing pyannote/huggingface, in this process only.
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["OTEL_SDK_DISABLED"] = "true"
    os.environ["OMP_NUM_THREADS"] = "4"


def load_pipeline():
    import warnings
    offline_environment()
    # Decoding uses PyAV and an in-memory tensor. TorchCodec's optional DLLs are unnecessary.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\ntorchcodec is not installed correctly.*")
        from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(str(MODEL_DIR), token=False)
    if pipeline is None:
        raise RuntimeError("화자 모델을 불러오지 못했습니다. 모델 준비에서 다시 다운로드해 주세요.")
    pipeline.segmentation_batch_size = 1
    pipeline.embedding_batch_size = 1
    return pipeline


def prepare_model(token, emit):
    if not token or not token.startswith("hf_"):
        raise ValueError("Hugging Face의 읽기 권한 토큰을 입력해 주세요.")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    os.environ["OTEL_SDK_DISABLED"] = "true"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download
    emit("status", message="Hugging Face에서 모델 다운로드 중 · 수 분 걸릴 수 있습니다.")
    snapshot_download(MODEL_ID, local_dir=str(MODEL_DIR), token=token, max_workers=2)
    # The token is used for download only; it is not saved or passed to the offline loader.
    token = None
    from huggingface_hub import constants
    constants.HF_HUB_OFFLINE = True
    emit("status", message="다운로드 완료 · 로컬 모델 및 GPU 확인 중")
    import torch
    pipeline = load_pipeline()
    pipeline.to(torch.device("cuda"))
    with torch.inference_mode():
        pipeline({"waveform": torch.zeros(1, 16000 * 3), "sample_rate": 16000})
    files = {str(p.relative_to(MODEL_DIR)): p.stat().st_size for p in MODEL_DIR.rglob("*")
             if p.is_file() and ".cache" not in p.parts and not p.name.startswith(".")}
    atomic_json(MODEL_DIR / ".ready.json", {"model": MODEL_ID, "files": files})
    emit("setup_done", message="화자 모델 준비 완료 · 이후에는 오프라인으로 사용할 수 있습니다.")


def safe_error(error, token=None):
    message = str(error)
    if token:
        message = message.replace(token, "[토큰 숨김]")
    message = re.sub(r"hf_[A-Za-z0-9]+", "[토큰 숨김]", message)
    lowered = message.lower()
    if any(term in lowered for term in ("gated", "403", "401", "unauthorized", "access to model")):
        return "모델 접근 권한을 확인해 주세요. 가입·이메일 확인 후 모델 페이지의 이용 조건에 동의하고, 해당 계정의 읽기 토큰을 입력해야 합니다."
    if "out of memory" in lowered:
        return "GPU 메모리가 부족합니다. 다른 GPU 앱을 닫거나 화자 분석 장치를 CPU로 바꿔 다시 시도해 주세요. 일반 전사본은 보존됩니다."
    return message[-1600:]


def run_diarization(job, emit, pipeline_factory=None):
    folder = Path(job["folder"]).resolve()
    path = folder / "metadata.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    transcript = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
    started = time.monotonic()
    mapped = None
    waveform = None
    pcm_path = folder / ".diarization.f32"
    try:
        if pipeline_factory is None and not model_ready():
            manifest.update(status="awaiting_diarization", diarization_status="setup_required")
            atomic_json(path, manifest)
            emit("needs_setup", folder=str(folder), transcript=str(local_file(folder, transcript["text_file"])),
                 message="일반 전사 완료 · 화자 모델 준비 후 이 항목의 화자 분석을 실행해 주세요.")
            return
        offline_environment()
        import numpy as np
        import torch
        from .audio import audio_chunks, SAMPLE_RATE

        torch.set_num_threads(4)
        device = job.get("diarization_device", "cuda")
        if device not in ("cuda", "cpu"):
            raise ValueError("화자 분석 장치는 GPU 또는 CPU여야 합니다.")
        manifest.update(status="diarizing", diarization_status="running")
        atomic_json(path, manifest)
        emit("status", message="화자 분석용 오디오 준비 중")
        # A disk-backed float buffer avoids concatenating hours of audio in RAM twice.
        samples = 0
        with pcm_path.open("wb") as output:
            for _, chunk in audio_chunks(local_file(folder, transcript["audio_file"])):
                output.write(chunk.astype(np.float32, copy=False).tobytes())
                samples += len(chunk)
        if not samples:
            raise ValueError("분석할 오디오가 없습니다.")
        mapped = np.memmap(pcm_path, dtype="float32", mode="r+", shape=(samples,))
        waveform = torch.from_numpy(mapped).unsqueeze(0)
        emit("status", message=f"화자 모델 로딩 중 · {'GPU' if device == 'cuda' else 'CPU'}")
        pipeline = (pipeline_factory or load_pipeline)()
        pipeline.to(torch.device(device))
        options = {}
        speakers = int(job.get("num_speakers", 0))
        if speakers:
            if not 1 <= speakers <= 30:
                raise ValueError("참석자 수는 1~30명 또는 자동으로 설정해 주세요.")
            options["num_speakers"] = speakers
        last = [0.0]

        def hook(step, artifact=None, file=None, total=None, completed=None):
            now = time.monotonic()
            if now - last[0] < 0.4 and completed != total:
                return
            last[0] = now
            names = {"segmentation": "발화 구간 분석", "embeddings": "목소리 특징 분석", "clustering": "화자 묶기", "discrete_diarization": "화자 구간 정리"}
            percent = f" {completed / total:.0%}" if total and completed is not None else ""
            emit("status", message="화자 분석 중 · " + names.get(step, "결과 계산") + percent)

        with torch.inference_mode():
            output = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, hook=hook, **options)
        turns = [{"start": float(turn.start), "end": float(turn.end), "speaker": label}
                 for turn, _, label in output.speaker_diarization.itertracks(yield_label=True)]
        exclusive = [{"start": float(turn.start), "end": float(turn.end), "speaker": label}
                     for turn, _, label in output.exclusive_speaker_diarization.itertracks(yield_label=True)]
        emit("status", message="전사 내용과 화자 연결 중")
        result = merge_speakers(transcript, turns, exclusive)
        result.update(device=device, num_speakers_requested=speakers, elapsed_seconds=round(time.monotonic() - started, 2))
        target = save_speakers(folder, result)
        manifest.update(status="completed", diarization_status="completed", speaker_count=len(result["speaker_names"]),
                        diarization_elapsed_seconds=result["elapsed_seconds"], diarization_device=device)
        manifest.pop("diarization_error", None)
        atomic_json(path, manifest)
        emit("done", folder=str(folder), transcript=str(target), segments=len(result["utterances"]),
             speaker_count=len(result["speaker_names"]), elapsed=result["elapsed_seconds"], stage="diarization")
    except Exception as error:
        manifest.update(status="diarization_failed", diarization_status="failed", diarization_error=safe_error(error))
        atomic_json(path, manifest)
        raise
    finally:
        waveform = None
        if mapped is not None:
            mapped._mmap.close()
            mapped = None
        try:
            pcm_path.unlink(missing_ok=True)
        except OSError:
            pass
