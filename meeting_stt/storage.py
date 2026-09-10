from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dateutil.parser import isoparse

SUPPORTED = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mov", ".mkv", ".webm", ".amr", ".3gp", ".aiff", ".aif"}


@dataclass
class Recording:
    path: str
    recorded_at: str
    date_source: str
    date_precision: str
    date_warning: str
    duration: float
    tags: dict
    size: int
    mtime_ns: int


def parse_date(value: str, timezone: str):
    value = str(value).strip().strip("\x00")
    # A year alone is usually an album tag, not a recording date.
    if not re.match(r"^\d{4}[-:/]?\d{2}[-:/]?\d{2}", value):
        return None
    if re.match(r"^\d{4}:\d{2}:\d{2}", value):
        value = value[:10].replace(":", "-") + value[10:]
    value = value.replace("/", "-")
    try:
        parsed = isoparse(value)
        if parsed.year < 1970 or parsed.year > datetime.now().year + 1:
            return None
        precision = "day" if len(value) <= 10 else "second"
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo(timezone))
        return parsed.astimezone(ZoneInfo(timezone)), precision
    except (ValueError, OverflowError):
        return None


def choose_date(tags: dict, path: Path, timezone: str):
    """Choose a recording date, retaining the exact evidence and fallback reason."""
    candidates = []
    for scope, values in tags.items():
        values = {key.lower(): str(value) for key, value in values.items()}
        if "origination_date" in values:
            value = values["origination_date"]
            if values.get("origination_time"):
                value += "T" + values["origination_time"]
            candidates.append((0, f"{scope}.origination_date", value))
        keys = ("creation_time", "com.apple.quicktime.creationdate", "date_recorded", "recording_time", "tdrc", "date", "icrd", "©day")
        for priority, key in enumerate(keys, 1):
            if key in values:
                candidates.append((priority, f"{scope}.{key}", values[key]))
    for _, source, value in sorted(candidates, key=lambda item: item[0]):
        parsed = parse_date(value, timezone)
        if parsed:
            timestamp, precision = parsed
            warning = "날짜만 기록되어 있어 시각은 00:00:00으로 표시합니다." if precision == "day" else ""
            return timestamp.isoformat(), source, precision, warning
    stat = path.stat()
    timestamp = getattr(stat, "st_birthtime", stat.st_ctime if os.name == "nt" else stat.st_mtime)
    source = "filesystem.creation_time" if os.name == "nt" or hasattr(stat, "st_birthtime") else "filesystem.modified_time"
    warning = "내부 녹음 날짜가 없어 파일 시스템 날짜를 사용했습니다. 복사·다운로드 시 바뀐 날짜일 수 있습니다."
    return datetime.fromtimestamp(timestamp, ZoneInfo(timezone)).isoformat(), source, "second", warning


def inspect_recording(path: Path, timezone="Asia/Seoul") -> Recording:
    import av
    import mutagen

    path = path.resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in SUPPORTED:
        raise ValueError("지원하는 녹음·동영상 파일을 선택해 주세요.")
    tags = {}
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("파일에 오디오 트랙이 없습니다.")
        tags["container"] = dict(container.metadata)
        for index, stream in enumerate(container.streams):
            tags[f"stream{index}"] = dict(stream.metadata)
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        audio = container.streams.audio[0]
        if not duration and audio.duration is not None:
            duration = float(audio.duration * audio.time_base)
    try:
        media = mutagen.File(path)
        if media is not None and media.tags:
            tags["audio_tags"] = {str(k): str(v[0] if isinstance(v, list) and v else v) for k, v in media.tags.items()}
    except (mutagen.MutagenError, ValueError, OSError):
        pass
    recorded_at, source, precision, warning = choose_date(tags, path, timezone)
    stat = path.stat()
    return Recording(str(path), recorded_at, source, precision, warning, duration, tags, stat.st_size, stat.st_mtime_ns)


def safe_stem(name: str):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")[:70].rstrip(" .") or "recording"
    if re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", name, re.I):
        name = "_" + name
    return name


def create_folder(root: Path, recording: Recording) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    date = datetime.fromisoformat(recording.recorded_at).strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{date}_{safe_stem(Path(recording.path).stem)}"
    # mkdir is atomic: never overwrite an existing recording, including from another app instance.
    for counter in range(10000):
        folder = root / (name if counter == 0 else f"{name}_{counter + 1:02d}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            continue
    raise RuntimeError("같은 이름의 결과 폴더가 너무 많습니다.")


def atomic_json(path: Path, data: dict):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def copy_recording(recording: Recording, target: Path, progress):
    import shutil

    source = Path(recording.path)
    partial = target.with_name(target.name + ".copying")
    digest = hashlib.sha256()
    copied = 0
    initial = source.stat()
    if initial.st_size != recording.size or initial.st_mtime_ns != recording.mtime_ns:
        raise RuntimeError("날짜를 읽은 뒤 원본 파일이 변경되었습니다. 녹음이 끝난 파일로 다시 시도해 주세요.")
    with source.open("rb") as reader, partial.open("xb") as writer:
        while chunk := reader.read(4 * 1024 * 1024):
            writer.write(chunk)
            digest.update(chunk)
            copied += len(chunk)
            progress(copied / max(recording.size, 1))
        writer.flush()
        os.fsync(writer.fileno())
    final = source.stat()
    if final.st_size != initial.st_size or final.st_mtime_ns != initial.st_mtime_ns or copied != initial.st_size:
        raise RuntimeError("복사 중 원본 파일이 변경되었습니다. 녹음이 끝난 파일로 다시 시도해 주세요.")
    shutil.copystat(source, partial)
    partial.replace(target)
    return digest.hexdigest()
