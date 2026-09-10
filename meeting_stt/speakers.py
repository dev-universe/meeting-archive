"""Join word timestamps and whole-recording speaker turns without a GPU dependency."""
from __future__ import annotations

import bisect
import json
import math
import re
from pathlib import Path

from .storage import atomic_json


def timestamp(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{seconds % 3600 // 60:02}:{seconds % 60:02}"


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig") as stream:
        stream.write(text)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def local_file(folder: Path, name: str) -> Path:
    """Stored manifests may be edited; do not follow paths outside the result folder."""
    target = (folder / name).resolve()
    if target.parent != folder.resolve() or not name or Path(name).name != name:
        raise ValueError("결과 파일의 경로가 올바르지 않습니다.")
    return target


class TurnIndex:
    def __init__(self, turns):
        self.turns = sorted((dict(t) for t in turns if math.isfinite(t["start"]) and math.isfinite(t["end"]) and t["end"] > t["start"]), key=lambda t: t["start"])
        self.starts = [t["start"] for t in self.turns]
        self.max_ends = []
        largest = 0
        for turn in self.turns:
            largest = max(largest, turn["end"])
            self.max_ends.append(largest)

    def overlaps(self, start, end):
        left = bisect.bisect_right(self.max_ends, start)
        right = bisect.bisect_left(self.starts, end)
        return [(turn, min(end, turn["end"]) - max(start, turn["start"]))
                for turn in self.turns[left:right] if turn["end"] > start]


def merge_speakers(transcript, turns, exclusive_turns):
    regular = TurnIndex(turns)
    exclusive = TurnIndex(exclusive_turns)
    ordered_ids = list(dict.fromkeys(turn["speaker"] for turn in regular.turns + exclusive.turns))
    names = {speaker: f"화자 {index + 1}" for index, speaker in enumerate(ordered_ids)}
    utterances = []
    for segment in transcript["segments"]:
        boundary = len(utterances)
        words = segment.get("words") or [{"start": segment["start"], "end": segment["end"], "word": segment["text"]}]
        # Preserve the original text if an aligner left out punctuation or unaligned tokens.
        aligned_text = "".join(word["word"] for word in words)
        alignment_missing = re.sub(r"\s", "", aligned_text) != re.sub(r"\s", "", segment["text"])
        if alignment_missing:
            words = [{"start": segment["start"], "end": segment["end"], "word": segment["text"]}]
        for word in words:
            start, end = float(word["start"]), float(word["end"])
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise ValueError("전사 시간 정보가 올바르지 않습니다.")
            scores = {}
            for turn, overlap in exclusive.overlaps(start, max(start + 0.001, end)):
                scores[turn["speaker"]] = scores.get(turn["speaker"], 0) + overlap
            ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
            speaker = ranked[0][0] if ranked else None
            coverage = ranked[0][1] / max(end - start, 0.001) if ranked else 0
            uncertain = coverage < 0.65 or alignment_missing or not segment.get("words")
            if coverage < 0.35:
                speaker = None
            # Multiple speakers must be simultaneous, not merely adjacent within a word.
            candidates = [t for t, _ in regular.overlaps(start, max(start + 0.001, end))]
            overlap = any(a["speaker"] != b["speaker"] and min(end, a["end"], b["end"]) - max(start, a["start"], b["start"]) > 0.05
                          for i, a in enumerate(candidates) for b in candidates[i + 1:])
            piece = {"start": start, "end": end, "speaker": speaker, "text": word["word"],
                     "uncertain": uncertain, "overlap": overlap}
            if len(utterances) > boundary and all(utterances[-1][key] == piece[key] for key in ("speaker", "uncertain", "overlap")) and 0 <= start - utterances[-1]["end"] < 1.0:
                utterances[-1]["text"] += piece["text"]
                utterances[-1]["end"] = end
            else:
                # Leading whitespace belongs to the word, and is kept in JSON for exact reconstruction.
                utterances.append(piece)
    return {"schema": 1, "model": "pyannote/speaker-diarization-community-1", "speaker_names": names,
            "turns": turns, "exclusive_turns": exclusive_turns, "utterances": utterances}


def render_speakers(transcript, speakers):
    lines = [transcript["header"].rstrip(), "화자 구분: 사용 · [겹말] 동시 발화 / [확인 필요] 배정이 불확실한 구간", ""]
    for utterance in speakers["utterances"]:
        name = speakers["speaker_names"].get(utterance["speaker"], "화자 미확인")
        tags = (" [겹말]" if utterance["overlap"] else "") + (" [확인 필요]" if utterance["uncertain"] else "")
        time = f"[{timestamp(utterance['start'])} → {timestamp(utterance['end'])}] " if transcript.get("timestamps", True) else ""
        lines.append(f"{time}{name}{tags}: {utterance['text'].strip()}")
    if not speakers["utterances"]:
        lines.append("[인식된 발화가 없습니다.]")
    return "\n".join(lines) + "\n"


def save_speakers(folder: Path, speakers):
    transcript = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
    target = local_file(folder, transcript["text_file"])
    plain = target.with_name(target.stem + ".plain.txt")
    if not plain.exists():
        # Rebuild plain text from canonical ASR data, never back up a labelled re-export.
        lines = [transcript["header"].rstrip(), ""]
        for segment in transcript["segments"]:
            time = f"[{timestamp(segment['start'])} → {timestamp(segment['end'])}] " if transcript.get("timestamps", True) else ""
            lines.append(time + segment["text"].strip())
        atomic_text(plain, "\n".join(lines) + "\n")
    atomic_json(folder / "speakers.json", speakers)
    atomic_text(target, render_speakers(transcript, speakers))
    return target


def rename_speakers(folder: Path, names):
    speakers = json.loads((folder / "speakers.json").read_text(encoding="utf-8"))
    if set(names) != set(speakers["speaker_names"]):
        raise ValueError("화자 목록이 변경되었습니다. 창을 다시 열어 주세요.")
    cleaned = {key: re.sub(r"[\r\n\t]", " ", value).strip()[:60] for key, value in names.items()}
    if any(not value for value in cleaned.values()):
        raise ValueError("화자 이름을 비워 둘 수 없습니다.")
    speakers["speaker_names"] = cleaned
    return save_speakers(folder, speakers)
