"""Small protocol peer for queue lifecycle tests; never imported by the app."""
import json
import os
import sys
import time
from pathlib import Path

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
for line in sys.stdin:
    job = json.loads(line)
    name = Path(job["source"]).stem
    if job.get("stage") == "diarization":
        folder = Path(job["folder"])
        if job.get("hotwords") == "diarization_crash":
            os._exit(24)
        if job.get("hotwords") == "diarization_slow":
            time.sleep(10)
        metadata = json.loads((folder / "metadata.json").read_text())
        metadata["diarization_pid"] = os.getpid()
        metadata["status"] = "completed"
        (folder / "metadata.json").write_text(json.dumps(metadata))
        print(json.dumps({"event": "done", "id": job["id"], "folder": str(folder), "transcript": str(folder / "result.txt"), "segments": 1, "speaker_count": 2, "elapsed": 0.1}), flush=True)
        sys.exit(0)
    if name == "crash":
        os._exit(23)
    if name == "error":
        print(json.dumps({"event": "error", "id": job["id"], "message": "invalid test audio"}), flush=True)
        sys.exit(1)
    folder = Path(job["output"]) / job["id"]
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({"status": "transcribing"}))
    print(json.dumps({"event": "prepared", "id": job["id"], "folder": str(folder), "recording": {}}), flush=True)
    time.sleep(10 if name == "slow" else 0.1)
    transcript = folder / "result.txt"
    transcript.write_text("한국어 테스트", encoding="utf-8-sig")
    (folder / "metadata.json").write_text(json.dumps({"status": "completed", "asr_pid": os.getpid()}))
    (folder / "transcript.json").write_text(json.dumps({"segments": [], "audio_file": "source.wav", "text_file": "result.txt"}))
    if name == "staged":
        print(json.dumps({"event": "asr_done", "id": job["id"], "folder": str(folder), "transcript": str(transcript), "segments": 1, "elapsed": 0.1}), flush=True)
        sys.exit(0)
    print(json.dumps({"event": "done", "id": job["id"], "folder": str(folder), "transcript": str(transcript), "segments": 1, "elapsed": 0.1}), flush=True)
