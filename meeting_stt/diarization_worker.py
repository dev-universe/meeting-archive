"""One stage per process. Tokens arrive over stdin and are never persisted."""
import json
import sys

from .diarization import prepare_model, run_diarization, safe_error


def main():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    job = json.loads(sys.stdin.readline())
    token = job.pop("token", None)

    def emit(event, **data):
        print(json.dumps({"event": event, "id": job.get("id"), **data}, ensure_ascii=False), flush=True)

    try:
        if job.get("mode") == "setup":
            prepare_model(token, emit)
        else:
            run_diarization(job, emit)
        return 0
    except Exception as error:
        emit("error", message=safe_error(error, token))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
