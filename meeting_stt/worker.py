"""One isolated GPU process, reused while a queue is active. JSON lines over stdio."""
import json
import sys
import traceback

from .engine import Transcriber


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    engine = Transcriber()
    for line in sys.stdin:
        job = json.loads(line)

        def emit(event, **data):
            print(json.dumps({"event": event, "id": job["id"], **data}, ensure_ascii=False), flush=True)

        try:
            engine.run(job, emit)
            if job.get("diarize"):
                # Exit before the GUI starts PyTorch in its separate environment.
                return 0
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit("error", message=f"{type(error).__name__}: {error}")
            # A CUDA error can poison the process; recycle it before the next file.
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
