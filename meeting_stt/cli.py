"""Optional command line entry point for the same pipeline used by the GUI."""
import argparse
import json
import sys
import uuid

from .engine import Transcriber
from .runtime import ROOT


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Local Korean transcription with date-based archiving")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output", default=str(ROOT / "output"))
    parser.add_argument("--model", choices=("large-v3", "turbo"), default="large-v3")
    parser.add_argument("--hotwords", default="")
    parser.add_argument("--no-timestamps", action="store_true")
    args = parser.parse_args()
    engine = Transcriber()
    failed = False
    for source in args.files:
        job = {"id": uuid.uuid4().hex, "source": source, "output": args.output, "model": args.model,
               "hotwords": args.hotwords, "timestamps": not args.no_timestamps}
        try:
            engine.run(job, lambda event, **data: print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True))
        except Exception as error:
            failed = True
            print(json.dumps({"event": "error", "message": str(error)}, ensure_ascii=False), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
