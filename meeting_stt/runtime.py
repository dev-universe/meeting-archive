from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
STATE = ROOT / "state"
_dll_handles = []


def configure_runtime():
    """Use project-local CUDA DLLs without modifying the machine's PATH."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if os.name == "nt":
        nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
        directories = [p for pattern in ("*/bin", "*/lib") for p in nvidia.glob(pattern) if p.is_dir()]
        if directories:
            os.environ["PATH"] = os.pathsep.join(map(str, directories)) + os.pathsep + os.environ.get("PATH", "")
            for directory in directories:
                _dll_handles.append(os.add_dll_directory(str(directory)))


def python_console():
    executable = Path(sys.executable)
    return str(executable.with_name("python.exe") if executable.name.lower() == "pythonw.exe" else executable)
