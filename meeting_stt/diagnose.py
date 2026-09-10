import json
import subprocess

from .runtime import configure_runtime


def main():
    configure_runtime()
    import av
    import ctranslate2
    from PySide6 import __version__ as qt_version

    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True, creationflags=flags).strip()
    types = ctranslate2.get_supported_compute_types("cuda")
    print(json.dumps({"gpu": gpu, "cuda_devices": ctranslate2.get_cuda_device_count(),
                      "compute_types": sorted(types), "ctranslate2": ctranslate2.__version__,
                      "pyav": av.__version__, "qt": qt_version}, indent=2))
    return 0 if "int8_float16" in types else 1


if __name__ == "__main__":
    raise SystemExit(main())
