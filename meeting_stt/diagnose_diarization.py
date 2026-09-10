import json

from .diarization import model_ready, offline_environment
from .runtime import STATE
from .storage import atomic_json


def main():
    offline_environment()
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\ntorchcodec is not installed correctly.*")
        import torch
        import pyannote.audio
        from pyannote.audio import Pipeline
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")
    layer = torch.nn.Conv1d(1, 8, 3).cuda()
    layer(torch.zeros(1, 1, 16000, device="cuda"))
    torch.cuda.synchronize()
    data = {"torch": torch.__version__, "pyannote": pyannote.audio.__version__,
            "gpu": torch.cuda.get_device_name(0), "runtime_ready": True, "model_ready": model_ready()}
    STATE.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE / "diarization-runtime.json", data)
    print(json.dumps(data))


if __name__ == "__main__":
    main()
