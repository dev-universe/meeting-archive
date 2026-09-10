from __future__ import annotations

from pathlib import Path

SAMPLE_RATE = 16000
CHUNK_SECONDS = 300


def split_position(pcm, rate=SAMPLE_RATE):
    """Prefer a quiet point in the last 20 s; never discard or repeat samples."""
    import numpy as np

    window = int(rate * 0.4)
    start = max(0, len(pcm) - rate * 20)
    tail = pcm[start:]
    count = len(tail) // window
    if not count:
        return len(pcm)
    blocks = tail[:count * window].astype(np.float32).reshape(count, window)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    quiet = np.flatnonzero(rms < 250)
    if len(quiet):
        return start + int(quiet[-1]) * window + window // 2
    return len(pcm)


def audio_chunks(path: Path, chunk_seconds=CHUNK_SECONDS):
    import av
    import numpy as np

    max_samples = int(chunk_seconds * SAMPLE_RATE)
    pending = bytearray()
    offset = 0

    def drain(final=False):
        nonlocal offset
        while len(pending) >= max_samples * 2 or (final and pending):
            count = min(len(pending) // 2, max_samples)
            pcm = np.frombuffer(bytes(pending[:count * 2]), dtype=np.int16)
            cut = split_position(pcm) if not final and count == max_samples else count
            cut = max(1, cut)
            samples = pcm[:cut].astype(np.float32) / 32768.0
            del pending[:cut * 2]
            yield offset / SAMPLE_RATE, samples
            offset += cut

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                pending.extend(converted.to_ndarray().tobytes())
                yield from drain()
        for converted in resampler.resample(None):
            pending.extend(converted.to_ndarray().tobytes())
        yield from drain(final=True)
