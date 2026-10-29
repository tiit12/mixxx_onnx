"""Create reference logits for the DownbeatDetector unit test.

The test signal is deterministic and generated identically in
src/test/downbeatdetectortest.cpp. The reference logits are computed with the
original PyTorch implementation of Beat This!.

Usage:
    python make_test_reference.py --sample-rate 44100 \
        --out src/test/beatthis/reference_logits_44100.bin
"""

import argparse

import numpy as np

SAMPLE_RATE = 44100
SECONDS = 70.0
BPM = 126.0
# Bass note per bar, cycling every 4 bars
BASS_HZ = [55.0, 73.42, 82.41, 65.41]


def test_signal(sample_rate=SAMPLE_RATE):
    n = int(SECONDS * sample_rate)
    signal = np.zeros(n)
    beat_frames = 60.0 / BPM * sample_rate
    num_beats = int(n / beat_frames)
    for i in range(num_beats):
        start = int(round(i * beat_frames))
        # Kick, accented on the downbeat
        length = min(int(0.15 * sample_rate), n - start)
        t = np.arange(length) / sample_rate
        gain = 1.0 if i % 4 == 0 else 0.6
        signal[start : start + length] += (
            gain * np.exp(-t / 0.03) * np.sin(2 * np.pi * 60.0 * t)
        )
        # Snare on 2 and 4
        if i % 4 in (1, 3):
            length = min(int(0.1 * sample_rate), n - start)
            t = np.arange(length) / sample_rate
            signal[start : start + length] += (
                0.3 * np.exp(-t / 0.02) * np.sin(2 * np.pi * 190.0 * t)
            )
        # Bass note changing with every bar
        if i % 4 == 0:
            length = min(int(round(4 * beat_frames)), n - start)
            t = np.arange(length) / sample_rate
            freq = BASS_HZ[(i // 4) % 4]
            signal[start : start + length] += 0.25 * np.sin(2 * np.pi * freq * t)
    return (0.5 * signal).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="small0")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor

    signal = test_signal(args.sample_rate)
    frames = Audio2Frames(args.checkpoint, "cpu")
    beat, downbeat = frames(signal.astype(np.float64), args.sample_rate)
    beats, downbeats = Postprocessor("minimal")(beat, downbeat)
    print(f"{len(beat)} frames, {len(beats)} beats, {len(downbeats)} downbeats")
    beat_len = 60.0 / BPM
    phases = np.round(downbeats / beat_len).astype(int) % 4
    print("downbeat positions in bar:", np.bincount(phases, minlength=4))
    np.stack([beat.numpy(), downbeat.numpy()]).astype("<f4").tofile(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
