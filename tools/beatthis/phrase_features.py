"""Print the per-bar band energies used by PhraseAnalyzer for a track.

Uses the same STFT and frequency bands as DownbeatDetector in Mixxx, and the
downbeats of Beat This! as bars. The energies are printed in units of 6 dB
relative to the loudest bars (95th percentile), like in PhraseAnalyzer.

Usage:
    python phrase_features.py track.mp3 [--first-bar 0] [--bars 400]
"""

import argparse

import numpy as np

from export_onnx import HOP, N_FFT, SAMPLE_RATE, load_signal, stft_magnitude

LOW_MAX_HZ = 150
MID_MAX_HZ = 2000
DB_PER_UNIT = 6.0


def band_energies(mag):
    hz_per_bin = SAMPLE_RATE / N_FFT
    low_end = int(LOW_MAX_HZ / hz_per_bin) + 1
    mid_end = int(MID_MAX_HZ / hz_per_bin) + 1
    power = mag.astype(np.float64) ** 2
    return np.stack(
        [
            power[:, 1:low_end].sum(axis=1),
            power[:, low_end:mid_end].sum(axis=1),
            power[:, mid_end:].sum(axis=1),
        ],
        axis=1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--first-bar", type=int, default=0)
    parser.add_argument("--bars", type=int, default=400)
    args = parser.parse_args()

    from beat_this.inference import Audio2Beats

    signal = load_signal(args.audio)
    energies = band_energies(stft_magnitude(signal))
    fps = SAMPLE_RATE / HOP
    _, downbeats = Audio2Beats("small0", "cpu")(signal, SAMPLE_RATE)

    bars = []
    for start, end in zip(downbeats[:-1], downbeats[1:]):
        frames = energies[int(round(start * fps)) : int(round(end * fps))]
        bars.append(10 * np.log10(frames.mean(axis=0) + 1e-10))
    bars = np.array(bars)
    loudest = np.percentile(bars, 95, axis=0, method="lower")
    units = np.clip((bars - loudest) / DB_PER_UNIT, -5, 0.5)

    print(f"{len(bars)} bars, tempo {60 * 4 / np.median(np.diff(downbeats)):.1f} BPM")
    print(" bar   time    low   mid  high")
    last = min(len(bars), args.first_bar + args.bars)
    for i in range(args.first_bar, last):
        low, mid, high = units[i]
        marker = "-" if i % 8 == 0 else " "
        print(
            f"{i:4d}{marker} {downbeats[i] // 60:2.0f}:{downbeats[i] % 60:04.1f}"
            f" {low:5.2f} {mid:5.2f} {high:5.2f}"
        )


if __name__ == "__main__":
    main()
