"""Dump the features of PhraseAnalyzer for tracks, to tune it without Mixxx.

Writes one <name>.features file per track, which is read by the disabled test
PhraseAnalyzerTest.DISABLED_FeatureFiles (see src/test/phraseanalyzertest.cpp):

    python dump_phrase_features.py --out features_dir track1.mp3 track2.mp3 ...
    set PHRASE_FEATURES_DIR=features_dir
    mixxx-test --gtest_also_run_disabled_tests --gtest_filter=*FeatureFiles

File format (little endian):
    int32 number of frames, float64 frames per second, float64 BPM,
    float64 position of the first downbeat in seconds,
    float32 values per frame: low, mid, high, 12 spectrum bands
The beats are a constant grid with the median beat period of Beat This!
"""

import argparse
import os
import struct

import numpy as np

from export_onnx import HOP, N_FFT, SAMPLE_RATE, load_signal, stft_magnitude
from phrase_features import band_energies

SPECTRUM_BANDS = 12
SPECTRUM_MIN_HZ = 30
SPECTRUM_MAX_HZ = 11000


def spectrum_bands(mag):
    """Same as DownbeatDetector::m_spectrumBandOfBin in Mixxx"""
    hz = np.arange(mag.shape[1]) * SAMPLE_RATE / N_FFT
    power = mag.astype(np.float64) ** 2
    result = np.zeros((mag.shape[0], SPECTRUM_BANDS))
    for b in range(1, mag.shape[1]):
        if hz[b] < SPECTRUM_MIN_HZ or hz[b] >= SPECTRUM_MAX_HZ:
            continue
        band = min(
            SPECTRUM_BANDS - 1,
            int(SPECTRUM_BANDS * np.log(hz[b] / SPECTRUM_MIN_HZ)
                / np.log(SPECTRUM_MAX_HZ / SPECTRUM_MIN_HZ)),
        )
        result[:, band] += power[:, b]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("tracks", nargs="+")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from beat_this.inference import Audio2Beats

    audio2beats = Audio2Beats("small0", "cpu")
    for track in args.tracks:
        signal = load_signal(track)
        mag = stft_magnitude(signal)
        features = np.concatenate([band_energies(mag), spectrum_bands(mag)], axis=1)
        beats, downbeats = audio2beats(signal, SAMPLE_RATE)
        bpm = 60.0 / np.median(np.diff(beats))
        name = os.path.splitext(os.path.basename(track))[0]
        with open(os.path.join(args.out, name + ".features"), "wb") as f:
            f.write(struct.pack("<iddd", len(features), SAMPLE_RATE / HOP, bpm, downbeats[0]))
            f.write(features.astype("<f4").tobytes())
        print(f"{name}: {bpm:.1f} BPM, first downbeat {downbeats[0]:.2f} s")


if __name__ == "__main__":
    main()
