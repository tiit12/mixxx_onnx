"""Export the Beat This! beat/downbeat tracker to ONNX for Mixxx.

The exported model takes the magnitude STFT of a 22050 Hz mono signal and
contains the mel filterbank and log compression, so that Mixxx only needs to
compute a plain STFT:

    input  "magnitude": float32 [1, 1500, 513]
           |rfft(hann_periodic(1024) * frame)| / sqrt(1024),
           hop 441, frames centered with reflect padding (torch.stft center=True)
    output "beat", "downbeat": float32 [1, 1500] logits (50 frames/second)

Usage:
    python export_onnx.py --checkpoint small0 --out beat_this_small.onnx [--audio file.wav]
"""

import argparse
import time

import numpy as np
import onnxruntime as ort
import torch

from beat_this.inference import Audio2Frames, load_model
from beat_this.preprocessing import LogMelSpect

SAMPLE_RATE = 22050
N_FFT = 1024
HOP = 441
CHUNK_FRAMES = 1500
N_BINS = N_FFT // 2 + 1


class MagnitudeToLogits(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.model = load_model(checkpoint, "cpu")
        mel = LogMelSpect()
        # (n_freqs, n_mels), see torchaudio.transforms.MelScale
        self.register_buffer("fb", mel.spect_class.mel_scale.fb.detach().clone())

    def forward(self, magnitude):
        mel = torch.matmul(magnitude, self.fb)
        out = self.model(torch.log1p(1000.0 * mel))
        return out["beat"], out["downbeat"]


def stft_magnitude(signal):
    """Reference for the C++ implementation in Mixxx."""
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    padded = np.pad(signal, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + len(signal) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP][:n_frames]
    mag = np.abs(np.fft.rfft(frames * window, axis=1))
    # torchaudio normalized="frame_length" -> torch.stft(normalized=True)
    return (mag / np.sqrt(N_FFT)).astype(np.float32)


BORDER_FRAMES = 6


def chunk_starts(n_frames):
    """Same as beat_this.inference.split_piece(avoid_short_end=True)."""
    step = CHUNK_FRAMES - 2 * BORDER_FRAMES
    starts = list(range(-BORDER_FRAMES, n_frames - BORDER_FRAMES, step))
    if n_frames > step:
        starts[-1] = n_frames - (CHUNK_FRAMES - BORDER_FRAMES)
    return starts


def onnx_logits(session, mag):
    """Reference for the chunking in Mixxx: zero padded chunks of 1500 frames,
    6 border frames are discarded and earlier chunks win in overlaps."""
    n_frames = len(mag)
    beat = np.full(n_frames, -1000.0, dtype=np.float32)
    downbeat = np.full(n_frames, -1000.0, dtype=np.float32)
    for start in reversed(chunk_starts(n_frames)):
        chunk = np.zeros((1, CHUNK_FRAMES, N_BINS), dtype=np.float32)
        src_begin, src_end = max(start, 0), min(start + CHUNK_FRAMES, n_frames)
        chunk[0, src_begin - start : src_end - start] = mag[src_begin:src_end]
        b, d = session.run(None, {"magnitude": chunk})
        dst_begin = start + BORDER_FRAMES
        dst_end = min(start + CHUNK_FRAMES - BORDER_FRAMES, n_frames)
        count = dst_end - dst_begin
        beat[dst_begin:dst_end] = b[0, BORDER_FRAMES : BORDER_FRAMES + count]
        downbeat[dst_begin:dst_end] = d[0, BORDER_FRAMES : BORDER_FRAMES + count]
    return beat, downbeat


def test_signal(seconds=95.0, bpm=128.0):
    """Synthetic 4/4 pattern with an accented downbeat and a bass note per bar."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    signal = 0.02 * np.random.default_rng(0).standard_normal(len(t))
    beat_len = 60.0 / bpm
    for i in range(int(seconds / beat_len)):
        start = int(i * beat_len * SAMPLE_RATE)
        n = min(int(0.15 * SAMPLE_RATE), len(t) - start)
        env = np.exp(-np.arange(n) / (0.03 * SAMPLE_RATE))
        gain = 1.0 if i % 4 == 0 else 0.5
        signal[start : start + n] += gain * env * np.sin(2 * np.pi * 60 * t[:n])
        if i % 4 == 0:
            n2 = min(int(beat_len * 2 * SAMPLE_RATE), len(t) - start)
            signal[start : start + n2] += 0.2 * np.sin(2 * np.pi * 110 * t[:n2])
    return signal.astype(np.float32)


def load_signal(path):
    import soundfile as sf
    import soxr

    signal, sr = sf.read(path, dtype="float32", always_2d=True)
    signal = signal.mean(axis=1)
    if sr != SAMPLE_RATE:
        signal = soxr.resample(signal, sr, SAMPLE_RATE)
    return signal.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="small0")
    parser.add_argument("--out", default="beat_this_small.onnx")
    parser.add_argument("--audio", help="optional audio file for verification")
    args = parser.parse_args()

    torch.manual_seed(0)
    module = MagnitudeToLogits(args.checkpoint).eval()
    n_params = sum(p.numel() for p in module.model.parameters())
    print(f"checkpoint {args.checkpoint}: {n_params / 1e6:.2f} M parameters")

    dummy = torch.rand(1, CHUNK_FRAMES, N_BINS)
    torch.onnx.export(
        module,
        (dummy,),
        args.out,
        input_names=["magnitude"],
        output_names=["beat", "downbeat"],
        opset_version=17,
        dynamo=False,
    )
    print(f"exported {args.out}")

    # 1. STFT reference matches torchaudio
    signal = load_signal(args.audio) if args.audio else test_signal()
    mag = stft_magnitude(signal)
    spect_ref = LogMelSpect()(torch.from_numpy(signal)).numpy()
    spect_ours = np.log1p(1000.0 * mag @ module.fb.numpy())
    print(f"log-mel max abs diff vs torchaudio: {np.abs(spect_ours - spect_ref).max():.2e}")

    # 2. Chunked ONNX inference matches PyTorch Beat This! on the same audio
    frames = Audio2Frames(args.checkpoint, "cpu")
    beat_ref, downbeat_ref = frames(signal, SAMPLE_RATE)
    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    beat, downbeat = onnx_logits(session, mag)
    print(
        f"ONNX vs PyTorch logits max abs diff ({len(mag)} frames, beat, downbeat): "
        f"{np.abs(beat - beat_ref.numpy()).max():.2e}, "
        f"{np.abs(downbeat - downbeat_ref.numpy()).max():.2e}"
    )
    padded = np.zeros((1, CHUNK_FRAMES, N_BINS), dtype=np.float32)
    padded[0] = mag[:CHUNK_FRAMES]

    # 3. CPU speed of one 30 s chunk
    for threads in (1, 2, 4):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        s = ort.InferenceSession(args.out, options, providers=["CPUExecutionProvider"])
        s.run(None, {"magnitude": padded})
        start = time.perf_counter()
        runs = 3
        for _ in range(runs):
            s.run(None, {"magnitude": padded})
        per_chunk = (time.perf_counter() - start) / runs
        # a 6 minute track needs ~13 chunks of 30 s (with overlap)
        print(f"{threads} thread(s): {per_chunk * 1000:.0f} ms per 30 s chunk, "
              f"~{per_chunk * 13:.1f} s per 6 min track")


if __name__ == "__main__":
    main()
