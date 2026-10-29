# Downbeat and phrase analysis (local fork)

Mixxx detects the downbeats of a track with the "Beat This!" neural network
(<https://github.com/CPJKU/beat_this>, MIT) and shows bars and phrases in the
waveform:

- thin lines: beats
- white lines: downbeats (first beat of a bar)
- orange lines: phrase starts (every 8 bars = 32 beats)

The neural network only decides which beat of the existing Mixxx beat grid is
the downbeat. Without ONNX Runtime (or without the model file) the first beat
after the initial silence is used instead.

## Building on Ubuntu (Ubuntu Studio)

ONNX Runtime is packaged in `universe` since Ubuntu 26.04:

```sh
sudo apt install libonnxruntime-dev
git clone <your-repo-url> mixxx
cd mixxx
git checkout phrase-analysis
tools/debian_buildenv.sh setup
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DONNXRUNTIME=ON
cmake --build build --parallel "$(nproc)"
build/mixxx-test --gtest_filter='Downbeat*:BeatsTest.*'
```

If the package is not available, extract an official release archive
(`onnxruntime-linux-x64-<version>.tgz` from
<https://github.com/microsoft/onnxruntime/releases>) and add
`-DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-<version>` to the
`cmake -S` command.

When started from the build directory, Mixxx loads the model from
`res/models/` of the source tree. `cmake --install build` installs it with
the other resources.

**Library safety:** Mixxx shares its settings and library database in
`~/.mixxx` with any other installed Mixxx version, and a newer version may
upgrade the database schema. Make a backup of `~/.mixxx` first, or test with
a separate profile:

```sh
build/mixxx --settings-path ~/mixxx-test-settings
```

## Controls

| Control | Meaning |
|---|---|
| `[ChannelN],beat_in_bar` | 1-4, 0 if unknown |
| `[ChannelN],bar_in_phrase` | 1-8, 0 if unknown |
| `[ChannelN],beats_until_phrase` | beats until the next phrase start, -1 if unknown |
| `[ChannelN],phrase_kind` | 0 unknown, 1 intro, 2 buildup, 3 drop, 4 breakdown, 5 outro |
| `[ChannelN],bar_shift_earlier` / `bar_shift_later` | move the downbeats by one beat |
| `[ChannelN],phrase_shift_earlier` / `phrase_shift_later` | move the phrase starts by one bar |
| `[ChannelN],bar_set_curpos` | make the beat at the play position a phrase-starting downbeat |

Keyboard shortcuts (en_US mapping): deck 1 Ctrl+Shift+A/S (bar), Q/W
(phrase), Z (set here); deck 2 Ctrl+Shift+J/K, I/P, M.

Manual corrections are stored with the track and are never overwritten by
the analyzer. The colors can be changed in the skin with `BarColor` and
`PhraseColor` next to `BeatColor`.

## Model and test data

The model and the reference data for `src/test/downbeatdetectortest.cpp` are
created with the Python tools in this directory:

```sh
python -m venv venv
venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install https://github.com/CPJKU/beat_this/archive/main.zip onnx onnxruntime soxr soundfile
venv/bin/python export_onnx.py --checkpoint small0 --out ../../res/models/beat_this_small.onnx
venv/bin/python make_test_reference.py --sample-rate 44100 --out ../../src/test/beatthis/reference_logits_44100.bin
venv/bin/python make_test_reference.py --sample-rate 22050 --out ../../src/test/beatthis/reference_logits_22050.bin
```

`export_onnx.py` also verifies the export against PyTorch and measures the
CPU speed. If the analysis results change, increment
`kNeuralAnalyzerVersion` in `src/analyzer/analyzerbars.cpp` to re-analyze the
tracks.
