# Downbeat and phrase analysis (local fork)

Mixxx detects the downbeats of a track with the "Beat This!" neural network
(<https://github.com/CPJKU/beat_this>, MIT) and shows bars and phrases in the
waveform:

- thin lines: beats
- white lines: downbeats (first beat of a bar)
- orange lines: phrase starts (every 8 bars = 32 beats)

The neural network decides which beat of the Mixxx beat grid is the
downbeat. Without ONNX Runtime (or without the model file) the first beat
after the initial silence is used instead.

## Beat detection

*Preferences → Beat Detection → "Beat This! Neural Network"* detects the
beats (BPM and beat grid) with the same neural network instead of the Queen
Mary beat tracker. It finds the correct tempo of drum and bass (175 instead
of 87.5 BPM) and does not place the grid on the offbeats. The network runs
only once per track for the beats and the phrases, so it is not slower than
the Queen Mary beat tracker plus the phrase analysis.

Already analyzed tracks keep their beats. To analyze them again, enable
*Re-analyze beats when settings change* or use *Analyze → Reanalyze* in the
context menu of the library.

With the Queen Mary beat tracker, newly detected beat grids are moved to the
beats of the neural network if they are off by 5 ms or more, e.g. on the
offbeats. Beat grids adjusted by the user are not changed.

The neural network places the beats 15-30 ms after the start of the kick
drum. Therefore newly detected beat grids (of both beat detectors) are
moved to the start of the kick drum, which is found in the mean amplitude
of the low band (below 600 Hz, the red part of the waveform) around the
louder half of the beats, i.e. without the intro or breakdown without kick
drum. Tracks without a clear kick drum, e.g. with a loud bassline between
the kicks, keep the grid of the beat detector.

`BEAT_PHASE_DIR=~/Music build/mixxx-test --gtest_also_run_disabled_tests
--gtest_filter='*BeatPhaseFiles'` prints the beat grids of both beat
detectors and the corrections for the audio files in a directory, with
`BEAT_PHASE_SECTIONS=1` also the analyzed sections.

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
the analyzer.

## Mix Assistant

*Preferences → Mix Assistant* switches these features on and off (all on by
default, persistent controls in the group `[MixAssistant]`):

| Option | Control | Effect |
|---|---|---|
| Set the intro end and the outro start from the sections | `auto_intro_outro` | Auto DJ ("Full Intro + Outro") mixes phrase by phrase |
| Create hotcues for the sections | `auto_hotcues` | Tracks without hotcues of the user get hotcues at the sections, labeled and colored like the sections |
| Number of hotcues | `section_hotcues` | 1-8, e.g. the hotcue pads of the controller. With more sections, the most useful ones are chosen: first drop, first breakdown, last drop, outro, buildups |
| Start decks on the next phrase | `play_on_phrase` | Play waits for the next phrase of the playing deck (green countdown); press play again to start immediately |
| Warn if the bass of two decks clashes | `bass_clash_warning` | Red bar at the top of the waveforms while two decks play full bass |

The cues are set during the analysis; cues and hotcues set by the user are
never changed. The section cues move along when the beatgrid is adjusted or
the downbeats are shifted. `[ChannelN],section_cues_reset` (keyboard
Ctrl+Shift+X / Ctrl+Shift+H) places them again at the current sections.
Hotcues, cues and Auto DJ always start immediately.

The section hotcues are normal hotcues, i.e. the hotcue pads of controllers
trigger them. The controls of this page are listed in the MIDI Learning
Wizard under *Bars / Phrases*.

Additional controls: `[ChannelN],play_on_phrase_armed`,
`[ChannelN],play_on_phrase_countdown`, `[ChannelN],bass_clash`,
`[ChannelN],section_bass` (bass level of the current section, 0-1).

## Waveform preferences

- *Play marker hints → Beats until next phrase* (`[Waveform],UntilPhraseShowBeats`,
  default on): orange countdown next to the play marker, 32 … 1 beats until
  the next phrase start. Requires an OpenGL waveform type.
- *Visual gain → Equalizer changes the waveform* (`[Waveform],EqAffectsWaveform`,
  default on): if disabled, the EQ knobs and kill switches only change the
  sound and the waveform always shows the original track. The colors can be changed in the skin with `BarColor` and
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
CPU speed.

## Tuning the phrase analysis

`phrase_features.py track.mp3` prints the band energies per bar.
`dump_phrase_features.py` saves the features of many tracks, which the
disabled test `PhraseAnalyzerTest.DISABLED_FeatureFiles` analyzes with the
C++ `PhraseAnalyzer`, so the thresholds can be tuned without Mixxx:

```sh
venv/bin/python dump_phrase_features.py --out /tmp/features ~/Music/*.mp3
PHRASE_FEATURES_DIR=/tmp/features build/mixxx-test \
    --gtest_also_run_disabled_tests --gtest_filter='*FeatureFiles'
``` If the analysis results change, increment
`kNeuralAnalyzerVersion` in `src/analyzer/analyzerbars.cpp` to re-analyze the
tracks.
