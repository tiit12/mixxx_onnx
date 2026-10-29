# Neural network models

## beat_this_small.onnx

Beat and downbeat tracker "Beat This!" (checkpoint `small0`) by the
Institute of Computational Perception, JKU Linz:
<https://github.com/CPJKU/beat_this>, MIT license (see `LICENSE.beat_this.txt`).

Exported with `tools/beatthis/export_onnx.py --checkpoint small0`. The ONNX
graph contains the mel filterbank and the log compression; the input is the
magnitude STFT of a 22050 Hz mono signal (see `src/analyzer/plugins/downbeatdetector.h`).

Used by `AnalyzerBars` if Mixxx is built with `-DONNXRUNTIME=ON`.
