#pragma once

#include <QString>
#include <array>
#include <memory>
#include <vector>

#include "analyzer/phraseanalyzer.h"
#include "audio/types.h"
#include "util/types.h"

class FFTReal;
class Resampler;

namespace Ort {
struct Session;
} // namespace Ort

namespace mixxx {

/// Frame-wise beat and downbeat detection with the "Beat This!" neural
/// network (https://github.com/CPJKU/beat_this) using ONNX Runtime.
///
/// The model (see tools/beatthis/export_onnx.py) expects the magnitude STFT
/// of a 22050 Hz mono signal and processes chunks of 30 seconds. The audio
/// is processed in a streaming fashion, so that only the logits are kept for
/// the whole track.
class DownbeatDetector {
  public:
    static constexpr int kSampleRate = 22050;
    static constexpr int kFftSize = 1024;
    static constexpr int kHopSize = 441;
    static constexpr int kFramesPerSecond = kSampleRate / kHopSize;
    static constexpr int kNumBins = kFftSize / 2 + 1;
    static constexpr int kChunkFrames = 1500;
    /// Predictions at the chunk borders are discarded.
    static constexpr int kBorderFrames = 6;
    /// Samples (at kSampleRate) per value of bassEnergy()
    static constexpr int kBassBlockSize = 32;
    static constexpr double kBassBlocksPerSecond =
            static_cast<double>(kSampleRate) / kBassBlockSize;

    DownbeatDetector();
    ~DownbeatDetector();

    /// Returns false if the model cannot be loaded.
    bool initialize(const QString& modelPath,
            audio::SampleRate sampleRate,
            audio::ChannelCount channelCount);
    /// Processes interleaved samples.
    bool processSamples(const CSAMPLE* pIn, SINT count);
    bool finalize();

    /// Logits (log-odds) per frame of `1 / kFramesPerSecond` seconds.
    /// Frame `i` is centered at `i / kFramesPerSecond` seconds.
    const std::vector<float>& beatLogits() const {
        return m_beatLogits;
    }
    const std::vector<float>& downbeatLogits() const {
        return m_downbeatLogits;
    }

    /// Band energies for the phrase analysis, one value per frame.
    const BandEnergies& bandEnergies() const {
        return m_bandEnergies;
    }

    /// Energy of the low-pass filtered signal (kick drum and bass) in blocks
    /// of kBassBlockSize samples. Block `i` starts at `i / kBassBlocksPerSecond`
    /// seconds. The neural network places the beats some milliseconds after
    /// the start of the kick drum, which can be found precisely in this
    /// signal.
    const std::vector<float>& bassEnergy() const {
        return m_bassEnergy;
    }

  private:
    void appendResampled(const double* pSamples, int count);
    void appendBassSample(double sample);
    double signalAt(SINT index) const;
    void computeFrames();
    bool runReadyChunks();
    bool runChunk(int start, int numFrames);

    std::shared_ptr<Ort::Session> m_pSession;
    std::unique_ptr<Resampler> m_pResampler;
    std::unique_ptr<FFTReal> m_pFft;
    std::vector<double> m_window;
    std::vector<double> m_fftIn;
    std::vector<double> m_fftOut;
    std::vector<double> m_monoBuffer;
    std::vector<double> m_resampledBuffer;
    std::vector<float> m_chunk;
    /// Index of the BandEnergies::spectrum band of each FFT bin, or -1
    std::vector<int> m_spectrumBandOfBin;

    audio::SampleRate m_inputSampleRate;
    audio::ChannelCount m_channelCount;
    SINT m_inputFrames;
    int m_latencyToDrop;

    /// Resampled mono signal, `m_signal[0]` is the sample with the index
    /// `m_signalOffset`.
    std::vector<float> m_signal;
    SINT m_signalOffset;
    SINT m_signalLength;
    /// Total number of resampled samples, only known after finalize().
    SINT m_totalSamples;

    /// Magnitude frames (kNumBins values each), `m_frames[0]` is the frame
    /// with the index `m_framesOffset`.
    std::vector<float> m_frames;
    int m_framesOffset;
    int m_numFrames;
    int m_numChunksProcessed;
    int m_logitsWrittenUntil;

    std::vector<float> m_beatLogits;
    std::vector<float> m_downbeatLogits;
    BandEnergies m_bandEnergies;

    /// Biquad low-pass filter for bassEnergy(): b0, b1, b2, a1, a2
    std::array<double, 5> m_bassFilter;
    std::array<double, 4> m_bassFilterState;
    double m_bassBlockEnergy;
    int m_bassBlockSamples;
    std::vector<float> m_bassEnergy;
};

} // namespace mixxx
