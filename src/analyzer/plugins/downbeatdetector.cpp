#include <dsp/rateconversion/Resampler.h>
#include <dsp/transforms/FFT.h>
#include <onnxruntime_cxx_api.h>

// Class header comes after library includes here since our preprocessor
// definitions interfere with qm-dsp's headers.
#include "analyzer/plugins/downbeatdetector.h"

#include <QMutex>
#include <algorithm>
#include <array>
#include <cmath>

#include "util/logger.h"

namespace mixxx {

namespace {

const Logger kLogger("DownbeatDetector");

// The analysis runs in multiple analyzer threads in parallel
constexpr int kIntraOpThreads = 2;
constexpr int kChunkStep = DownbeatDetector::kChunkFrames - 2 * DownbeatDetector::kBorderFrames;
// Samples in front of the signal that are needed for the STFT
constexpr SINT kSignalMargin = DownbeatDetector::kFftSize;

// Upper limits of the low and mid band of BandEnergies
constexpr double kLowBandMaxHz = 150;
constexpr double kMidBandMaxHz = 2000;
constexpr double kHzPerBin = static_cast<double>(DownbeatDetector::kSampleRate) /
        DownbeatDetector::kFftSize;
constexpr int kLowBandEndBin = static_cast<int>(kLowBandMaxHz / kHzPerBin) + 1;
constexpr int kMidBandEndBin = static_cast<int>(kMidBandMaxHz / kHzPerBin) + 1;

// Cutoff of the low-pass filter for DownbeatDetector::bassEnergy(), like the
// low band of the waveform
constexpr double kBassCutoffHz = 600;

Ort::Env& ortEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mixxx");
    return env;
}

/// The session is shared by all analyzer threads, Ort::Session::Run() is
/// thread-safe.
std::shared_ptr<Ort::Session> sharedSession(const QString& modelPath) {
    static QMutex mutex;
    static std::shared_ptr<Ort::Session> pSession;
    static QString loadedModelPath;

    QMutexLocker locker(&mutex);
    if (pSession && loadedModelPath == modelPath) {
        return pSession;
    }
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(kIntraOpThreads);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // Don't keep the memory of the largest chunk between the analyses
    options.DisableCpuMemArena();
#ifdef _WIN32
    const std::wstring path = modelPath.toStdWString();
#else
    const std::string path = modelPath.toStdString();
#endif
    pSession = std::make_shared<Ort::Session>(ortEnv(), path.c_str(), options);
    loadedModelPath = modelPath;
    kLogger.info() << "Loaded model" << modelPath;
    return pSession;
}

/// Same as `split_piece()` in beat_this/inference.py with avoid_short_end=True
std::vector<int> chunkStarts(int numFrames) {
    std::vector<int> starts;
    for (int start = -DownbeatDetector::kBorderFrames;
            start < numFrames - DownbeatDetector::kBorderFrames;
            start += kChunkStep) {
        starts.push_back(start);
    }
    if (numFrames > kChunkStep) {
        starts.back() = numFrames -
                (DownbeatDetector::kChunkFrames - DownbeatDetector::kBorderFrames);
    }
    return starts;
}

} // namespace

DownbeatDetector::DownbeatDetector()
        : m_inputFrames(0),
          m_latencyToDrop(0),
          m_signalOffset(0),
          m_signalLength(0),
          m_totalSamples(-1),
          m_framesOffset(0),
          m_numFrames(0),
          m_numChunksProcessed(0),
          m_logitsWrittenUntil(0),
          m_bassFilterState{},
          m_bassBlockEnergy(0),
          m_bassBlockSamples(0) {
    // Butterworth low-pass (Audio EQ Cookbook)
    const double w0 = 2 * M_PI * kBassCutoffHz / kSampleRate;
    const double alpha = std::sin(w0) / std::sqrt(2.0);
    const double a0 = 1 + alpha;
    const double b1 = (1 - std::cos(w0)) / a0;
    m_bassFilter = {b1 / 2, b1, b1 / 2, -2 * std::cos(w0) / a0, (1 - alpha) / a0};
}

DownbeatDetector::~DownbeatDetector() = default;

bool DownbeatDetector::initialize(const QString& modelPath,
        audio::SampleRate sampleRate,
        audio::ChannelCount channelCount) {
    try {
        m_pSession = sharedSession(modelPath);
    } catch (const Ort::Exception& e) {
        kLogger.warning() << "Failed to load model" << modelPath << ":" << e.what();
        return false;
    }
    m_inputSampleRate = sampleRate;
    m_channelCount = channelCount;
    if (static_cast<int>(sampleRate.value()) != kSampleRate) {
        m_pResampler = std::make_unique<Resampler>(
                static_cast<int>(sampleRate.value()), kSampleRate);
        m_latencyToDrop = m_pResampler->getLatency();
    }
    m_pFft = std::make_unique<FFTReal>(kFftSize);
    m_window.resize(kFftSize);
    for (int i = 0; i < kFftSize; ++i) {
        // Periodic Hann window like torch.hann_window()
        m_window[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / kFftSize);
    }
    m_fftIn.resize(kFftSize);
    m_fftOut.resize(kFftSize);
    m_bandEnergies.framesPerSecond = kFramesPerSecond;
    m_spectrumBandOfBin.resize(kNumBins);
    for (int bin = 0; bin < kNumBins; ++bin) {
        const double hz = bin * kHzPerBin;
        if (hz < BandEnergies::kSpectrumMinHz || hz >= BandEnergies::kSpectrumMaxHz) {
            m_spectrumBandOfBin[bin] = -1;
            continue;
        }
        // Logarithmically spaced bands
        m_spectrumBandOfBin[bin] = std::min(BandEnergies::kNumSpectrumBands - 1,
                static_cast<int>(BandEnergies::kNumSpectrumBands *
                        std::log(hz / BandEnergies::kSpectrumMinHz) /
                        std::log(BandEnergies::kSpectrumMaxHz /
                                BandEnergies::kSpectrumMinHz)));
    }
    m_chunk.resize(static_cast<std::size_t>(kChunkFrames) * kNumBins);
    return true;
}

bool DownbeatDetector::processSamples(const CSAMPLE* pIn, SINT count) {
    const int channels = m_channelCount.value();
    const auto frames = static_cast<int>(count / channels);
    m_monoBuffer.resize(frames);
    for (int i = 0; i < frames; ++i) {
        double sum = 0;
        for (int c = 0; c < channels; ++c) {
            sum += pIn[i * channels + c];
        }
        m_monoBuffer[i] = sum / channels;
    }
    m_inputFrames += frames;
    if (!m_pResampler) {
        appendResampled(m_monoBuffer.data(), frames);
        computeFrames();
        return runReadyChunks();
    }
    m_resampledBuffer.resize(
            static_cast<std::size_t>(
                    std::ceil(frames * double(kSampleRate) / m_inputSampleRate.value())) +
            16);
    const int resampled = m_pResampler->process(
            m_monoBuffer.data(), m_resampledBuffer.data(), frames);
    appendResampled(m_resampledBuffer.data(), resampled);
    computeFrames();
    return runReadyChunks();
}

void DownbeatDetector::appendResampled(const double* pSamples, int count) {
    // Compensate the latency of the resampler filter
    const int dropped = std::min(count, m_latencyToDrop);
    m_latencyToDrop -= dropped;
    for (int i = dropped; i < count; ++i) {
        m_signal.push_back(static_cast<float>(pSamples[i]));
        appendBassSample(pSamples[i]);
    }
    m_signalLength += count - dropped;
}

void DownbeatDetector::appendBassSample(double sample) {
    auto& [x1, x2, y1, y2] = m_bassFilterState;
    const auto& [b0, b1, b2, a1, a2] = m_bassFilter;
    const double y = b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1;
    x1 = sample;
    y2 = y1;
    y1 = y;
    m_bassBlockEnergy += y * y;
    if (++m_bassBlockSamples == kBassBlockSize) {
        m_bassEnergy.push_back(static_cast<float>(m_bassBlockEnergy));
        m_bassBlockEnergy = 0;
        m_bassBlockSamples = 0;
    }
}

double DownbeatDetector::signalAt(SINT index) const {
    // Reflect padding like torch.stft(center=True, pad_mode="reflect")
    if (index < 0) {
        index = -index;
    }
    if (m_totalSamples >= 0 && index >= m_totalSamples) {
        index = 2 * (m_totalSamples - 1) - index;
    }
    DEBUG_ASSERT(index >= m_signalOffset && index < m_signalLength);
    return m_signal[index - m_signalOffset];
}

void DownbeatDetector::computeFrames() {
    const int totalFrames = m_totalSamples >= 0
            ? static_cast<int>(1 + m_totalSamples / kHopSize)
            : -1;
    while (true) {
        const SINT center = static_cast<SINT>(m_numFrames) * kHopSize;
        if (totalFrames >= 0 ? m_numFrames >= totalFrames
                             : center + kFftSize / 2 > m_signalLength) {
            break;
        }
        for (int i = 0; i < kFftSize; ++i) {
            m_fftIn[i] = signalAt(center - kFftSize / 2 + i) * m_window[i];
        }
        m_pFft->forwardMagnitude(m_fftIn.data(), m_fftOut.data());
        // torchaudio Spectrogram(normalized="frame_length")
        const double norm = 1.0 / std::sqrt(static_cast<double>(kFftSize));
        std::array<double, 3> bandPower{};
        std::array<float, BandEnergies::kNumSpectrumBands> spectrum{};
        for (int bin = 0; bin < kNumBins; ++bin) {
            const double magnitude = m_fftOut[bin] * norm;
            m_frames.push_back(static_cast<float>(magnitude));
            // Skip the DC bin
            if (bin > 0) {
                const double power = magnitude * magnitude;
                const int band = bin < kLowBandEndBin ? 0 : (bin < kMidBandEndBin ? 1 : 2);
                bandPower[band] += power;
                const int spectrumBand = m_spectrumBandOfBin[bin];
                if (spectrumBand >= 0) {
                    spectrum[spectrumBand] += static_cast<float>(power);
                }
            }
        }
        m_bandEnergies.low.push_back(static_cast<float>(bandPower[0]));
        m_bandEnergies.mid.push_back(static_cast<float>(bandPower[1]));
        m_bandEnergies.high.push_back(static_cast<float>(bandPower[2]));
        m_bandEnergies.spectrum.push_back(spectrum);
        m_numFrames++;
    }

    // Discard samples that are not needed anymore
    const SINT firstNeeded = static_cast<SINT>(m_numFrames) * kHopSize - kSignalMargin;
    if (firstNeeded - m_signalOffset > 16 * kFftSize) {
        m_signal.erase(m_signal.begin(), m_signal.begin() + (firstNeeded - m_signalOffset));
        m_signalOffset = firstNeeded;
    }
}

bool DownbeatDetector::runReadyChunks() {
    // A chunk can be processed before the end of the track is known, if all
    // of its frames are available. These chunks are never the last one,
    // which is aligned with the end of the track.
    while (true) {
        const int start = -kBorderFrames + m_numChunksProcessed * kChunkStep;
        if (start + kChunkFrames > m_numFrames) {
            return true;
        }
        if (!runChunk(start, m_numFrames)) {
            return false;
        }
        m_numChunksProcessed++;
        // The following chunks start at `start + kBorderFrames` or later
        const int firstNeeded = start + kBorderFrames;
        if (firstNeeded > m_framesOffset) {
            m_frames.erase(m_frames.begin(),
                    m_frames.begin() +
                            static_cast<std::ptrdiff_t>(firstNeeded - m_framesOffset) *
                                    kNumBins);
            m_framesOffset = firstNeeded;
        }
    }
}

bool DownbeatDetector::runChunk(int start, int numFrames) {
    std::fill(m_chunk.begin(), m_chunk.end(), 0.f);
    const int srcBegin = std::max(start, 0);
    const int srcEnd = std::min(start + kChunkFrames, numFrames);
    DEBUG_ASSERT(srcBegin >= m_framesOffset);
    std::copy(m_frames.begin() + static_cast<std::ptrdiff_t>(srcBegin - m_framesOffset) * kNumBins,
            m_frames.begin() + static_cast<std::ptrdiff_t>(srcEnd - m_framesOffset) * kNumBins,
            m_chunk.begin() + static_cast<std::ptrdiff_t>(srcBegin - start) * kNumBins);

    std::vector<Ort::Value> outputs;
    try {
        const auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 3> shape{1, kChunkFrames, kNumBins};
        Ort::Value input = Ort::Value::CreateTensor<float>(memoryInfo,
                m_chunk.data(),
                m_chunk.size(),
                shape.data(),
                shape.size());
        const char* inputNames[] = {"magnitude"};
        const char* outputNames[] = {"beat", "downbeat"};
        outputs = m_pSession->Run(Ort::RunOptions{nullptr},
                inputNames,
                &input,
                1,
                outputNames,
                2);
    } catch (const Ort::Exception& e) {
        kLogger.warning() << "Inference failed:" << e.what();
        return false;
    }
    const float* pBeat = outputs[0].GetTensorData<float>();
    const float* pDownbeat = outputs[1].GetTensorData<float>();

    // Earlier chunks take precedence in overlapping regions ("keep_first")
    const int dstBegin = std::max(start + kBorderFrames, m_logitsWrittenUntil);
    const int dstEnd = std::min(start + kChunkFrames - kBorderFrames, numFrames);
    if (dstEnd <= dstBegin) {
        return true;
    }
    m_beatLogits.resize(dstEnd);
    m_downbeatLogits.resize(dstEnd);
    for (int frame = dstBegin; frame < dstEnd; ++frame) {
        m_beatLogits[frame] = pBeat[frame - start];
        m_downbeatLogits[frame] = pDownbeat[frame - start];
    }
    m_logitsWrittenUntil = dstEnd;
    return true;
}

bool DownbeatDetector::finalize() {
    // Flush the resampler and cut the signal to the expected length
    const auto expectedLength = static_cast<SINT>(
            std::ceil(m_inputFrames * double(kSampleRate) / m_inputSampleRate.value()));
    std::vector<double> zeros(4096, 0.0);
    while (m_pResampler && m_signalLength < expectedLength) {
        m_resampledBuffer.resize(zeros.size() + 16);
        const int resampled = m_pResampler->process(
                zeros.data(), m_resampledBuffer.data(), static_cast<int>(zeros.size()));
        appendResampled(m_resampledBuffer.data(), resampled);
    }
    if (m_signalLength > expectedLength) {
        m_signal.resize(m_signal.size() - (m_signalLength - expectedLength));
        m_signalLength = expectedLength;
    }
    if (m_signalLength <= kFftSize / 2) {
        kLogger.warning() << "Track too short for downbeat detection";
        return false;
    }
    m_totalSamples = m_signalLength;
    computeFrames();

    const std::vector<int> starts = chunkStarts(m_numFrames);
    for (std::size_t i = m_numChunksProcessed; i < starts.size(); ++i) {
        if (!runChunk(starts[i], m_numFrames)) {
            return false;
        }
    }
    DEBUG_ASSERT(static_cast<int>(m_beatLogits.size()) == m_numFrames);
    m_frames.clear();
    m_signal.clear();
    return true;
}

} // namespace mixxx
