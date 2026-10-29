#include "analyzer/plugins/analyzerbeatthisbeats.h"

#include <algorithm>

#include "analyzer/neuralanalysis.h"
#include "analyzer/plugins/downbeatdetector.h"

namespace mixxx {

namespace {

/// A beat is a local maximum of the logits within +/- 3 frames (+/- 60 ms)
constexpr int kPeakNeighborFrames = 3;

} // namespace

AnalyzerBeatThisBeats::AnalyzerBeatThisBeats(std::shared_ptr<NeuralAnalysis> pNeuralAnalysis)
        : m_pNeuralAnalysis(std::move(pNeuralAnalysis)) {
}

AnalyzerBeatThisBeats::~AnalyzerBeatThisBeats() = default;

bool AnalyzerBeatThisBeats::initialize(mixxx::audio::SampleRate sampleRate) {
    m_sampleRate = sampleRate;
    m_resultBeats.clear();
    return m_pNeuralAnalysis && m_pNeuralAnalysis->request();
}

bool AnalyzerBeatThisBeats::processSamples(const CSAMPLE* pIn, SINT iLen) {
    // The samples are processed by NeuralAnalysis
    Q_UNUSED(pIn);
    Q_UNUSED(iLen);
    return true;
}

bool AnalyzerBeatThisBeats::finalize() {
    const DownbeatDetector* pDetector = m_pNeuralAnalysis->results();
    if (!pDetector) {
        return false;
    }
    m_resultBeats = beatsFromLogits(
            pDetector->beatLogits(), DownbeatDetector::kFramesPerSecond, m_sampleRate);
    return !m_resultBeats.isEmpty();
}

// static
QVector<mixxx::audio::FramePos> AnalyzerBeatThisBeats::beatsFromLogits(
        const std::vector<float>& beatLogits,
        double framesPerSecond,
        mixxx::audio::SampleRate sampleRate) {
    const auto numFrames = static_cast<int>(beatLogits.size());
    // Peak positions in frames of the logits
    std::vector<double> peaks;
    for (int i = 0; i < numFrames; ++i) {
        const float value = beatLogits[i];
        if (value <= 0) {
            continue;
        }
        const int first = std::max(0, i - kPeakNeighborFrames);
        const int last = std::min(numFrames - 1, i + kPeakNeighborFrames);
        if (*std::max_element(beatLogits.cbegin() + first, beatLogits.cbegin() + last + 1) >
                value) {
            continue;
        }
        double position = i;
        if (i > 0 && i + 1 < numFrames) {
            // Parabolic interpolation of the maximum
            const double left = beatLogits[i - 1];
            const double right = beatLogits[i + 1];
            const double curvature = left - 2 * value + right;
            if (curvature < 0) {
                position += std::clamp(0.5 * (left - right) / curvature, -0.5, 0.5);
            }
        }
        // Peaks in adjacent frames with the same value are one beat
        if (!peaks.empty() && position - peaks.back() <= 1.0) {
            peaks.back() = (peaks.back() + position) / 2;
            continue;
        }
        peaks.push_back(position);
    }

    QVector<mixxx::audio::FramePos> beats;
    beats.reserve(static_cast<int>(peaks.size()));
    for (const double peak : peaks) {
        beats.append(mixxx::audio::FramePos(peak / framesPerSecond * sampleRate.value()));
    }
    return beats;
}

} // namespace mixxx
