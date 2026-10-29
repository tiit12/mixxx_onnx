#include "analyzer/analyzerbars.h"

#include <QFileInfo>
#include <array>
#include <cmath>
#include <limits>
#include <optional>

#include "analyzer/analyzertrack.h"
#include "track/track.h"
#include "util/logger.h"
#ifdef __ONNXRUNTIME__
#include "analyzer/plugins/downbeatdetector.h"
#endif

namespace {

mixxx::Logger kLogger("AnalyzerBars");

// A beat that lies slightly before the first sound is still considered to
// be the first beat, because onsets are detected with some latency.
constexpr double kFirstBeatToleranceBeats = 0.25;

// The downbeat activation is searched in this range around each beat
// (+/- 40 ms at 50 frames per second)
constexpr int kDownbeatSearchFrames = 2;

// Increment if the results of the analysis change, to re-analyze the tracks
const QString kFirstSoundAnalyzerVersion = QStringLiteral("first-sound-1");
const QString kNeuralAnalyzerVersion = QStringLiteral("beat-this-small0-1");
const QString kModelFile = QStringLiteral("models/beat_this_small.onnx");

mixxx::audio::FramePos firstSoundPosition(const Track& track) {
    const CuePointer pN60dBSound = track.findCueByType(mixxx::CueType::N60dBSound);
    if (pN60dBSound && pN60dBSound->getPosition().isValid()) {
        return pN60dBSound->getPosition();
    }
    return mixxx::audio::kStartFramePos;
}

/// Index of the first beat of the audible signal
std::optional<int> firstBeatIndex(
        const mixxx::Beats& beats, mixxx::audio::FramePos firstSound) {
    auto it = beats.iteratorFrom(firstSound);
    if (it == beats.cend() || it == beats.cbegin()) {
        return std::nullopt;
    }
    const auto prev = std::prev(it);
    if (firstSound - *prev < kFirstBeatToleranceBeats * (*it - *prev)) {
        it = prev;
    }
    return beats.beatIndex(it);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

} // namespace

AnalyzerBars::AnalyzerBars(UserSettingsPointer pConfig)
        : m_pConfig(pConfig) {
}

AnalyzerBars::~AnalyzerBars() = default;

QString AnalyzerBars::modelPath() const {
    return m_pConfig->getResourcePath() + kModelFile;
}

bool AnalyzerBars::isNeuralDetectionAvailable() const {
#ifdef __ONNXRUNTIME__
    return QFileInfo::exists(modelPath());
#else
    return false;
#endif
}

bool AnalyzerBars::hasCurrentBarInfo(const mixxx::Beats& beats) const {
    const mixxx::BarInfoPointer pBarInfo = beats.getBarInfo();
    if (!pBarInfo) {
        return false;
    }
    if (pBarInfo->source == mixxx::BarInfo::Source::User) {
        // Never overwrite manual adjustments
        return true;
    }
    return pBarInfo->analyzerVersion ==
            (isNeuralDetectionAvailable() ? kNeuralAnalyzerVersion
                                          : kFirstSoundAnalyzerVersion);
}

bool AnalyzerBars::initialize(const AnalyzerTrack& track,
        mixxx::audio::SampleRate sampleRate,
        mixxx::audio::ChannelCount channelCount,
        SINT frameLength) {
    Q_UNUSED(frameLength);
    // The beats might not exist yet, because they are detected in the
    // same analysis pass.
    const mixxx::BeatsPointer pBeats = track.getTrack()->getBeats();
    if (pBeats && hasCurrentBarInfo(*pBeats)) {
        return false;
    }
#ifdef __ONNXRUNTIME__
    if (isNeuralDetectionAvailable()) {
        m_pDetector = std::make_unique<mixxx::DownbeatDetector>();
        if (!m_pDetector->initialize(modelPath(), sampleRate, channelCount)) {
            m_pDetector.reset();
        }
    }
#else
    Q_UNUSED(sampleRate);
    Q_UNUSED(channelCount);
#endif
    return true;
}

bool AnalyzerBars::processSamples(const CSAMPLE* pIn, SINT count) {
#ifdef __ONNXRUNTIME__
    if (m_pDetector && !m_pDetector->processSamples(pIn, count)) {
        // Fall back to the first sound heuristic
        m_pDetector.reset();
    }
#else
    Q_UNUSED(pIn);
    Q_UNUSED(count);
#endif
    return true;
}

void AnalyzerBars::storeResults(TrackPointer pTrack) {
    const mixxx::BeatsPointer pBeats = pTrack->getBeats();
    if (!pBeats || hasCurrentBarInfo(*pBeats)) {
        return;
    }

    mixxx::BarInfoPointer pBarInfo;
#ifdef __ONNXRUNTIME__
    if (m_pDetector && m_pDetector->finalize()) {
        pBarInfo = barInfoFromDownbeatLogits(*pBeats,
                m_pDetector->downbeatLogits(),
                mixxx::DownbeatDetector::kFramesPerSecond,
                firstSoundPosition(*pTrack));
    }
#endif
    if (!pBarInfo) {
        pBarInfo = guessBarInfoFromFirstSound(*pTrack);
    }
    if (!pBarInfo) {
        return;
    }
    kLogger.debug() << "Bar anchor at beat" << pBarInfo->anchorBeat
                    << "by" << pBarInfo->analyzerVersion;
    pTrack->trySetBarInfo(std::move(pBarInfo));
}

void AnalyzerBars::cleanup() {
    m_pDetector.reset();
}

// static
mixxx::BarInfoPointer AnalyzerBars::guessBarInfoFromFirstSound(const Track& track) {
    const mixxx::BeatsPointer pBeats = track.getBeats();
    if (!pBeats) {
        return nullptr;
    }
    const std::optional<int> firstIndex = firstBeatIndex(*pBeats, firstSoundPosition(track));
    if (!firstIndex) {
        return nullptr;
    }
    auto pBarInfo = std::make_shared<mixxx::BarInfo>();
    pBarInfo->anchorBeat = *firstIndex;
    pBarInfo->analyzerVersion = kFirstSoundAnalyzerVersion;
    return pBarInfo;
}

// static
mixxx::BarInfoPointer AnalyzerBars::barInfoFromDownbeatLogits(
        const mixxx::Beats& beats,
        const std::vector<float>& downbeatLogits,
        double framesPerSecond,
        mixxx::audio::FramePos firstSound) {
    const std::optional<int> firstIndex = firstBeatIndex(beats, firstSound);
    if (!firstIndex || downbeatLogits.empty()) {
        return nullptr;
    }
    const double sampleRate = beats.getSampleRate().value();
    const auto numFrames = static_cast<int>(downbeatLogits.size());

    std::array<double, mixxx::kDefaultBeatsPerBar> scores{};
    int numBeats = 0;
    int index = *firstIndex;
    for (auto it = beats.cfirstmarker() + index; it != beats.cend(); ++it, ++index) {
        const auto center = static_cast<int>(
                std::round(it->value() / sampleRate * framesPerSecond));
        if (center >= numFrames) {
            break;
        }
        float maxLogit = -std::numeric_limits<float>::infinity();
        for (int frame = std::max(0, center - kDownbeatSearchFrames);
                frame <= std::min(numFrames - 1, center + kDownbeatSearchFrames);
                ++frame) {
            maxLogit = std::max(maxLogit, downbeatLogits[frame]);
        }
        scores[mixxx::BarInfo::positiveModulo(index, mixxx::kDefaultBeatsPerBar)] +=
                sigmoid(maxLogit);
        numBeats++;
    }
    if (numBeats < 2 * mixxx::kDefaultBeatsPerBar) {
        return nullptr;
    }

    const auto best = std::max_element(scores.cbegin(), scores.cend());
    const int phase = static_cast<int>(best - scores.cbegin());
    double secondBest = 0;
    for (int i = 0; i < mixxx::kDefaultBeatsPerBar; ++i) {
        if (i != phase) {
            secondBest = std::max(secondBest, scores[i]);
        }
    }
    kLogger.info() << "Downbeat scores" << scores[0] << scores[1] << scores[2]
                   << scores[3] << "confidence"
                   << (*best > 0 ? (*best - secondBest) / *best : 0);

    auto pBarInfo = std::make_shared<mixxx::BarInfo>();
    pBarInfo->anchorBeat = *firstIndex +
            mixxx::BarInfo::positiveModulo(
                    phase - *firstIndex, mixxx::kDefaultBeatsPerBar);
    pBarInfo->analyzerVersion = kNeuralAnalyzerVersion;
    return pBarInfo;
}
