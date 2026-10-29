#include "analyzer/analyzerbars.h"

#include <QFileInfo>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>

#include "analyzer/analyzertrack.h"
#include "analyzer/phraseanalyzer.h"
#include "control/pollingcontrolproxy.h"
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
const QString kNeuralAnalyzerVersion = QStringLiteral("beat-this-small0-phrases-5");
const QString kModelFile = QStringLiteral("models/beat_this_small.onnx");

const QString kMixAssistantGroup = QStringLiteral("[MixAssistant]");

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

bool isSectionHotcue(const CuePointer& pCue) {
    if (pCue->getHotCue() == Cue::kNoHotCue) {
        return false;
    }
    const QString label = pCue->getLabel();
    for (const auto kind : {mixxx::PhraseKind::Unknown,
                 mixxx::PhraseKind::Intro,
                 mixxx::PhraseKind::Buildup,
                 mixxx::PhraseKind::Drop,
                 mixxx::PhraseKind::Breakdown,
                 mixxx::PhraseKind::Outro}) {
        if (label == mixxx::phraseKindName(kind)) {
            return true;
        }
    }
    return false;
}

/// Removes the hotcues placed by AnalyzerBars::setupCuesFromSections()
void removeSectionHotcues(Track* pTrack) {
    for (const CuePointer& pCue : pTrack->getCuePoints()) {
        if (isSectionHotcue(pCue)) {
            pTrack->removeCue(pCue);
        }
    }
}

bool isMixAssistantEnabled(const QString& item) {
    // Missing in tests
    return PollingControlProxy(kMixAssistantGroup, item, ControlFlag::AllowMissingOrInvalid)
            .toBool();
}

int maxSectionHotcues() {
    const auto value = static_cast<int>(PollingControlProxy(kMixAssistantGroup,
            QStringLiteral("section_hotcues"),
            ControlFlag::AllowMissingOrInvalid)
                    .get());
    return value > 0 ? std::min(value, AnalyzerBars::kMaxSectionHotcues)
                     : AnalyzerBars::kMaxSectionHotcues;
}

/// Indices of the sections that get a hotcue, in the order of the track
std::vector<std::size_t> chooseSectionsForHotcues(
        const std::vector<mixxx::Phrase>& sections, int maxHotcues) {
    std::vector<std::size_t> candidates;
    for (std::size_t i = 0; i < sections.size(); ++i) {
        // The start of the intro is the main cue
        if (sections[i].kind != mixxx::PhraseKind::Intro) {
            candidates.push_back(i);
        }
    }
    if (static_cast<int>(candidates.size()) <= maxHotcues) {
        return candidates;
    }

    std::vector<std::size_t> chosen;
    const auto choose = [&](std::size_t index) {
        if (static_cast<int>(chosen.size()) < maxHotcues &&
                std::find(chosen.cbegin(), chosen.cend(), index) == chosen.cend()) {
            chosen.push_back(index);
        }
    };
    const auto findFirst = [&](mixxx::PhraseKind kind) {
        return std::find_if(candidates.cbegin(), candidates.cend(), [&](std::size_t i) {
            return sections[i].kind == kind;
        });
    };
    const auto findLast = [&](mixxx::PhraseKind kind) {
        return std::find_if(candidates.crbegin(), candidates.crend(), [&](std::size_t i) {
            return sections[i].kind == kind;
        });
    };
    if (const auto it = findFirst(mixxx::PhraseKind::Drop); it != candidates.cend()) {
        choose(*it);
    }
    if (const auto it = findFirst(mixxx::PhraseKind::Breakdown); it != candidates.cend()) {
        choose(*it);
    }
    if (const auto it = findLast(mixxx::PhraseKind::Drop); it != candidates.crend()) {
        choose(*it);
    }
    if (const auto it = findFirst(mixxx::PhraseKind::Outro); it != candidates.cend()) {
        choose(*it);
    }
    for (const auto kind : {mixxx::PhraseKind::Buildup,
                 mixxx::PhraseKind::Drop,
                 mixxx::PhraseKind::Breakdown,
                 mixxx::PhraseKind::Unknown}) {
        for (const std::size_t i : candidates) {
            if (sections[i].kind == kind) {
                choose(i);
            }
        }
    }
    std::sort(chosen.begin(), chosen.end());
    return chosen;
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

QString AnalyzerBars::currentAnalyzerVersion() const {
    return isNeuralDetectionAvailable() ? kNeuralAnalyzerVersion : kFirstSoundAnalyzerVersion;
}

bool AnalyzerBars::hasCurrentBarInfo(const mixxx::Beats& beats) const {
    const mixxx::BarInfoPointer pBarInfo = beats.getBarInfo();
    // Manually adjusted downbeats are kept when re-analyzing, see storeResults()
    return pBarInfo && pBarInfo->analyzerVersion == currentAnalyzerVersion();
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

    const mixxx::BarInfoPointer pUserBarInfo =
            pBeats->getBarInfo() &&
                    pBeats->getBarInfo()->source == mixxx::BarInfo::Source::User
            ? pBeats->getBarInfo()
            : nullptr;
#ifdef __ONNXRUNTIME__
    const bool detected = m_pDetector && m_pDetector->finalize();
#else
    const bool detected = false;
#endif

    std::shared_ptr<mixxx::BarInfo> pBarInfo;
    if (pUserBarInfo) {
        // Keep the manually adjusted downbeats
        pBarInfo = std::make_shared<mixxx::BarInfo>(*pUserBarInfo);
        pBarInfo->phrases.clear();
        pBarInfo->analyzerVersion = detected ? kNeuralAnalyzerVersion : kFirstSoundAnalyzerVersion;
    } else {
        mixxx::BarInfoPointer pDetected;
#ifdef __ONNXRUNTIME__
        if (detected) {
            pDetected = barInfoFromDownbeatLogits(*pBeats,
                    m_pDetector->downbeatLogits(),
                    mixxx::DownbeatDetector::kFramesPerSecond,
                    firstSoundPosition(*pTrack));
        }
#endif
        if (!pDetected) {
            pDetected = guessBarInfoFromFirstSound(*pTrack);
        }
        if (!pDetected) {
            return;
        }
        pBarInfo = std::make_shared<mixxx::BarInfo>(*pDetected);
    }

#ifdef __ONNXRUNTIME__
    if (detected) {
        pBarInfo->phrases = mixxx::PhraseAnalyzer::analyze(
                *pBeats, *pBarInfo, m_pDetector->bandEnergies());
    }
#endif
    kLogger.debug() << "Bar anchor at beat" << pBarInfo->anchorBeat << "with"
                    << pBarInfo->phrases.size() << "sections by"
                    << pBarInfo->analyzerVersion;
    if (!pTrack->trySetBarInfo(std::move(pBarInfo))) {
        return;
    }

    const bool hotcues = isMixAssistantEnabled(QStringLiteral("auto_hotcues"));
    if (hotcues) {
        // Replace the hotcues of a previous analysis
        removeSectionHotcues(pTrack.get());
    }
    setupCuesFromSections(pTrack.get(),
            *pTrack->getBeats(),
            isMixAssistantEnabled(QStringLiteral("auto_intro_outro")),
            hotcues,
            maxSectionHotcues());
}

// static
void AnalyzerBars::resetCuesFromSections(Track* pTrack) {
    const mixxx::BeatsPointer pBeats = pTrack->getBeats();
    if (!pBeats || !pBeats->getBarInfo()) {
        return;
    }
    const bool introAndOutro = isMixAssistantEnabled(QStringLiteral("auto_intro_outro"));
    const bool hotcues = isMixAssistantEnabled(QStringLiteral("auto_hotcues"));
    if (introAndOutro) {
        const CuePointer pIntro = pTrack->findCueByType(mixxx::CueType::Intro);
        if (pIntro) {
            pIntro->setStartAndEndPosition(pIntro->getPosition(), mixxx::audio::kInvalidFramePos);
        }
        const CuePointer pOutro = pTrack->findCueByType(mixxx::CueType::Outro);
        if (pOutro) {
            pOutro->setStartAndEndPosition(mixxx::audio::kInvalidFramePos, pOutro->getEndPosition());
        }
    }
    if (hotcues) {
        removeSectionHotcues(pTrack);
    }
    setupCuesFromSections(pTrack, *pBeats, introAndOutro, hotcues, maxSectionHotcues());
}

// static
void AnalyzerBars::setupCuesFromSections(Track* pTrack,
        const mixxx::Beats& beats,
        bool introAndOutro,
        bool hotcues,
        int maxHotcues) {
    const mixxx::BarInfoPointer pBarInfo = beats.getBarInfo();
    if (!pBarInfo || pBarInfo->phrases.size() < 2) {
        return;
    }
    const auto& sections = pBarInfo->phrases;
    const auto sectionStart = [&](const mixxx::Phrase& section) {
        return *(beats.cfirstmarker() + section.startBeat);
    };

    if (introAndOutro) {
        // The intro start and the outro end are set by AnalyzerSilence
        if (sections.front().kind == mixxx::PhraseKind::Intro) {
            const CuePointer pIntro = pTrack->findCueByType(mixxx::CueType::Intro);
            const auto introEnd = sectionStart(sections[1]);
            if (pIntro && !pIntro->getEndPosition().isValid() &&
                    pIntro->getPosition() < introEnd) {
                pIntro->setStartAndEndPosition(pIntro->getPosition(), introEnd);
            }
        }
        if (sections.back().kind == mixxx::PhraseKind::Outro) {
            const CuePointer pOutro = pTrack->findCueByType(mixxx::CueType::Outro);
            const auto outroStart = sectionStart(sections.back());
            if (pOutro && !pOutro->getPosition().isValid() &&
                    (!pOutro->getEndPosition().isValid() ||
                            outroStart < pOutro->getEndPosition())) {
                pOutro->setStartAndEndPosition(outroStart, pOutro->getEndPosition());
            }
        }
    }

    if (hotcues) {
        const QList<CuePointer> cues = pTrack->getCuePoints();
        const bool hasHotcues = std::any_of(cues.cbegin(), cues.cend(), [](const CuePointer& pCue) {
            return pCue->getHotCue() != Cue::kNoHotCue;
        });
        if (hasHotcues) {
            return;
        }
        int hotcueIndex = 0;
        for (const std::size_t sectionIndex : chooseSectionsForHotcues(sections, maxHotcues)) {
            const mixxx::Phrase& section = sections[sectionIndex];
            const CuePointer pCue = pTrack->createAndAddCue(mixxx::CueType::HotCue,
                    hotcueIndex++,
                    sectionStart(section),
                    mixxx::audio::kInvalidFramePos,
                    mixxx::PhraseAnalyzer::sectionColor(section.kind));
            pCue->setLabel(mixxx::PhraseAnalyzer::sectionName(section.kind));
        }
    }
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
