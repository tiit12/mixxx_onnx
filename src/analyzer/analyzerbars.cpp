#include "analyzer/analyzerbars.h"

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
#include "analyzer/neuralanalysis.h"
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

// The phase of the beat grid is searched in steps of 1 ms
constexpr double kBeatPhaseStepSeconds = 0.001;
// Smaller phase errors are inaudible
constexpr double kMinBeatPhaseCorrectionSeconds = 0.005;
// The beat probabilities at the moved beats must be this high, i.e. the
// neural network must clearly hear the beats
constexpr double kMinBeatPhaseActivation = 0.3;

// The kick drum is searched in this range around the beats. The range ends
// shortly after the beats, because the bass that follows the kick drum can
// be louder than the kick drum.
constexpr double kBeatOnsetSearchBeforeSeconds = 0.08;
constexpr double kBeatOnsetSearchAfterSeconds = 0.025;
// The minimum before the kick drum is searched from here
constexpr double kBeatOnsetBaseBeforeSeconds = 0.1;
// Smoothing of the mean bass energy, against the ripple of low frequencies
constexpr double kBeatOnsetSmoothingSeconds = 0.007;
// The onset is where the bass amplitude has risen by this part from the
// minimum before the kick drum to its maximum
constexpr double kBeatOnsetLevel = 0.15;
// The maximum of the bass amplitude must be this much higher than the
// minimum
constexpr double kMinBeatOnsetContrast = 1.3;
constexpr int kMinBeatOnsetBeats = 16;
constexpr double kMinBeatOnsetCorrectionSeconds = 0.002;

// Increment if the results of the analysis change, to re-analyze the tracks
const QString kFirstSoundAnalyzerVersion = QStringLiteral("first-sound-1");
const QString kNeuralAnalyzerVersion = QStringLiteral("beat-this-small0-phrases-6");

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

AnalyzerBars::AnalyzerBars(
        UserSettingsPointer pConfig, std::shared_ptr<NeuralAnalysis> pNeuralAnalysis)
        : m_pConfig(pConfig),
          m_pNeuralAnalysis(std::move(pNeuralAnalysis)) {
}

AnalyzerBars::~AnalyzerBars() = default;

bool AnalyzerBars::isNeuralDetectionAvailable() const {
#ifdef __ONNXRUNTIME__
    return m_pNeuralAnalysis && m_pNeuralAnalysis->isAvailable();
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

// static
std::optional<AnalyzerBars::BeatPhase> AnalyzerBars::findBeatPhase(
        const mixxx::Beats& beats,
        const std::vector<float>& beatLogits,
        double framesPerSecond) {
    if (!beats.hasConstantTempo() || beatLogits.empty() || framesPerSecond <= 0) {
        return std::nullopt;
    }
    const double sampleRate = beats.getSampleRate().value();
    std::vector<double> activation(beatLogits.size());
    std::transform(beatLogits.cbegin(), beatLogits.cend(), activation.begin(), sigmoid);
    const auto lastFrame = static_cast<double>(activation.size() - 1);
    const auto activationAt = [&](double frame) {
        if (frame < 0 || frame > lastFrame) {
            return 0.0;
        }
        const auto index = static_cast<std::size_t>(frame);
        const double fraction = frame - index;
        if (fraction == 0) {
            return activation[index];
        }
        return activation[index] * (1 - fraction) + activation[index + 1] * fraction;
    };

    // The positions of the beats in frames of the beat logits
    std::vector<double> beatFrames;
    auto it = beats.iteratorFrom(mixxx::audio::kStartFramePos);
    if (it == beats.cend()) {
        return std::nullopt;
    }
    const double beatLength = it.beatLengthFrames() * framesPerSecond / sampleRate;
    for (; it != beats.cend(); ++it) {
        const double frame = it->value() * framesPerSecond / sampleRate;
        if (frame > lastFrame) {
            break;
        }
        beatFrames.push_back(frame);
    }
    if (beatFrames.empty() || beatLength <= 0) {
        return std::nullopt;
    }
    const auto meanActivation = [&](double offset) {
        double sum = 0;
        for (const double frame : beatFrames) {
            sum += activationAt(frame + offset);
        }
        return sum / beatFrames.size();
    };

    // Search within half a beat in both directions, which also finds beat
    // grids on the offbeats
    const double step = kBeatPhaseStepSeconds * framesPerSecond;
    double bestOffset = 0;
    double bestActivation = meanActivation(0);
    const double originalActivation = bestActivation;
    for (double offset = -beatLength / 2; offset < beatLength / 2; offset += step) {
        const double value = meanActivation(offset);
        if (value > bestActivation) {
            bestActivation = value;
            bestOffset = offset;
        }
    }
    return BeatPhase{bestOffset * sampleRate / framesPerSecond,
            bestActivation,
            originalActivation};
}

// static
bool AnalyzerBars::shouldCorrectBeatPhase(
        const BeatPhase& phase, mixxx::audio::SampleRate sampleRate) {
    return std::abs(phase.offset) >= kMinBeatPhaseCorrectionSeconds * sampleRate.value() &&
            phase.activation >= kMinBeatPhaseActivation;
}

// static
std::optional<mixxx::audio::FrameDiff_t> AnalyzerBars::findBeatOnset(
        const mixxx::Beats& beats,
        const std::vector<float>& bassEnergy,
        double blocksPerSecond) {
    if (!beats.hasConstantTempo() || bassEnergy.empty() || blocksPerSecond <= 0) {
        return std::nullopt;
    }
    const double sampleRate = beats.getSampleRate().value();
    const auto numBlocks = static_cast<int>(bassEnergy.size());
    const auto toBlocks = [&](double seconds) {
        return static_cast<int>(std::round(seconds * blocksPerSecond));
    };
    const int halfWindow = toBlocks(kBeatOnsetSmoothingSeconds / 2);
    const int before = toBlocks(kBeatOnsetBaseBeforeSeconds);
    const int after = toBlocks(kBeatOnsetSearchAfterSeconds) + halfWindow;

    // Mean bass energy around the beats, index `before` is at the beats
    std::vector<double> profile(before + after + 1, 0.0);
    int numBeats = 0;
    for (auto it = beats.iteratorFrom(mixxx::audio::kStartFramePos); it != beats.cend(); ++it) {
        // Blocks are centered at `(i + 0.5) / blocksPerSecond`
        const auto block = static_cast<int>(
                std::round(it->value() / sampleRate * blocksPerSecond - 0.5));
        if (block - before < 0) {
            continue;
        }
        if (block + after >= numBlocks) {
            break;
        }
        for (std::size_t i = 0; i < profile.size(); ++i) {
            profile[i] += bassEnergy[block - before + i];
        }
        numBeats++;
    }
    if (numBeats < kMinBeatOnsetBeats) {
        return std::nullopt;
    }

    // Smoothed amplitude, like the waveform
    const auto size = static_cast<int>(profile.size());
    std::vector<double> amplitude(profile.size());
    for (int i = 0; i < size; ++i) {
        const int begin = std::max(0, i - halfWindow);
        const int end = std::min(size, i + halfWindow + 1);
        double sum = 0;
        for (int j = begin; j < end; ++j) {
            sum += profile[j];
        }
        amplitude[i] = std::sqrt(sum / (end - begin));
    }

    const auto searchBegin = amplitude.cbegin() + before - toBlocks(kBeatOnsetSearchBeforeSeconds);
    const auto searchEnd = amplitude.cbegin() + before + toBlocks(kBeatOnsetSearchAfterSeconds) + 1;
    const auto peak = static_cast<int>(
            std::max_element(searchBegin, searchEnd) - amplitude.cbegin());
    const double minimum = *std::min_element(amplitude.cbegin(), amplitude.cbegin() + peak + 1);
    if (amplitude[peak] < kMinBeatOnsetContrast * minimum || amplitude[peak] <= 0) {
        return std::nullopt;
    }
    const double level = minimum + kBeatOnsetLevel * (amplitude[peak] - minimum);
    int onset = peak;
    while (onset > 0 && amplitude[onset - 1] >= level) {
        onset--;
    }
    if (onset == 0) {
        // The bass rises before the search range
        return std::nullopt;
    }
    // Interpolate between the blocks
    const double position = onset - 1 +
            (level - amplitude[onset - 1]) / (amplitude[onset] - amplitude[onset - 1]);
    return (position - before) / blocksPerSecond * sampleRate;
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
    m_pBeatsBeforeAnalysis = pBeats;
    Q_UNUSED(sampleRate);
    Q_UNUSED(channelCount);
#ifdef __ONNXRUNTIME__
    if (isNeuralDetectionAvailable()) {
        m_pNeuralAnalysis->request();
    }
#endif
    return true;
}

bool AnalyzerBars::processSamples(const CSAMPLE* pIn, SINT count) {
    // The samples are processed by NeuralAnalysis
    Q_UNUSED(pIn);
    Q_UNUSED(count);
    return true;
}

void AnalyzerBars::storeResults(TrackPointer pTrack) {
    mixxx::BeatsPointer pBeats = pTrack->getBeats();
    if (!pBeats || hasCurrentBarInfo(*pBeats)) {
        return;
    }

#ifdef __ONNXRUNTIME__
    // Falls back to the first sound heuristic if the analysis has failed
    const mixxx::DownbeatDetector* pDetector =
            m_pNeuralAnalysis ? m_pNeuralAnalysis->results() : nullptr;
    const bool detected = pDetector != nullptr;
    // Only beats that have just been detected by AnalyzerBeats are corrected,
    // beats adjusted by the user are kept.
    if (detected && pBeats != m_pBeatsBeforeAnalysis) {
        const double sampleRate = pBeats->getSampleRate().value();
        const auto translate = [&](mixxx::audio::FrameDiff_t offset) {
            const auto translated = pBeats->tryTranslate(offset);
            if (translated && pTrack->trySetBeats(*translated)) {
                pBeats = *translated;
            }
        };
        const auto phase = findBeatPhase(*pBeats,
                pDetector->beatLogits(),
                mixxx::DownbeatDetector::kFramesPerSecond);
        if (phase) {
            const bool correct = shouldCorrectBeatPhase(*phase, pBeats->getSampleRate());
            kLogger.info() << "Beat phase offset" << phase->offset / sampleRate * 1000
                           << "ms, beat probability" << phase->originalActivation
                           << "->" << phase->activation
                           << (correct ? "corrected" : "not corrected");
            if (correct) {
                translate(phase->offset);
            }
        }
        // The neural network places the beats slightly after the start of
        // the kick drum
        const auto onset = findBeatOnset(*pBeats,
                pDetector->bassEnergy(),
                mixxx::DownbeatDetector::kBassBlocksPerSecond);
        if (onset) {
            kLogger.info() << "Beat onset offset" << *onset / sampleRate * 1000 << "ms";
            if (std::abs(*onset) >= kMinBeatOnsetCorrectionSeconds * sampleRate) {
                translate(*onset);
            }
        }
    }
#else
    const bool detected = false;
#endif

    const mixxx::BarInfoPointer pUserBarInfo =
            pBeats->getBarInfo() &&
                    pBeats->getBarInfo()->source == mixxx::BarInfo::Source::User
            ? pBeats->getBarInfo()
            : nullptr;

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
                    pDetector->downbeatLogits(),
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
                *pBeats, *pBarInfo, pDetector->bandEnergies());
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
    m_pBeatsBeforeAnalysis.reset();
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
