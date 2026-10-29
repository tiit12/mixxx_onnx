#include "analyzer/analyzerbars.h"

#include "analyzer/analyzertrack.h"
#include "track/track.h"
#include "util/logger.h"

namespace {

mixxx::Logger kLogger("AnalyzerBars");

// A beat that lies slightly before the first sound is still considered to
// be the first beat, because onsets are detected with some latency.
constexpr double kFirstBeatToleranceBeats = 0.25;

} // namespace

AnalyzerBars::AnalyzerBars(UserSettingsPointer pConfig)
        : m_pConfig(pConfig) {
}

bool AnalyzerBars::initialize(const AnalyzerTrack& track,
        mixxx::audio::SampleRate sampleRate,
        mixxx::audio::ChannelCount channelCount,
        SINT frameLength) {
    Q_UNUSED(sampleRate);
    Q_UNUSED(channelCount);
    Q_UNUSED(frameLength);
    // The beats might not exist yet, because they are detected in the
    // same analysis pass. Only skip tracks with a known bar structure.
    const mixxx::BeatsPointer pBeats = track.getTrack()->getBeats();
    return !pBeats || !pBeats->getBarInfo();
}

bool AnalyzerBars::processSamples(const CSAMPLE* pIn, SINT count) {
    Q_UNUSED(pIn);
    Q_UNUSED(count);
    return true;
}

void AnalyzerBars::storeResults(TrackPointer pTrack) {
    const mixxx::BeatsPointer pBeats = pTrack->getBeats();
    if (!pBeats || pBeats->getBarInfo()) {
        return;
    }
    mixxx::BarInfoPointer pBarInfo = guessBarInfoFromFirstSound(*pTrack);
    if (!pBarInfo) {
        return;
    }
    kLogger.debug() << "Guessed bar anchor at beat" << pBarInfo->anchorBeat;
    pTrack->trySetBarInfo(std::move(pBarInfo));
}

void AnalyzerBars::cleanup() {
}

// static
mixxx::BarInfoPointer AnalyzerBars::guessBarInfoFromFirstSound(const Track& track) {
    const mixxx::BeatsPointer pBeats = track.getBeats();
    if (!pBeats) {
        return nullptr;
    }
    mixxx::audio::FramePos firstSound = mixxx::audio::kStartFramePos;
    const CuePointer pN60dBSound = track.findCueByType(mixxx::CueType::N60dBSound);
    if (pN60dBSound && pN60dBSound->getPosition().isValid()) {
        firstSound = pN60dBSound->getPosition();
    }

    auto it = pBeats->iteratorFrom(firstSound);
    if (it == pBeats->cend() || it == pBeats->cbegin()) {
        return nullptr;
    }
    const auto prev = std::prev(it);
    if (firstSound - *prev < kFirstBeatToleranceBeats * (*it - *prev)) {
        it = prev;
    }

    auto pBarInfo = std::make_shared<mixxx::BarInfo>();
    pBarInfo->anchorBeat = pBeats->beatIndex(it);
    return pBarInfo;
}
