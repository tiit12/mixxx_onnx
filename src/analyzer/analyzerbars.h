#pragma once

#include <memory>
#include <vector>

#include "analyzer/analyzer.h"
#include "preferences/usersettings.h"
#include "track/beats.h"

namespace mixxx {
class DownbeatDetector;
} // namespace mixxx

/// Detects the bar and phrase structure (downbeats and phrases) on top of
/// the beats of a track.
///
/// Must run after AnalyzerBeats and AnalyzerSilence, because it depends on
/// the beats and on the N60dBSound cue stored by those analyzers.
///
/// If Mixxx is built with ONNX Runtime and the model is installed, the
/// downbeats are detected by a neural network. Otherwise the first beat
/// of the audible signal is assumed to be a downbeat.
class AnalyzerBars : public Analyzer {
  public:
    explicit AnalyzerBars(UserSettingsPointer pConfig);
    ~AnalyzerBars() override;

    bool initialize(const AnalyzerTrack& track,
            mixxx::audio::SampleRate sampleRate,
            mixxx::audio::ChannelCount channelCount,
            SINT frameLength) override;
    bool processSamples(const CSAMPLE* pIn, SINT count) override;
    void storeResults(TrackPointer pTrack) override;
    void cleanup() override;

    /// Assumes that the first beat at the beginning of the audible signal
    /// is a downbeat that starts a phrase. Returns nullptr if the track
    /// has no beats.
    static mixxx::BarInfoPointer guessBarInfoFromFirstSound(const Track& track);

    /// Chooses the downbeats among the beats by summing up the downbeat
    /// probabilities of each position in the bar. The phrases start at the
    /// first downbeat at or after `firstSound`.
    static mixxx::BarInfoPointer barInfoFromDownbeatLogits(
            const mixxx::Beats& beats,
            const std::vector<float>& downbeatLogits,
            double framesPerSecond,
            mixxx::audio::FramePos firstSound);

  private:
    QString modelPath() const;
    bool isNeuralDetectionAvailable() const;
    bool hasCurrentBarInfo(const mixxx::Beats& beats) const;

    UserSettingsPointer m_pConfig;
    std::unique_ptr<mixxx::DownbeatDetector> m_pDetector;
};
