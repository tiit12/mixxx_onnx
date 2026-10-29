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

    /// Moves the end of the intro and the start of the outro to the analyzed
    /// sections and creates hotcues for the sections, if enabled. Cues set
    /// by the user are not changed, and hotcues are only created for tracks
    /// without hotcues. If there are more sections than `maxHotcues`, the
    /// most useful ones are chosen (first drop, first breakdown, last drop,
    /// outro, buildups), numbered in the order of the track.
    static void setupCuesFromSections(Track* pTrack,
            const mixxx::Beats& beats,
            bool introAndOutro,
            bool hotcues,
            int maxHotcues = kMaxSectionHotcues);

    /// The pads of most controllers
    static constexpr int kMaxSectionHotcues = 8;

    /// Removes the hotcues placed at the sections and places the cues again
    /// at the current sections, according to the settings of the
    /// MixAssistant. Also replaces the end of the intro and the start of the
    /// outro.
    static void resetCuesFromSections(Track* pTrack);

  private:
    QString modelPath() const;
    bool isNeuralDetectionAvailable() const;
    QString currentAnalyzerVersion() const;
    bool hasCurrentBarInfo(const mixxx::Beats& beats) const;

    UserSettingsPointer m_pConfig;
    std::unique_ptr<mixxx::DownbeatDetector> m_pDetector;
};
