#pragma once

#include "analyzer/analyzer.h"
#include "preferences/usersettings.h"

/// Detects the bar and phrase structure (downbeats and phrases) on top of
/// the beats of a track.
///
/// Must run after AnalyzerBeats and AnalyzerSilence, because it depends on
/// the beats and on the N60dBSound cue stored by those analyzers.
class AnalyzerBars : public Analyzer {
  public:
    explicit AnalyzerBars(UserSettingsPointer pConfig);
    ~AnalyzerBars() override = default;

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

  private:
    UserSettingsPointer m_pConfig;
};
