#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "analyzer/analyzer.h"
#include "preferences/usersettings.h"
#include "track/beats.h"

class NeuralAnalysis;

/// Detects the bar and phrase structure (downbeats and phrases) on top of
/// the beats of a track.
///
/// Must run after AnalyzerBeats and AnalyzerSilence, because it depends on
/// the beats and on the N60dBSound cue stored by those analyzers.
///
/// If Mixxx is built with ONNX Runtime and the model is installed, the
/// downbeats are detected by a neural network (`pNeuralAnalysis`). Otherwise
/// the first beat of the audible signal is assumed to be a downbeat.
class AnalyzerBars : public Analyzer {
  public:
    explicit AnalyzerBars(UserSettingsPointer pConfig,
            std::shared_ptr<NeuralAnalysis> pNeuralAnalysis = nullptr);
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

    struct BeatPhase {
        /// Frames by which the beats need to be moved
        mixxx::audio::FrameDiff_t offset;
        /// Mean beat probability at the moved beats and at the original beats
        double activation;
        double originalActivation;
    };

    /// Finds the phase of a beat grid with constant tempo that matches the
    /// beat probabilities of the neural network best, e.g. if the beat
    /// detector placed the beats some milliseconds late or on the offbeats.
    /// Returns std::nullopt for beats without constant tempo.
    static std::optional<BeatPhase> findBeatPhase(const mixxx::Beats& beats,
            const std::vector<float>& beatLogits,
            double framesPerSecond);

    /// Whether the beats should be moved by the result of findBeatPhase()
    static bool shouldCorrectBeatPhase(const BeatPhase& phase,
            mixxx::audio::SampleRate sampleRate);

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
    bool isNeuralDetectionAvailable() const;
    QString currentAnalyzerVersion() const;
    bool hasCurrentBarInfo(const mixxx::Beats& beats) const;

    UserSettingsPointer m_pConfig;
    const std::shared_ptr<NeuralAnalysis> m_pNeuralAnalysis;
    /// To find out whether AnalyzerBeats has detected new beats
    mixxx::BeatsPointer m_pBeatsBeforeAnalysis;
};
