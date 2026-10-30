#pragma once

#include <QObject>
#include <memory>
#include <vector>

#include "analyzer/plugins/analyzerplugin.h"

class NeuralAnalysis;

namespace mixxx {

/// Beat detection with the "Beat This!" neural network
/// (https://github.com/CPJKU/beat_this), which also detects the downbeats
/// for AnalyzerBars. The samples are processed by NeuralAnalysis.
class AnalyzerBeatThisBeats : public AnalyzerBeatsPlugin {
  public:
    static AnalyzerPluginInfo pluginInfo() {
        return AnalyzerPluginInfo(
                QStringLiteral("beat-this-small0"),
                QObject::tr("Johannes Kepler University Linz"),
                QObject::tr("Beat This! Neural Network"),
                true);
    }

    /// With `fixedTempo`, the beats are a grid of constant tempo
    AnalyzerBeatThisBeats(std::shared_ptr<NeuralAnalysis> pNeuralAnalysis, bool fixedTempo);
    ~AnalyzerBeatThisBeats() override;

    AnalyzerPluginInfo info() const override {
        return pluginInfo();
    }

    bool initialize(mixxx::audio::SampleRate sampleRate) override;
    bool processSamples(const CSAMPLE* pIn, SINT iLen) override;
    bool finalize() override;

    bool supportsBeatTracking() const override {
        return true;
    }

    QVector<mixxx::audio::FramePos> getBeats() const override {
        return m_resultBeats;
    }

    /// Picks the beats from the beat logits (log-odds per frame) like the
    /// "minimal" postprocessing of Beat This!, with sub-frame precision.
    static QVector<mixxx::audio::FramePos> beatsFromLogits(
            const std::vector<float>& beatLogits,
            double framesPerSecond,
            mixxx::audio::SampleRate sampleRate);

    /// Fits a grid of constant tempo to the beats, ignoring missing beats
    /// (e.g. in breaks or half-time parts) and outliers. The tempo detection
    /// of BeatUtils is made for the beat tracker of Queen Mary, which places
    /// beats everywhere, and mistakes long breaks between the beats of the
    /// neural network for a very slow tempo. Returns the beats of the grid
    /// from the first to the last beat, or nothing if most beats are not on
    /// a grid.
    static QVector<mixxx::audio::FramePos> constantTempoBeats(
            const QVector<mixxx::audio::FramePos>& beats,
            mixxx::audio::SampleRate sampleRate);

  private:
    const std::shared_ptr<NeuralAnalysis> m_pNeuralAnalysis;
    const bool m_fixedTempo;
    mixxx::audio::SampleRate m_sampleRate;
    QVector<mixxx::audio::FramePos> m_resultBeats;
};

} // namespace mixxx
