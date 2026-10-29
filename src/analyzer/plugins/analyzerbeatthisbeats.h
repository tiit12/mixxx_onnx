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

    explicit AnalyzerBeatThisBeats(std::shared_ptr<NeuralAnalysis> pNeuralAnalysis);
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

  private:
    const std::shared_ptr<NeuralAnalysis> m_pNeuralAnalysis;
    mixxx::audio::SampleRate m_sampleRate;
    QVector<mixxx::audio::FramePos> m_resultBeats;
};

} // namespace mixxx
