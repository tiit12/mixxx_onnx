#pragma once

#include <QHash>
#include <QList>
#include <memory>

#include "analyzer/analyzer.h"
#include "analyzer/plugins/analyzerplugin.h"
#include "preferences/beatdetectionsettings.h"
#include "preferences/usersettings.h"

class NeuralAnalysis;

class AnalyzerBeats : public Analyzer {
  public:
    /// `pNeuralAnalysis` is required for the Beat This! plugin, which is
    /// replaced by the default plugin otherwise.
    explicit AnalyzerBeats(
            UserSettingsPointer pConfig,
            bool enforceBpmDetection = false,
            std::shared_ptr<NeuralAnalysis> pNeuralAnalysis = nullptr);
    ~AnalyzerBeats() override = default;

    static QList<mixxx::AnalyzerPluginInfo> availablePlugins();
    static mixxx::AnalyzerPluginInfo defaultPlugin();

    bool initialize(const AnalyzerTrack& track,
            mixxx::audio::SampleRate sampleRate,
            mixxx::audio::ChannelCount channelCount,
            SINT frameLength) override;
    bool processSamples(const CSAMPLE* pIn, SINT count) override;
    void storeResults(TrackPointer tio) override;
    void cleanup() override;

  private:
    /// The configured plugin, or the default plugin if it is not available
    QString availablePluginId() const;
    bool shouldAnalyze(TrackPointer pTrack) const;
    static QHash<QString, QString> getExtraVersionInfo(
            const QString& pluginId, bool bPreferencesFastAnalysis);

    BeatDetectionSettings m_bpmSettings;
    const std::shared_ptr<NeuralAnalysis> m_pNeuralAnalysis;
    std::unique_ptr<mixxx::AnalyzerBeatsPlugin> m_pPlugin;
    const bool m_enforceBpmDetection;
    QString m_pluginId;
    bool m_bPreferencesReanalyzeOldBpm;
    bool m_bPreferencesReanalyzeImported;
    bool m_bPreferencesFixedTempo;
    bool m_bPreferencesFastAnalysis;

    mixxx::audio::SampleRate m_sampleRate;
    mixxx::audio::ChannelCount m_channelCount;
    SINT m_maxFramesToProcess;
    SINT m_currentFrame;
};
