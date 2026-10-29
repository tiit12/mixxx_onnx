#pragma once

#include <QString>
#include <memory>
#include <optional>

#include "audio/types.h"
#include "preferences/usersettings.h"
#include "util/types.h"

namespace mixxx {
class DownbeatDetector;
} // namespace mixxx

/// Runs the "Beat This!" neural network (see DownbeatDetector) once per track
/// for all analyzers that need its results, i.e. AnalyzerBeats with the
/// Beat This! plugin and AnalyzerBars. The analyzers request it in
/// initialize(), and the AnalyzerThread feeds it with the samples of the
/// track.
///
/// Only available if Mixxx is built with ONNX Runtime.
class NeuralAnalysis {
  public:
    explicit NeuralAnalysis(const UserSettingsPointer& pConfig);
    ~NeuralAnalysis();

    static QString modelPath(const UserSettingsPointer& pConfig);

    /// Whether the model is installed
    bool isAvailable() const {
        return m_available;
    }

    /// Discards the results of the previous track
    void reset();
    /// Requests the analysis of the current track. Returns false if the
    /// model is not available.
    bool request();
    bool isRequested() const {
        return m_requested;
    }

    /// Called by the AnalyzerThread if requested
    void start(mixxx::audio::SampleRate sampleRate, mixxx::audio::ChannelCount channelCount);
    void processSamples(const CSAMPLE* pIn, SINT count);

    /// Returns the detector with the results for the whole track, or nullptr
    /// if the analysis has failed.
    const mixxx::DownbeatDetector* results();

  private:
    const QString m_modelPath;
    const bool m_available;
    bool m_requested;
    std::unique_ptr<mixxx::DownbeatDetector> m_pDetector;
    std::optional<bool> m_finalized;
};
