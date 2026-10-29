#include "analyzer/neuralanalysis.h"

#include <QFileInfo>

#include "analyzer/plugins/downbeatdetector.h"

namespace {

const QString kModelFile = QStringLiteral("models/beat_this_small.onnx");

} // namespace

NeuralAnalysis::NeuralAnalysis(const UserSettingsPointer& pConfig)
        : m_modelPath(modelPath(pConfig)),
          m_available(QFileInfo::exists(m_modelPath)),
          m_requested(false) {
}

NeuralAnalysis::~NeuralAnalysis() = default;

// static
QString NeuralAnalysis::modelPath(const UserSettingsPointer& pConfig) {
    return pConfig->getResourcePath() + kModelFile;
}

void NeuralAnalysis::reset() {
    m_requested = false;
    m_pDetector.reset();
    m_finalized.reset();
}

bool NeuralAnalysis::request() {
    if (m_available) {
        m_requested = true;
    }
    return m_requested;
}

void NeuralAnalysis::start(
        mixxx::audio::SampleRate sampleRate, mixxx::audio::ChannelCount channelCount) {
    DEBUG_ASSERT(m_requested);
    m_pDetector = std::make_unique<mixxx::DownbeatDetector>();
    if (!m_pDetector->initialize(m_modelPath, sampleRate, channelCount)) {
        m_pDetector.reset();
    }
}

void NeuralAnalysis::processSamples(const CSAMPLE* pIn, SINT count) {
    if (m_pDetector && !m_pDetector->processSamples(pIn, count)) {
        m_pDetector.reset();
    }
}

const mixxx::DownbeatDetector* NeuralAnalysis::results() {
    if (!m_pDetector) {
        return nullptr;
    }
    if (!m_finalized) {
        m_finalized = m_pDetector->finalize();
    }
    return *m_finalized ? m_pDetector.get() : nullptr;
}
