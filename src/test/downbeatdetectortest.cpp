#include <dsp/rateconversion/Resampler.h>
#include <gtest/gtest.h>

#include <QFile>
#include <array>
#include <cmath>
#include <vector>

#include "analyzer/analyzerbars.h"
#include "analyzer/plugins/downbeatdetector.h"
#include "test/mixxxtest.h"
#include "track/beats.h"

namespace {

constexpr double kSeconds = 70.0;
constexpr double kBpm = 126.0;
constexpr std::array<double, 4> kBassHz{55.0, 73.42, 82.41, 65.41};

void addDecayingSine(std::vector<double>* pSignal,
        int sampleRate,
        int start,
        int length,
        double gain,
        double decaySeconds,
        double frequency) {
    for (int i = 0; i < length; ++i) {
        const double t = static_cast<double>(i) / sampleRate;
        double value = gain * std::sin(2 * M_PI * frequency * t);
        if (decaySeconds > 0) {
            value *= std::exp(-t / decaySeconds);
        }
        (*pSignal)[start + i] += value;
    }
}

/// Same as test_signal() in tools/beatthis/make_test_reference.py
std::vector<float> testSignal(int sampleRate) {
    const int n = static_cast<int>(kSeconds * sampleRate);
    std::vector<double> signal(n, 0.0);
    const double beatFrames = 60.0 / kBpm * sampleRate;
    const int numBeats = static_cast<int>(n / beatFrames);
    for (int i = 0; i < numBeats; ++i) {
        const int start = static_cast<int>(std::round(i * beatFrames));
        addDecayingSine(&signal,
                sampleRate,
                start,
                std::min(static_cast<int>(0.15 * sampleRate), n - start),
                i % 4 == 0 ? 1.0 : 0.6,
                0.03,
                60.0);
        if (i % 4 == 1 || i % 4 == 3) {
            addDecayingSine(&signal,
                    sampleRate,
                    start,
                    std::min(static_cast<int>(0.1 * sampleRate), n - start),
                    0.3,
                    0.02,
                    190.0);
        }
        if (i % 4 == 0) {
            addDecayingSine(&signal,
                    sampleRate,
                    start,
                    std::min(static_cast<int>(std::round(4 * beatFrames)), n - start),
                    0.25,
                    0.0,
                    kBassHz[(i / 4) % 4]);
        }
    }
    std::vector<float> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = static_cast<float>(0.5 * signal[i]);
    }
    return result;
}

struct Deviation {
    double max = 0;
    double mean = 0;
    // Frames where the beat or downbeat decision (probability > 0.5) differs
    int decisionMismatches = 0;
};

TEST(DownbeatDetectorResamplerTest, UnityGain) {
    for (const int sourceRate : {44100, 48000}) {
        Resampler resampler(sourceRate, mixxx::DownbeatDetector::kSampleRate);
        std::vector<double> in(sourceRate);
        for (std::size_t i = 0; i < in.size(); ++i) {
            in[i] = std::sin(2 * M_PI * 100.0 * i / sourceRate);
        }
        std::vector<double> out(in.size());
        const int n = resampler.process(in.data(), out.data(), static_cast<int>(in.size()));
        double peak = 0;
        for (int i = n / 2; i < n; ++i) {
            peak = std::max(peak, std::abs(out[i]));
        }
        EXPECT_NEAR(1.0, peak, 0.005) << "source rate " << sourceRate;
    }
}

class DownbeatDetectorTest : public MixxxTest {
  protected:
    /// Runs the detector on the test signal and compares the logits with
    /// those of the PyTorch implementation, created with
    /// tools/beatthis/make_test_reference.py
    Deviation runAndCompare(int sampleRate) {
        QFile referenceFile(getTestDir().filePath(
                QStringLiteral("beatthis/reference_logits_%1.bin").arg(sampleRate)));
        EXPECT_TRUE(referenceFile.open(QIODevice::ReadOnly));
        const QByteArray referenceData = referenceFile.readAll();
        const auto numFrames = static_cast<std::size_t>(referenceData.size()) /
                (2 * sizeof(float));
        const auto* pReferenceBeat = reinterpret_cast<const float*>(referenceData.constData());
        const float* pReferenceDownbeat = pReferenceBeat + numFrames;

        const std::vector<float> mono = testSignal(sampleRate);
        std::vector<CSAMPLE> stereo(2 * mono.size());
        for (std::size_t i = 0; i < mono.size(); ++i) {
            stereo[2 * i] = mono[i];
            stereo[2 * i + 1] = mono[i];
        }

        const bool initialized = m_detector.initialize(
                config()->getResourcePath() + QStringLiteral("models/beat_this_small.onnx"),
                mixxx::audio::SampleRate(sampleRate),
                mixxx::audio::ChannelCount::stereo());
        EXPECT_TRUE(initialized);
        if (!initialized) {
            return {};
        }
        constexpr SINT kChunkSamples = 2 * 4096;
        for (SINT i = 0; i < static_cast<SINT>(stereo.size()); i += kChunkSamples) {
            EXPECT_TRUE(m_detector.processSamples(&stereo[i],
                    std::min(kChunkSamples, static_cast<SINT>(stereo.size()) - i)));
        }
        EXPECT_TRUE(m_detector.finalize());

        const auto& beat = m_detector.beatLogits();
        const auto& downbeat = m_detector.downbeatLogits();
        EXPECT_EQ(numFrames, beat.size());
        EXPECT_EQ(numFrames, downbeat.size());

        Deviation deviation;
        const std::size_t n = std::min(numFrames, beat.size());
        for (std::size_t i = 0; i < n; ++i) {
            const double diff = std::max(std::abs(beat[i] - pReferenceBeat[i]),
                    std::abs(downbeat[i] - pReferenceDownbeat[i]));
            deviation.max = std::max(deviation.max, diff);
            deviation.mean += diff / n;
            if ((downbeat[i] > 0) != (pReferenceDownbeat[i] > 0) ||
                    (beat[i] > 0) != (pReferenceBeat[i] > 0)) {
                deviation.decisionMismatches++;
            }
        }
        return deviation;
    }

    void expectDownbeatsOnFirstBeat(int sampleRate) {
        const auto pBeats = mixxx::Beats::fromConstTempo(
                mixxx::audio::SampleRate(sampleRate),
                mixxx::audio::kStartFramePos,
                mixxx::Bpm(kBpm));
        const auto pBarInfo = AnalyzerBars::barInfoFromDownbeatLogits(*pBeats,
                m_detector.downbeatLogits(),
                mixxx::DownbeatDetector::kFramesPerSecond,
                mixxx::audio::kStartFramePos);
        ASSERT_NE(nullptr, pBarInfo);
        EXPECT_EQ(0, pBarInfo->anchorBeat);
    }

    mixxx::DownbeatDetector m_detector;
};

TEST_F(DownbeatDetectorTest, MatchesReferenceAtModelSampleRate) {
    // Without resampling the STFT, chunking and inference must match
    const Deviation deviation = runAndCompare(mixxx::DownbeatDetector::kSampleRate);
    EXPECT_LT(deviation.max, 0.01);
    EXPECT_EQ(0, deviation.decisionMismatches);
    expectDownbeatsOnFirstBeat(mixxx::DownbeatDetector::kSampleRate);
}

TEST_F(DownbeatDetectorTest, MatchesReferenceWithResampling) {
    // The resampler differs from the one of the reference (soxr), which
    // mainly affects the highest mel bands.
    const Deviation deviation = runAndCompare(44100);
    EXPECT_LT(deviation.mean, 0.15);
    EXPECT_LT(deviation.max, 1.0);
    EXPECT_LE(deviation.decisionMismatches, 10);
    expectDownbeatsOnFirstBeat(44100);
}

} // namespace
