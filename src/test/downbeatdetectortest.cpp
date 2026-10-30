#include <dsp/rateconversion/Resampler.h>
#include <gtest/gtest.h>

#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "analyzer/analyzerbars.h"
#include "analyzer/constants.h"
#include "analyzer/phraseanalyzer.h"
#include "analyzer/plugins/analyzerbeatthisbeats.h"
#include "analyzer/plugins/analyzerqueenmarybeats.h"
#include "analyzer/plugins/downbeatdetector.h"
#include "sources/soundsourceproxy.h"
#include "test/mixxxtest.h"
#include "track/beatfactory.h"
#include "track/beats.h"
#include "track/track.h"
#include "util/samplebuffer.h"

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

    /// Offset found by AnalyzerBars::findBeatPhase() in seconds for a beat
    /// grid that starts at `firstBeatSeconds`
    AnalyzerBars::BeatPhase findBeatPhase(int sampleRate, double firstBeatSeconds) {
        const auto pBeats = mixxx::Beats::fromConstTempo(
                mixxx::audio::SampleRate(sampleRate),
                mixxx::audio::FramePos(firstBeatSeconds * sampleRate),
                mixxx::Bpm(kBpm));
        const auto phase = AnalyzerBars::findBeatPhase(*pBeats,
                m_detector.beatLogits(),
                mixxx::DownbeatDetector::kFramesPerSecond);
        EXPECT_TRUE(phase.has_value());
        return phase.value_or(AnalyzerBars::BeatPhase{});
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

TEST_F(DownbeatDetectorTest, BeatPhase) {
    constexpr int kSampleRate = 44100;
    runAndCompare(kSampleRate);
    const auto sampleRate = mixxx::audio::SampleRate(kSampleRate);
    constexpr double kToleranceSeconds = 0.003;

    // Correct beats
    auto phase = findBeatPhase(kSampleRate, 0);
    EXPECT_NEAR(0, phase.offset / kSampleRate, kToleranceSeconds);
    EXPECT_FALSE(AnalyzerBars::shouldCorrectBeatPhase(phase, sampleRate));

    // Beats 30 ms late
    phase = findBeatPhase(kSampleRate, 0.03);
    EXPECT_NEAR(-0.03, phase.offset / kSampleRate, kToleranceSeconds);
    EXPECT_GT(phase.activation, phase.originalActivation);
    EXPECT_TRUE(AnalyzerBars::shouldCorrectBeatPhase(phase, sampleRate));

    // Beats on the offbeats
    const double halfBeatSeconds = 30.0 / kBpm;
    phase = findBeatPhase(kSampleRate, halfBeatSeconds);
    EXPECT_NEAR(halfBeatSeconds, std::abs(phase.offset) / kSampleRate, kToleranceSeconds);
    EXPECT_TRUE(AnalyzerBars::shouldCorrectBeatPhase(phase, sampleRate));
}

TEST_F(DownbeatDetectorTest, BeatOnset) {
    constexpr int kSampleRate = 44100;
    runAndCompare(kSampleRate);
    // The kicks of the test signal start exactly at the beats
    for (const double firstBeatSeconds : {0.0, 0.03, -0.02}) {
        const auto pBeats = mixxx::Beats::fromConstTempo(
                mixxx::audio::SampleRate(kSampleRate),
                mixxx::audio::FramePos((firstBeatSeconds + 60.0 / kBpm) * kSampleRate),
                mixxx::Bpm(kBpm));
        const auto offset = AnalyzerBars::findBeatOnset(*pBeats,
                m_detector.bassEnergy(),
                mixxx::DownbeatDetector::kBassBlocksPerSecond);
        ASSERT_TRUE(offset.has_value()) << firstBeatSeconds;
        EXPECT_NEAR(-firstBeatSeconds, *offset / kSampleRate, 0.003) << firstBeatSeconds;
    }
}

TEST_F(DownbeatDetectorTest, BeatThisBeats) {
    constexpr int kSampleRate = 44100;
    runAndCompare(kSampleRate);
    const auto sampleRate = mixxx::audio::SampleRate(kSampleRate);
    const auto beats = mixxx::AnalyzerBeatThisBeats::beatsFromLogits(m_detector.beatLogits(),
            mixxx::DownbeatDetector::kFramesPerSecond,
            sampleRate);
    // All beats of the test signal, within half a frame of the logits
    const double beatFrames = 60.0 / kBpm * kSampleRate;
    EXPECT_NEAR(kSeconds * kBpm / 60, beats.size(), 2);
    for (const auto& beat : beats) {
        const double offset = beat.value() - std::round(beat.value() / beatFrames) * beatFrames;
        EXPECT_NEAR(0, offset / kSampleRate, 0.01) << beat.value() / kSampleRate;
    }

    const auto pBeats = BeatFactory::makePreferredBeats(beats, {}, true, sampleRate);
    ASSERT_NE(nullptr, pBeats);
    EXPECT_DOUBLE_EQ(kBpm,
            pBeats->getBpmInRange(mixxx::audio::kStartFramePos,
                          mixxx::audio::FramePos(kSeconds * kSampleRate))
                    .value());
    const double firstBeat = pBeats->iteratorFrom(mixxx::audio::kStartFramePos)->value();
    const double phaseError =
            std::fmod(firstBeat + beatFrames / 2, beatFrames) - beatFrames / 2;
    EXPECT_NEAR(0, phaseError / kSampleRate, 0.003);
}

TEST(AnalyzerBeatThisBeatsTest, ConstantTempoBeats) {
    constexpr int kSampleRate = 44100;
    constexpr double kBpm = 172.0;
    constexpr double kFirstBeat = 0.3;
    const double beatSeconds = 60.0 / kBpm;
    QVector<mixxx::audio::FramePos> beats;
    const auto addBeat = [&](int index, double errorSeconds) {
        beats.append(mixxx::audio::FramePos(
                (kFirstBeat + index * beatSeconds + errorSeconds) * kSampleRate));
    };
    for (int i = 0; i < 600; ++i) {
        // A break without beats
        if (i >= 50 && i < 130) {
            continue;
        }
        // Half-time part
        if (i >= 130 && i < 260 && i % 2 == 1) {
            continue;
        }
        // Jitter of the neural network
        addBeat(i, (i % 5 - 2) * 0.004);
        // Drum fill between the beats
        if (i % 64 == 63) {
            addBeat(i, beatSeconds / 2);
        }
    }

    const auto grid = mixxx::AnalyzerBeatThisBeats::constantTempoBeats(
            beats, mixxx::audio::SampleRate(kSampleRate));
    ASSERT_EQ(600, grid.size());
    EXPECT_NEAR(kFirstBeat, grid.first().value() / kSampleRate, 0.002);
    EXPECT_NEAR(beatSeconds, (grid.last() - grid.first()) / 599 / kSampleRate, 1e-5);
}

/// Development tool: compares the beats detected by Mixxx with the beat
/// probabilities of the neural network for the audio files in the directory
/// BEAT_PHASE_DIR and its subdirectories, like the analysis of the tracks.
TEST_F(DownbeatDetectorTest, DISABLED_BeatPhaseFiles) {
    ASSERT_TRUE(SoundSourceProxy::isFileTypeSupported(QStringLiteral("mp3")) ||
            SoundSourceProxy::registerProviders());
    const QDir dir(qEnvironmentVariable("BEAT_PHASE_DIR"));
    QStringList files;
    QDirIterator it(dir.path(),
            {QStringLiteral("*.mp3"),
                    QStringLiteral("*.m4a"),
                    QStringLiteral("*.flac"),
                    QStringLiteral("*.wav")},
            QDir::Files,
            QDirIterator::Subdirectories);
    while (it.hasNext()) {
        files.append(dir.relativeFilePath(it.next()));
    }
    files.sort();
    ASSERT_FALSE(files.isEmpty()) << "Set BEAT_PHASE_DIR";
    for (const QString& fileName : files) {
        const TrackPointer pTrack = Track::newTemporary(dir.filePath(fileName));
        mixxx::AudioSource::OpenParams openParams;
        openParams.setChannelCount(mixxx::audio::ChannelCount::stereo());
        const auto pAudioSource = SoundSourceProxy(pTrack).openAudioSource(openParams);
        ASSERT_NE(nullptr, pAudioSource) << fileName.toStdString();
        const auto sampleRate = pAudioSource->getSignalInfo().getSampleRate();

        mixxx::AnalyzerQueenMaryBeats beatDetector;
        ASSERT_TRUE(beatDetector.initialize(sampleRate));
        mixxx::DownbeatDetector downbeatDetector;
        ASSERT_TRUE(downbeatDetector.initialize(
                config()->getResourcePath() + QStringLiteral("models/beat_this_small.onnx"),
                sampleRate,
                mixxx::audio::ChannelCount::stereo()));
        mixxx::SampleBuffer buffer(2 * mixxx::kAnalysisFramesPerChunk);
        mixxx::IndexRange remaining = pAudioSource->frameIndexRange();
        while (!remaining.empty()) {
            const auto chunk = remaining.splitAndShrinkFront(
                    std::min(mixxx::kAnalysisFramesPerChunk, remaining.length()));
            const auto readable = pAudioSource->readSampleFrames(mixxx::WritableSampleFrames(
                    chunk, mixxx::SampleBuffer::WritableSlice(buffer)));
            const auto samples = readable.readableSlice();
            ASSERT_TRUE(beatDetector.processSamples(samples.data(), samples.length()));
            ASSERT_TRUE(downbeatDetector.processSamples(samples.data(), samples.length()));
        }
        ASSERT_TRUE(beatDetector.finalize());
        ASSERT_TRUE(downbeatDetector.finalize());

        std::printf("%s\n", fileName.toStdString().c_str());
        const auto printBeats = [&](const char* pName,
                                        const QVector<mixxx::audio::FramePos>& beats) {
            // Like AnalyzerBeats
            auto pBeats = BeatFactory::makePreferredBeats(beats, {}, true, sampleRate);
            ASSERT_NE(nullptr, pBeats);
            const auto phase = AnalyzerBars::findBeatPhase(*pBeats,
                    downbeatDetector.beatLogits(),
                    mixxx::DownbeatDetector::kFramesPerSecond);
            ASSERT_TRUE(phase.has_value());
            const bool correct = AnalyzerBars::shouldCorrectBeatPhase(*phase, sampleRate);
            const auto firstBeat = pBeats->iteratorFrom(mixxx::audio::kStartFramePos)->value();
            const auto bpm = pBeats->getBpmInRange(mixxx::audio::kStartFramePos,
                    mixxx::audio::FramePos(pAudioSource->frameLength()));
            if (correct) {
                pBeats = pBeats->tryTranslate(phase->offset).value_or(pBeats);
            }
            const auto onset = AnalyzerBars::findBeatOnset(*pBeats,
                    downbeatDetector.bassEnergy(),
                    mixxx::DownbeatDetector::kBassBlocksPerSecond);
            std::printf(
                    "  %-11s %.3f BPM, first beat %.4f s, phase offset %+.1f ms, "
                    "beat probability %.2f -> %.2f%s, onset %s\n",
                    pName,
                    bpm.value(),
                    firstBeat / sampleRate.value(),
                    phase->offset / sampleRate.value() * 1000,
                    phase->originalActivation,
                    phase->activation,
                    correct ? ", corrected" : "",
                    onset ? QString::number(*onset / sampleRate.value() * 1000, 'f', 1)
                                    .append(QStringLiteral(" ms"))
                                    .toStdString()
                                    .c_str()
                          : "none");
            if (onset) {
                pBeats = pBeats->tryTranslate(*onset).value_or(pBeats);
            }

            // Bars and sections like AnalyzerBars, with BEAT_PHASE_SECTIONS
            if (!qEnvironmentVariableIsSet("BEAT_PHASE_SECTIONS")) {
                return;
            }
            const auto pDetectedBarInfo = AnalyzerBars::barInfoFromDownbeatLogits(*pBeats,
                    downbeatDetector.downbeatLogits(),
                    mixxx::DownbeatDetector::kFramesPerSecond,
                    mixxx::audio::kStartFramePos);
            ASSERT_NE(nullptr, pDetectedBarInfo);
            auto pBarInfo = std::make_shared<mixxx::BarInfo>(*pDetectedBarInfo);
            pBarInfo->anchorBeat = mixxx::PhraseAnalyzer::findPhraseAnchor(
                    *pBeats, *pBarInfo, downbeatDetector.bandEnergies());
            std::printf("    first downbeat %d, phrase anchor beat %d\n",
                    pDetectedBarInfo->anchorBeat,
                    pBarInfo->anchorBeat);
            // Band energies of the beats in the range BEAT_PHASE_BEATS=first-last
            const QStringList beatRange = qEnvironmentVariable("BEAT_PHASE_BEATS").split('-');
            const auto& bands = downbeatDetector.bandEnergies();
            const auto toFrame = [&](int beat) {
                return static_cast<int>((pBeats->cfirstmarker() + beat)->value() /
                        sampleRate.value() * bands.framesPerSecond);
            };
            for (int beat = beatRange.size() == 2 ? beatRange[0].toInt() : 0;
                    beatRange.size() == 2 && beat <= beatRange[1].toInt();
                    ++beat) {
                const int begin = toFrame(beat);
                const int end = std::min(toFrame(beat + 1), static_cast<int>(bands.low.size()));
                const auto meanDb = [&](const std::vector<float>& band) {
                    double sum = 0;
                    for (int frame = begin; frame < end; ++frame) {
                        sum += band[frame];
                    }
                    return 10 * std::log10(sum / std::max(1, end - begin) + 1e-10);
                };
                const double low = meanDb(bands.low);
                std::printf("    beat %4d %7.2f s  phrase beat %2d  low %5.1f  mid %5.1f  "
                            "high %5.1f %s\n",
                        beat,
                        (pBeats->cfirstmarker() + beat)->value() / sampleRate.value(),
                        pBarInfo->beatInPhrase(beat),
                        low,
                        meanDb(bands.mid),
                        meanDb(bands.high),
                        std::string(std::max(0, static_cast<int>(low + 40)), '#').c_str());
            }
            for (const auto& section : mixxx::PhraseAnalyzer::analyze(
                         *pBeats, *pBarInfo, downbeatDetector.bandEnergies())) {
                const auto position = *(pBeats->cfirstmarker() + section.startBeat);
                std::printf("    %7.2f s  beat %4d  beat in phrase %2d  %s\n",
                        position.value() / sampleRate.value(),
                        section.startBeat,
                        pBarInfo->beatInPhrase(section.startBeat),
                        mixxx::phraseKindName(section.kind).toStdString().c_str());
            }
        };
        printBeats("Queen Mary", beatDetector.getBeats());
        // Like AnalyzerBeatThisBeats with fixed tempo
        const auto beatThisBeats = mixxx::AnalyzerBeatThisBeats::beatsFromLogits(
                downbeatDetector.beatLogits(),
                mixxx::DownbeatDetector::kFramesPerSecond,
                sampleRate);
        const auto beatThisGrid =
                mixxx::AnalyzerBeatThisBeats::constantTempoBeats(beatThisBeats, sampleRate);
        printBeats("Beat This!", beatThisGrid.isEmpty() ? beatThisBeats : beatThisGrid);
    }
}

} // namespace
