#include "analyzer/phraseanalyzer.h"

#include <gtest/gtest.h>

#include <QDataStream>
#include <QDir>
#include <QFile>
#include <array>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "track/beats.h"

using namespace mixxx;

namespace {

constexpr double kFramesPerSecond = 50;
constexpr double kBpm = 120;
// 2 seconds per bar at 120 BPM
constexpr int kFramesPerBar = 100;

struct SectionSpec {
    int bars;
    double lowDb;
    double midDb;
    double highDb;
    // Rise of the high band from the first to the last bar
    double highRiseDb = 0;
    // Changes of the spectrum from bar to bar, small in loops
    double variationDb = 3;
};

BandEnergies makeEnergies(const std::vector<SectionSpec>& sections) {
    BandEnergies energies;
    energies.framesPerSecond = kFramesPerSecond;
    int frame = 0;
    int trackBar = 0;
    for (const auto& section : sections) {
        for (int bar = 0; bar < section.bars; ++bar, ++trackBar) {
            const double rise = section.bars > 1
                    ? section.highRiseDb * bar / (section.bars - 1)
                    : 0;
            for (int i = 0; i < kFramesPerBar; ++i, ++frame) {
                // Deterministic fluctuations of +/- 2 dB
                const double noise = 2.0 * std::sin(frame * 0.37);
                const auto toPower = [&](double db) {
                    return static_cast<float>(std::pow(10.0, (db + noise) / 10.0));
                };
                energies.low.push_back(toPower(section.lowDb));
                energies.mid.push_back(toPower(section.midDb));
                energies.high.push_back(toPower(section.highDb + rise));
                std::array<float, BandEnergies::kNumSpectrumBands> spectrum;
                for (int band = 0; band < BandEnergies::kNumSpectrumBands; ++band) {
                    const double db = band < 3
                            ? section.lowDb
                            : (band < 9 ? section.midDb : section.highDb + rise);
                    spectrum[band] = toPower(
                            db + section.variationDb * std::sin(1.7 * trackBar + 0.9 * band));
                }
                energies.spectrum.push_back(spectrum);
            }
        }
    }
    return energies;
}

BeatsPointer makeBeats() {
    return Beats::fromConstTempo(audio::SampleRate(44100), audio::kStartFramePos, Bpm(kBpm));
}

std::string describe(const std::vector<Phrase>& phrases) {
    std::string result;
    for (const auto& phrase : phrases) {
        result += "bar " + std::to_string(phrase.startBeat / kDefaultBeatsPerBar) +
                " kind " + std::to_string(static_cast<int>(phrase.kind)) +
                " energy " + std::to_string(phrase.energy) + "\n";
    }
    return result;
}

BarInfo makeBarInfo(int anchorBeat) {
    BarInfo barInfo;
    barInfo.anchorBeat = anchorBeat;
    return barInfo;
}

TEST(PhraseAnalyzerTest, DanceTrackStructure) {
    const auto energies = makeEnergies({
            {32, -12, -4, -8},      // intro with kick, without bassline
            {8, -40, -6, -18, 12},  // buildup without kick, rising
            {32, 0, 0, 0},          // drop
            {16, -40, -6, -14},     // breakdown
            {8, -40, -6, -18, 12},  // buildup
            {32, 0, 0, 0},          // drop
            {16, -12, -6, -12},     // outro
    });
    const auto phrases = PhraseAnalyzer::analyze(*makeBeats(), makeBarInfo(0), energies);

    ASSERT_EQ(7u, phrases.size()) << describe(phrases);
    const std::vector<int> expectedStartBars{0, 32, 40, 72, 88, 96, 128};
    const std::vector<PhraseKind> expectedKinds{PhraseKind::Intro,
            PhraseKind::Buildup,
            PhraseKind::Drop,
            PhraseKind::Breakdown,
            PhraseKind::Buildup,
            PhraseKind::Drop,
            PhraseKind::Outro};
    for (std::size_t i = 0; i < phrases.size(); ++i) {
        EXPECT_EQ(expectedStartBars[i] * kDefaultBeatsPerBar, phrases[i].startBeat)
                << "section " << i;
        EXPECT_EQ(expectedKinds[i], phrases[i].kind) << "section " << i;
    }
    EXPECT_GT(phrases[2].energy, 0.95f);
    EXPECT_LT(phrases[3].energy, 0.5f);
}

TEST(PhraseAnalyzerTest, BoundariesAlignedToAnchor) {
    // Downbeat anchor at beat 8, i.e. the phrases start at bar 2
    const auto energies = makeEnergies({
            {18, -12, -4, -8},
            {32, 0, 0, 0},
            {16, -40, -6, -14},
            {32, 0, 0, 0},
    });
    const int anchorBeat = 8;
    const auto phrases = PhraseAnalyzer::analyze(*makeBeats(), makeBarInfo(anchorBeat), energies);

    ASSERT_GE(phrases.size(), 4u);
    EXPECT_EQ(0, phrases[0].startBeat);
    for (std::size_t i = 1; i < phrases.size(); ++i) {
        EXPECT_EQ(0, (phrases[i].startBeat - anchorBeat) % (4 * kDefaultBeatsPerBar))
                << "section " << i;
    }
    EXPECT_EQ(18 * kDefaultBeatsPerBar, phrases[1].startBeat);
    EXPECT_EQ(PhraseKind::Drop, phrases[1].kind);
    EXPECT_EQ(PhraseKind::Breakdown, phrases[2].kind);
}

TEST(PhraseAnalyzerTest, BuildupWithKickIsNotDrop) {
    // Like "I Need You" by James D: the buildup has a kick roll, i.e. a bass
    // about 7 dB below the drop, and loud highs.
    const auto energies = makeEnergies({
            {32, 0, 0, 0},           // drop
            {8, -27, -6, -10},       // breakdown
            {8, -7.5, -3, -6, 6},    // buildup with kick
            {32, 0, 0, 0},           // drop
            {16, -12, -8, -14},      // outro
    });
    const auto phrases = PhraseAnalyzer::analyze(*makeBeats(), makeBarInfo(0), energies);

    ASSERT_EQ(5u, phrases.size()) << describe(phrases);
    EXPECT_EQ(PhraseKind::Drop, phrases[0].kind) << describe(phrases);
    EXPECT_EQ(PhraseKind::Breakdown, phrases[1].kind) << describe(phrases);
    EXPECT_EQ(40 * kDefaultBeatsPerBar, phrases[2].startBeat) << describe(phrases);
    EXPECT_EQ(PhraseKind::Buildup, phrases[2].kind) << describe(phrases);
    EXPECT_EQ(PhraseKind::Drop, phrases[3].kind) << describe(phrases);
    EXPECT_EQ(PhraseKind::Outro, phrases[4].kind) << describe(phrases);
}

TEST(PhraseAnalyzerTest, PreDropIsBuildup) {
    // Like "Sugar" by Bachelors of Science: after the breakdown, drums, bass
    // and vocals return, but the highs only come back with the drop.
    const auto energies = makeEnergies({
            {32, 0, 0, 0},       // drop
            {16, -24, -4, -14},  // breakdown
            {16, -1, -3, -9},    // pre-drop
            {32, 0, 0, 0},       // drop
            {16, -12, -8, -14},  // outro
    });
    const auto phrases = PhraseAnalyzer::analyze(*makeBeats(), makeBarInfo(0), energies);

    ASSERT_EQ(5u, phrases.size()) << describe(phrases);
    EXPECT_EQ(PhraseKind::Breakdown, phrases[1].kind) << describe(phrases);
    EXPECT_EQ(48 * kDefaultBeatsPerBar, phrases[2].startBeat) << describe(phrases);
    EXPECT_EQ(PhraseKind::Buildup, phrases[2].kind) << describe(phrases);
    EXPECT_EQ(PhraseKind::Drop, phrases[3].kind) << describe(phrases);
}

TEST(PhraseAnalyzerTest, LoopAtTheEndIsOutro) {
    // Like the end of "I Need You": after the last drop, the track continues
    // with the same energy, but as a loop without variation.
    const auto energies = makeEnergies({
            {32, 0, 0, 0},                  // drop
            {8, -27, -6, -10},              // breakdown
            {8, -7.5, -3, -6, 6},           // buildup
            {32, 0, 0, 0},                  // drop
            {32, 0, -1, -2, 0, 0.3},        // loop
            {8, -20, -20, -20, 0, 0.3},     // fade out
    });
    const auto phrases = PhraseAnalyzer::analyze(*makeBeats(), makeBarInfo(0), energies);

    ASSERT_GE(phrases.size(), 5u) << describe(phrases);
    EXPECT_EQ(PhraseKind::Drop, phrases[3].kind) << describe(phrases);
    EXPECT_EQ(80 * kDefaultBeatsPerBar, phrases[4].startBeat) << describe(phrases);
    for (std::size_t i = 4; i < phrases.size(); ++i) {
        EXPECT_EQ(PhraseKind::Outro, phrases[i].kind) << describe(phrases);
    }
}

TEST(PhraseAnalyzerTest, GrooveTrackWithoutBreakdown) {
    // Like "Harmony in Blue": no breakdown, so there are no drops
    const auto energies = makeEnergies({
            {16, -3, -12, -12},  // intro
            {96, 0, 0, 0},       // groove
            {16, -3, -12, -12},  // outro
    });
    const auto phrases = PhraseAnalyzer::analyze(*makeBeats(), makeBarInfo(0), energies);

    ASSERT_GE(phrases.size(), 3u) << describe(phrases);
    EXPECT_EQ(PhraseKind::Intro, phrases.front().kind) << describe(phrases);
    EXPECT_EQ(16 * kDefaultBeatsPerBar, phrases[1].startBeat) << describe(phrases);
    EXPECT_EQ(PhraseKind::Outro, phrases.back().kind) << describe(phrases);
    EXPECT_EQ(112 * kDefaultBeatsPerBar, phrases.back().startBeat) << describe(phrases);
    for (std::size_t i = 1; i + 1 < phrases.size(); ++i) {
        EXPECT_EQ(PhraseKind::Unknown, phrases[i].kind) << describe(phrases);
    }
}

TEST(PhraseAnalyzerTest, ConstantEnergyIsUnknown) {
    // E.g. a minimal techno track without breakdowns
    const auto energies = makeEnergies({{96, 0, 0, 0}});
    const auto phrases = PhraseAnalyzer::analyze(*makeBeats(), makeBarInfo(0), energies);

    ASSERT_FALSE(phrases.empty());
    EXPECT_EQ(0, phrases[0].startBeat);
    for (const auto& phrase : phrases) {
        EXPECT_EQ(PhraseKind::Unknown, phrase.kind);
    }
}

TEST(PhraseAnalyzerTest, TooShort) {
    const auto energies = makeEnergies({{8, 0, 0, 0}});
    EXPECT_TRUE(PhraseAnalyzer::analyze(*makeBeats(), makeBarInfo(0), energies).empty());
}

/// Development tool: prints the sections of the tracks whose features were
/// dumped with tools/beatthis/dump_phrase_features.py into the directory
/// PHRASE_FEATURES_DIR.
TEST(PhraseAnalyzerTest, DISABLED_FeatureFiles) {
    const QDir dir(qEnvironmentVariable("PHRASE_FEATURES_DIR"));
    const QStringList files = dir.entryList({QStringLiteral("*.features")}, QDir::Files);
    ASSERT_FALSE(files.isEmpty()) << "Set PHRASE_FEATURES_DIR";
    constexpr int kValuesPerFrame = 3 + BandEnergies::kNumSpectrumBands;
    const char* kKindNames[] = {"Unknown", "Intro", "Buildup", "Drop", "Breakdown", "Outro"};
    for (const QString& fileName : files) {
        QFile file(dir.filePath(fileName));
        ASSERT_TRUE(file.open(QIODevice::ReadOnly));
        QDataStream stream(&file);
        stream.setByteOrder(QDataStream::LittleEndian);
        qint32 numFrames;
        double bpm;
        double firstDownbeat;
        BandEnergies energies;
        stream.setFloatingPointPrecision(QDataStream::DoublePrecision);
        stream >> numFrames >> energies.framesPerSecond >> bpm >> firstDownbeat;
        stream.setFloatingPointPrecision(QDataStream::SinglePrecision);
        for (int frame = 0; frame < numFrames; ++frame) {
            std::array<float, kValuesPerFrame> values;
            for (auto& value : values) {
                stream >> value;
            }
            energies.low.push_back(values[0]);
            energies.mid.push_back(values[1]);
            energies.high.push_back(values[2]);
            std::array<float, BandEnergies::kNumSpectrumBands> spectrum;
            std::copy(values.begin() + 3, values.end(), spectrum.begin());
            energies.spectrum.push_back(spectrum);
        }

        constexpr auto kSampleRate = audio::SampleRate(44100);
        const auto pBeats = Beats::fromConstTempo(kSampleRate,
                audio::FramePos(std::round(firstDownbeat * kSampleRate.value())),
                Bpm(bpm));
        const auto phrases = PhraseAnalyzer::analyze(*pBeats, makeBarInfo(0), energies);
        std::string result = fileName.toStdString() + "\n";
        for (const auto& phrase : phrases) {
            const double seconds = (*(pBeats->cfirstmarker() + phrase.startBeat)).value() /
                    kSampleRate.value();
            char line[100];
            std::snprintf(line,
                    sizeof(line),
                    "  bar %3d  %2d:%04.1f  %-9s energy %.2f\n",
                    phrase.startBeat / kDefaultBeatsPerBar,
                    static_cast<int>(seconds) / 60,
                    std::fmod(seconds, 60.0),
                    kKindNames[static_cast<int>(phrase.kind)],
                    phrase.energy);
            result += line;
        }
        std::printf("%s", result.c_str());
    }
}

} // namespace
