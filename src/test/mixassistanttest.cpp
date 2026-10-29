#include "mixer/mixassistant.h"

#include <gtest/gtest.h>

#include "analyzer/analyzerbars.h"
#include "control/controlobject.h"
#include "test/mixxxtest.h"
#include "track/track.h"

namespace {

constexpr auto kSampleRate = mixxx::audio::SampleRate(44100);
// 120 BPM
constexpr double kFramesPerBeat = 22050;

TEST(MixAssistantTest, CrossfaderGain) {
    constexpr double kLeft = 0;
    constexpr double kCenter = 1;
    constexpr double kRight = 2;
    EXPECT_DOUBLE_EQ(1.0, MixAssistant::crossfaderGain(-1.0, kLeft));
    EXPECT_DOUBLE_EQ(1.0, MixAssistant::crossfaderGain(0.0, kLeft));
    EXPECT_DOUBLE_EQ(0.0, MixAssistant::crossfaderGain(1.0, kLeft));
    EXPECT_DOUBLE_EQ(0.0, MixAssistant::crossfaderGain(-1.0, kRight));
    EXPECT_DOUBLE_EQ(0.5, MixAssistant::crossfaderGain(-0.5, kRight));
    EXPECT_DOUBLE_EQ(1.0, MixAssistant::crossfaderGain(1.0, kRight));
    EXPECT_DOUBLE_EQ(1.0, MixAssistant::crossfaderGain(-1.0, kCenter));
}

TEST(MixAssistantTest, BassAudible) {
    // Drop at full volume
    EXPECT_TRUE(MixAssistant::isBassAudible(1.0, 1.0, 1.0, 1.0, false));
    // Breakdown without bass
    EXPECT_FALSE(MixAssistant::isBassAudible(0.2, 1.0, 1.0, 1.0, false));
    // Unknown sections
    EXPECT_FALSE(MixAssistant::isBassAudible(-1.0, 1.0, 1.0, 1.0, false));
    // Faded out with the volume or the crossfader
    EXPECT_FALSE(MixAssistant::isBassAudible(1.0, 0.05, 1.0, 1.0, false));
    EXPECT_FALSE(MixAssistant::isBassAudible(1.0, 1.0, 0.0, 1.0, false));
    // Low EQ turned down or killed
    EXPECT_FALSE(MixAssistant::isBassAudible(1.0, 1.0, 1.0, 0.2, false));
    EXPECT_FALSE(MixAssistant::isBassAudible(1.0, 1.0, 1.0, 1.0, true));
}

class SectionCuesTest : public MixxxTest {
  protected:
    void SetUp() override {
        m_pTrack = Track::newTemporary();
        m_pTrack->setAudioProperties(mixxx::audio::ChannelCount::stereo(),
                kSampleRate,
                mixxx::audio::Bitrate(),
                mixxx::Duration::fromSeconds(200));
        auto pBarInfo = std::make_shared<mixxx::BarInfo>();
        pBarInfo->phrases = {
                mixxx::Phrase{0, mixxx::PhraseKind::Intro},
                mixxx::Phrase{64, mixxx::PhraseKind::Buildup},
                mixxx::Phrase{96, mixxx::PhraseKind::Drop},
                mixxx::Phrase{224, mixxx::PhraseKind::Breakdown},
                mixxx::Phrase{320, mixxx::PhraseKind::Outro},
        };
        ASSERT_TRUE(m_pTrack->trySetBeats(
                mixxx::Beats::fromConstTempo(
                        kSampleRate, mixxx::audio::kStartFramePos, mixxx::Bpm(120))
                        ->withBarInfo(pBarInfo)));
        // Created by AnalyzerSilence
        m_pTrack->createAndAddCue(mixxx::CueType::Intro,
                Cue::kNoHotCue,
                mixxx::audio::kStartFramePos,
                mixxx::audio::kInvalidFramePos);
        m_pTrack->createAndAddCue(mixxx::CueType::Outro,
                Cue::kNoHotCue,
                mixxx::audio::kInvalidFramePos,
                mixxx::audio::FramePos(400 * kFramesPerBeat));
    }

    void setupCues(bool introAndOutro, bool hotcues) {
        AnalyzerBars::setupCuesFromSections(
                m_pTrack.get(), *m_pTrack->getBeats(), introAndOutro, hotcues);
    }

    QList<CuePointer> hotcues() const {
        QList<CuePointer> result;
        for (const auto& pCue : m_pTrack->getCuePoints()) {
            if (pCue->getHotCue() != Cue::kNoHotCue) {
                result.append(pCue);
            }
        }
        return result;
    }

    TrackPointer m_pTrack;
};

TEST_F(SectionCuesTest, IntroOutroAndHotcues) {
    setupCues(true, true);

    const CuePointer pIntro = m_pTrack->findCueByType(mixxx::CueType::Intro);
    EXPECT_EQ(mixxx::audio::kStartFramePos, pIntro->getPosition());
    EXPECT_EQ(mixxx::audio::FramePos(64 * kFramesPerBeat), pIntro->getEndPosition());
    const CuePointer pOutro = m_pTrack->findCueByType(mixxx::CueType::Outro);
    EXPECT_EQ(mixxx::audio::FramePos(320 * kFramesPerBeat), pOutro->getPosition());
    EXPECT_EQ(mixxx::audio::FramePos(400 * kFramesPerBeat), pOutro->getEndPosition());

    // One hotcue per section, except the intro
    const QList<CuePointer> cues = hotcues();
    ASSERT_EQ(4, cues.size());
    const QStringList labels{"Buildup", "Drop", "Breakdown", "Outro"};
    const QList<int> beats{64, 96, 224, 320};
    for (const auto& pCue : cues) {
        const int index = pCue->getHotCue();
        ASSERT_LT(index, 4);
        EXPECT_EQ(labels[index], pCue->getLabel());
        EXPECT_EQ(mixxx::audio::FramePos(beats[index] * kFramesPerBeat), pCue->getPosition());
    }
}

TEST_F(SectionCuesTest, Disabled) {
    setupCues(false, false);

    EXPECT_FALSE(
            m_pTrack->findCueByType(mixxx::CueType::Intro)->getEndPosition().isValid());
    EXPECT_FALSE(m_pTrack->findCueByType(mixxx::CueType::Outro)->getPosition().isValid());
    EXPECT_TRUE(hotcues().isEmpty());
}

TEST_F(SectionCuesTest, CuesFollowTranslatedBeats) {
    setupCues(true, true);
    // A hotcue of the user at the start of a section
    m_pTrack->createAndAddCue(mixxx::CueType::HotCue,
            7,
            mixxx::audio::FramePos(96 * kFramesPerBeat),
            mixxx::audio::kInvalidFramePos);

    constexpr double kOffset = 1000;
    ASSERT_TRUE(m_pTrack->trySetBeats(*m_pTrack->getBeats()->tryTranslate(kOffset)));

    EXPECT_EQ(mixxx::audio::FramePos(64 * kFramesPerBeat + kOffset),
            m_pTrack->findCueByType(mixxx::CueType::Intro)->getEndPosition());
    EXPECT_EQ(mixxx::audio::FramePos(320 * kFramesPerBeat + kOffset),
            m_pTrack->findCueByType(mixxx::CueType::Outro)->getPosition());
    for (const auto& pCue : hotcues()) {
        if (pCue->getHotCue() == 7) {
            EXPECT_EQ(mixxx::audio::FramePos(96 * kFramesPerBeat), pCue->getPosition());
        } else if (pCue->getLabel() == QStringLiteral("Drop")) {
            EXPECT_EQ(mixxx::audio::FramePos(96 * kFramesPerBeat + kOffset),
                    pCue->getPosition());
        }
    }
}

TEST_F(SectionCuesTest, CuesFollowShiftedDownbeats) {
    setupCues(true, true);

    // Like bar_shift_later: all sections start one beat later
    auto pBarInfo = std::make_shared<mixxx::BarInfo>(*m_pTrack->getBeats()->getBarInfo());
    for (auto& section : pBarInfo->phrases) {
        section.startBeat += 1;
    }
    ASSERT_TRUE(m_pTrack->trySetBarInfo(pBarInfo));

    EXPECT_EQ(mixxx::audio::FramePos(65 * kFramesPerBeat),
            m_pTrack->findCueByType(mixxx::CueType::Intro)->getEndPosition());
    for (const auto& pCue : hotcues()) {
        if (pCue->getLabel() == QStringLiteral("Breakdown")) {
            EXPECT_EQ(mixxx::audio::FramePos(225 * kFramesPerBeat), pCue->getPosition());
        }
    }
}

TEST_F(SectionCuesTest, Reset) {
    ControlObject autoIntroOutro(ConfigKey(MixAssistant::kGroup, "auto_intro_outro"));
    autoIntroOutro.set(1);
    ControlObject autoHotcues(ConfigKey(MixAssistant::kGroup, "auto_hotcues"));
    autoHotcues.set(1);
    setupCues(true, true);
    // E.g. placed before the beats were corrected
    const auto wrongPosition = mixxx::audio::FramePos(100 * kFramesPerBeat);
    for (const auto& pCue : hotcues()) {
        pCue->setStartAndEndPosition(wrongPosition, mixxx::audio::kInvalidFramePos);
    }
    m_pTrack->findCueByType(mixxx::CueType::Intro)
            ->setStartAndEndPosition(mixxx::audio::kStartFramePos, wrongPosition);

    AnalyzerBars::resetCuesFromSections(m_pTrack.get());

    EXPECT_EQ(mixxx::audio::FramePos(64 * kFramesPerBeat),
            m_pTrack->findCueByType(mixxx::CueType::Intro)->getEndPosition());
    ASSERT_EQ(4, hotcues().size());
    for (const auto& pCue : hotcues()) {
        EXPECT_NE(wrongPosition, pCue->getPosition());
    }
}

TEST_F(SectionCuesTest, MostUsefulSectionsForFourPads) {
    auto pBarInfo = std::make_shared<mixxx::BarInfo>();
    pBarInfo->phrases = {
            mixxx::Phrase{0, mixxx::PhraseKind::Intro},
            mixxx::Phrase{32, mixxx::PhraseKind::Buildup},
            mixxx::Phrase{40, mixxx::PhraseKind::Drop},
            mixxx::Phrase{104, mixxx::PhraseKind::Breakdown},
            mixxx::Phrase{120, mixxx::PhraseKind::Buildup},
            mixxx::Phrase{128, mixxx::PhraseKind::Drop},
            mixxx::Phrase{192, mixxx::PhraseKind::Outro},
    };
    ASSERT_TRUE(m_pTrack->trySetBarInfo(pBarInfo));

    AnalyzerBars::setupCuesFromSections(m_pTrack.get(), *m_pTrack->getBeats(), false, true, 4);

    // First drop, first breakdown, last drop and outro in the order of the track
    const QList<int> expectedBeats{40, 104, 128, 192};
    const QStringList expectedLabels{"Drop", "Breakdown", "Drop", "Outro"};
    const QList<CuePointer> cues = hotcues();
    ASSERT_EQ(4, cues.size());
    for (const auto& pCue : cues) {
        const int index = pCue->getHotCue();
        ASSERT_LT(index, 4);
        EXPECT_EQ(mixxx::audio::FramePos(expectedBeats[index] * kFramesPerBeat),
                pCue->getPosition());
        EXPECT_EQ(expectedLabels[index], pCue->getLabel());
    }
}

TEST_F(SectionCuesTest, UserCuesAreKept) {
    const auto userIntroEnd = mixxx::audio::FramePos(32 * kFramesPerBeat);
    m_pTrack->findCueByType(mixxx::CueType::Intro)
            ->setStartAndEndPosition(mixxx::audio::kStartFramePos, userIntroEnd);
    m_pTrack->createAndAddCue(mixxx::CueType::HotCue,
            5,
            mixxx::audio::FramePos(10 * kFramesPerBeat),
            mixxx::audio::kInvalidFramePos);

    setupCues(true, true);

    EXPECT_EQ(userIntroEnd, m_pTrack->findCueByType(mixxx::CueType::Intro)->getEndPosition());
    ASSERT_EQ(1, hotcues().size());
    EXPECT_EQ(5, hotcues().front()->getHotCue());
}

} // namespace
