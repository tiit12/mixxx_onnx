#pragma once

#include <QObject>
#include <QString>
#include <QTimer>
#include <array>
#include <memory>

#include "control/pollingcontrolproxy.h"

class ControlObject;

/// Helps to mix based on the bar and phrase analysis of the tracks. Each
/// feature can be switched on and off with a persistent control in the group
/// [MixAssistant], see DlgPrefMixAssistant:
///
/// - auto_intro_outro, auto_hotcues, section_hotcues: cues placed by
///   AnalyzerBars
/// - play_on_phrase: A deck that is started while another deck is playing
///   waits for the next phrase of the playing deck. EngineBuffer arms the deck
///   instead of starting it, and this class launches it.
/// - bass_clash_warning: `[ChannelN],bass_clash` is set if the bass of two
///   decks is audible at the same time, e.g. two drops without EQ.
class MixAssistant : public QObject {
    Q_OBJECT
  public:
    static constexpr int kNumDecks = 4;
    static const QString kGroup;

    explicit MixAssistant(QObject* pParent = nullptr);
    ~MixAssistant() override;

    /// Approximate gain of a deck with the orientation (0 = left, 1 = center,
    /// 2 = right) at the crossfader position (-1 = left, 1 = right).
    static double crossfaderGain(double crossfader, double orientation);

    /// Returns true if the bass of the current section of a playing deck is
    /// audible. `sectionBass` is the bass level of the section (-1 if
    /// unknown), `eqLow` the gain of the low EQ (1 = neutral).
    static bool isBassAudible(double sectionBass,
            double volume,
            double crossfaderGain,
            double eqLow,
            bool lowKill);

  private slots:
    void slotUpdate();

  private:
    struct Deck {
        QString group;
        std::unique_ptr<PollingControlProxy> pPlay;
        std::unique_ptr<PollingControlProxy> pBeatsUntilPhrase;
        std::unique_ptr<PollingControlProxy> pSectionBass;
        std::unique_ptr<PollingControlProxy> pVolume;
        std::unique_ptr<PollingControlProxy> pOrientation;
        std::unique_ptr<PollingControlProxy> pEqLow;
        std::unique_ptr<PollingControlProxy> pEqLowKill;
        std::unique_ptr<PollingControlProxy> pArmed;
        std::unique_ptr<PollingControlProxy> pLaunch;
        std::unique_ptr<ControlObject> pBassClash;
        // Beats of the playing deck until its next phrase, while armed
        std::unique_ptr<ControlObject> pPhraseCountdown;
        int referenceDeck = -1;
        double lastReferenceBeatsUntilPhrase = -1;
    };

    /// Returns the proxy of the control, which is created once the control
    /// exists, or nullptr.
    static PollingControlProxy* proxy(std::unique_ptr<PollingControlProxy>* ppProxy,
            const QString& group,
            const QString& item);
    static double value(std::unique_ptr<PollingControlProxy>* ppProxy,
            const QString& group,
            const QString& item,
            double defaultValue);

    void updatePhraseLaunch();
    void updateBassClash();
    void launch(Deck* pDeck);

    std::unique_ptr<ControlObject> m_pAutoIntroOutro;
    std::unique_ptr<ControlObject> m_pAutoHotcues;
    // Maximum number of hotcues of the sections, e.g. the pads of the controller
    std::unique_ptr<ControlObject> m_pSectionHotcues;
    std::unique_ptr<ControlObject> m_pPlayOnPhrase;
    std::unique_ptr<ControlObject> m_pBassClashWarning;
    std::unique_ptr<PollingControlProxy> m_pCrossfader;
    std::unique_ptr<PollingControlProxy> m_pAutoDjEnabled;
    std::array<Deck, kNumDecks> m_decks;
    QTimer m_timer;
    int m_tick;
};
