#include "mixer/mixassistant.h"

#include <algorithm>

#include "control/controlobject.h"
#include "mixer/playermanager.h"
#include "moc_mixassistant.cpp"

namespace {

// The launch of an armed deck is at most this late; EngineBuffer aligns the
// phase afterwards.
constexpr int kUpdateIntervalMillis = 10;
constexpr int kBassClashUpdateTicks = 10;

// A section with a bass level of at least this value has a full bass
constexpr double kMinSectionBass = 0.6;
// Volume times crossfader gain
constexpr double kMinAudibleGain = 0.1;
// Low EQ gain, i.e. the bass is cut by less than 6 dB
constexpr double kMinEqLow = 0.5;

} // namespace

const QString MixAssistant::kGroup = QStringLiteral("[MixAssistant]");

MixAssistant::MixAssistant(QObject* pParent)
        : QObject(pParent),
          m_pAutoIntroOutro(std::make_unique<ControlObject>(
                  ConfigKey(kGroup, QStringLiteral("auto_intro_outro")),
                  true,
                  false,
                  true,
                  1.0)),
          m_pAutoHotcues(std::make_unique<ControlObject>(
                  ConfigKey(kGroup, QStringLiteral("auto_hotcues")),
                  true,
                  false,
                  true,
                  1.0)),
          m_pSectionHotcues(std::make_unique<ControlObject>(
                  ConfigKey(kGroup, QStringLiteral("section_hotcues")),
                  true,
                  false,
                  true,
                  8.0)),
          m_pPlayOnPhrase(std::make_unique<ControlObject>(
                  ConfigKey(kGroup, QStringLiteral("play_on_phrase")),
                  true,
                  false,
                  true,
                  1.0)),
          m_pBassClashWarning(std::make_unique<ControlObject>(
                  ConfigKey(kGroup, QStringLiteral("bass_clash_warning")),
                  true,
                  false,
                  true,
                  1.0)),
          m_tick(0) {
    for (int i = 0; i < kNumDecks; ++i) {
        Deck& deck = m_decks[i];
        deck.group = PlayerManager::groupForDeck(i);
        deck.pBassClash = std::make_unique<ControlObject>(
                ConfigKey(deck.group, QStringLiteral("bass_clash")));
        deck.pBassClash->setReadOnly();
        deck.pPhraseCountdown = std::make_unique<ControlObject>(
                ConfigKey(deck.group, QStringLiteral("play_on_phrase_countdown")));
        deck.pPhraseCountdown->setReadOnly();
    }
    connect(&m_timer, &QTimer::timeout, this, &MixAssistant::slotUpdate);
    m_timer.start(kUpdateIntervalMillis);
}

MixAssistant::~MixAssistant() = default;

// static
double MixAssistant::crossfaderGain(double crossfader, double orientation) {
    if (orientation < 0.5) {
        // left
        return std::clamp(1.0 - crossfader, 0.0, 1.0);
    }
    if (orientation > 1.5) {
        // right
        return std::clamp(1.0 + crossfader, 0.0, 1.0);
    }
    return 1.0;
}

// static
bool MixAssistant::isBassAudible(double sectionBass,
        double volume,
        double crossfaderGain,
        double eqLow,
        bool lowKill) {
    return sectionBass >= kMinSectionBass && volume * crossfaderGain >= kMinAudibleGain &&
            eqLow >= kMinEqLow && !lowKill;
}

// static
PollingControlProxy* MixAssistant::proxy(std::unique_ptr<PollingControlProxy>* ppProxy,
        const QString& group,
        const QString& item) {
    if (!*ppProxy) {
        const ConfigKey key(group, item);
        // The controls of the decks and the effects are created later
        if (!ControlObject::exists(key)) {
            return nullptr;
        }
        *ppProxy = std::make_unique<PollingControlProxy>(key);
    }
    return ppProxy->get();
}

// static
double MixAssistant::value(std::unique_ptr<PollingControlProxy>* ppProxy,
        const QString& group,
        const QString& item,
        double defaultValue) {
    const PollingControlProxy* pProxy = proxy(ppProxy, group, item);
    return pProxy ? pProxy->get() : defaultValue;
}

void MixAssistant::slotUpdate() {
    updatePhraseLaunch();
    if (++m_tick >= kBassClashUpdateTicks) {
        m_tick = 0;
        updateBassClash();
    }
}

void MixAssistant::launch(Deck* pDeck) {
    PollingControlProxy* pLaunch =
            proxy(&pDeck->pLaunch, pDeck->group, QStringLiteral("play_on_phrase_launch"));
    if (pLaunch) {
        pLaunch->set(1.0);
        pLaunch->set(0.0);
    }
    pDeck->referenceDeck = -1;
    pDeck->lastReferenceBeatsUntilPhrase = -1;
    pDeck->pPhraseCountdown->forceSet(0);
}

void MixAssistant::updatePhraseLaunch() {
    const bool autoDj = value(&m_pAutoDjEnabled,
                                QStringLiteral("[AutoDJ]"),
                                QStringLiteral("enabled"),
                                0) > 0;
    for (int i = 0; i < kNumDecks; ++i) {
        Deck& deck = m_decks[i];
        if (value(&deck.pArmed, deck.group, QStringLiteral("play_on_phrase_armed"), 0) <= 0) {
            if (deck.referenceDeck >= 0) {
                deck.referenceDeck = -1;
                deck.lastReferenceBeatsUntilPhrase = -1;
                deck.pPhraseCountdown->forceSet(0);
            }
            continue;
        }
        // Auto DJ times the transitions itself
        if (!m_pPlayOnPhrase->toBool() || autoDj) {
            launch(&deck);
            continue;
        }

        // The playing deck with a known phrase structure
        int reference = -1;
        double beatsUntilPhrase = -1;
        for (int j = 0; j < kNumDecks; ++j) {
            Deck& other = m_decks[j];
            if (j == i || value(&other.pPlay, other.group, QStringLiteral("play"), 0) <= 0) {
                continue;
            }
            beatsUntilPhrase = value(&other.pBeatsUntilPhrase,
                    other.group,
                    QStringLiteral("beats_until_phrase"),
                    -1);
            if (beatsUntilPhrase >= 1) {
                reference = j;
                break;
            }
        }
        if (reference < 0) {
            // Nothing to wait for
            launch(&deck);
            continue;
        }
        if (reference == deck.referenceDeck &&
                beatsUntilPhrase > deck.lastReferenceBeatsUntilPhrase) {
            // The countdown restarted, i.e. a new phrase started
            launch(&deck);
            continue;
        }
        deck.referenceDeck = reference;
        deck.lastReferenceBeatsUntilPhrase = beatsUntilPhrase;
        deck.pPhraseCountdown->forceSet(beatsUntilPhrase);
    }
}

void MixAssistant::updateBassClash() {
    const double crossfader =
            value(&m_pCrossfader, QStringLiteral("[Master]"), QStringLiteral("crossfader"), 0);
    std::array<bool, kNumDecks> audible{};
    int numAudible = 0;
    if (m_pBassClashWarning->toBool()) {
        for (int i = 0; i < kNumDecks; ++i) {
            Deck& deck = m_decks[i];
            if (value(&deck.pPlay, deck.group, QStringLiteral("play"), 0) <= 0) {
                continue;
            }
            audible[i] = isBassAudible(
                    value(&deck.pSectionBass, deck.group, QStringLiteral("section_bass"), -1),
                    value(&deck.pVolume, deck.group, QStringLiteral("volume"), 1),
                    crossfaderGain(crossfader,
                            value(&deck.pOrientation,
                                    deck.group,
                                    QStringLiteral("orientation"),
                                    1)),
                    value(&deck.pEqLow, deck.group, QStringLiteral("filterLow"), 1),
                    value(&deck.pEqLowKill, deck.group, QStringLiteral("filterLowKill"), 0) >
                            0);
            if (audible[i]) {
                numAudible++;
            }
        }
    }
    for (int i = 0; i < kNumDecks; ++i) {
        m_decks[i].pBassClash->forceSet(audible[i] && numAudible >= 2 ? 1 : 0);
    }
}
