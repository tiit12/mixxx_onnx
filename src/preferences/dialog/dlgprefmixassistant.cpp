#include "preferences/dialog/dlgprefmixassistant.h"

#include <QCheckBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QVBoxLayout>

#include "analyzer/analyzerbars.h"
#include "control/controlobject.h"
#include "mixer/mixassistant.h"
#include "moc_dlgprefmixassistant.cpp"

DlgPrefMixAssistant::DlgPrefMixAssistant(QWidget* pParent)
        : DlgPreferencePage(pParent) {
    auto* pLayout = new QVBoxLayout(this);
    auto* pIntro = new QLabel(
            tr("The Mix Assistant uses the analyzed bars and sections of the "
               "tracks (intro, buildup, drop, breakdown, outro) to help with "
               "mixing. Each feature can be switched on and off."),
            this);
    pIntro->setWordWrap(true);
    pLayout->addWidget(pIntro);

    addOption(QStringLiteral("auto_intro_outro"),
            tr("Set the intro end and the outro start from the sections"),
            tr("Auto DJ uses them to mix phrase by phrase. Markers set by "
               "you are not changed."));
    addOption(QStringLiteral("auto_hotcues"),
            tr("Create hotcues for the sections"),
            tr("Only for tracks without hotcues of your own: hotcues at the "
               "sections, labeled and colored like the sections in the "
               "waveform. The hotcue pads of controllers trigger them."));
    auto* pHotcuesLayout = new QHBoxLayout();
    auto* pHotcuesLabel = new QLabel(
            tr("Number of hotcues, e.g. the hotcue pads of your controller:"), this);
    pHotcuesLabel->setIndent(24);
    m_pSectionHotcues = new QSpinBox(this);
    m_pSectionHotcues->setRange(1, AnalyzerBars::kMaxSectionHotcues);
    m_pSectionHotcues->setToolTip(
            tr("If a track has more sections, the most useful ones are chosen: "
               "first drop, first breakdown, last drop, outro, buildups."));
    pHotcuesLayout->addWidget(pHotcuesLabel);
    pHotcuesLayout->addWidget(m_pSectionHotcues);
    pHotcuesLayout->addStretch();
    static_cast<QVBoxLayout*>(layout())->addLayout(pHotcuesLayout);
    addOption(QStringLiteral("play_on_phrase"),
            tr("Start decks on the next phrase"),
            tr("If another deck is playing, pressing play waits for its next "
               "phrase (green countdown in the waveform), so that the phrases "
               "of both tracks are aligned. Press play again to start "
               "immediately. Cues and hotcues always start immediately."));
    addOption(QStringLiteral("bass_clash_warning"),
            tr("Warn if the bass of two decks clashes"),
            tr("Shows a red bar at the top of the waveforms if two decks play "
               "a section with full bass at the same time. Turn down the low "
               "EQ of one of the decks."));

    auto* pNote = new QLabel(
            tr("The cues are set when a track is analyzed. To place them again "
               "for a loaded track, e.g. after changing these settings, use "
               "\"Place Section Cues Again\" (keyboard Ctrl+Shift+X for deck 1, "
               "Ctrl+Shift+H for deck 2, or a controller button)."),
            this);
    pNote->setWordWrap(true);
    pLayout->addWidget(pNote);
    pLayout->addStretch();

    slotUpdate();
}

void DlgPrefMixAssistant::addOption(
        const QString& control, const QString& title, const QString& description) {
    auto* pCheckBox = new QCheckBox(title, this);
    auto* pDescription = new QLabel(description, this);
    pDescription->setWordWrap(true);
    pDescription->setIndent(24);
    auto* pLayout = static_cast<QVBoxLayout*>(layout());
    pLayout->addSpacing(8);
    pLayout->addWidget(pCheckBox);
    pLayout->addWidget(pDescription);
    m_options.push_back(Option{control, pCheckBox});
}

void DlgPrefMixAssistant::slotUpdate() {
    for (const auto& option : m_options) {
        option.pCheckBox->setChecked(
                ControlObject::get(ConfigKey(MixAssistant::kGroup, option.control)) > 0);
    }
    m_pSectionHotcues->setValue(static_cast<int>(ControlObject::get(
            ConfigKey(MixAssistant::kGroup, QStringLiteral("section_hotcues")))));
}

void DlgPrefMixAssistant::slotApply() {
    for (const auto& option : m_options) {
        ControlObject::set(ConfigKey(MixAssistant::kGroup, option.control),
                option.pCheckBox->isChecked() ? 1.0 : 0.0);
    }
    ControlObject::set(ConfigKey(MixAssistant::kGroup, QStringLiteral("section_hotcues")),
            m_pSectionHotcues->value());
}

void DlgPrefMixAssistant::slotResetToDefaults() {
    for (const auto& option : m_options) {
        option.pCheckBox->setChecked(true);
    }
    m_pSectionHotcues->setValue(AnalyzerBars::kMaxSectionHotcues);
}
