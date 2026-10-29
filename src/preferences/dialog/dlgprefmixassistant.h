#pragma once

#include <QString>
#include <vector>

#include "preferences/dialog/dlgpreferencepage.h"

class QCheckBox;
class QSpinBox;

/// Switches the features of the MixAssistant on and off.
class DlgPrefMixAssistant : public DlgPreferencePage {
    Q_OBJECT
  public:
    explicit DlgPrefMixAssistant(QWidget* pParent);

  public slots:
    void slotUpdate() override;
    void slotApply() override;
    void slotResetToDefaults() override;

  private:
    struct Option {
        QString control;
        QCheckBox* pCheckBox;
    };

    void addOption(const QString& control, const QString& title, const QString& description);

    std::vector<Option> m_options;
    QSpinBox* m_pSectionHotcues;
};
