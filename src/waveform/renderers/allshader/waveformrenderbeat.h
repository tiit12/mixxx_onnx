#pragma once

#include <QColor>
#include <QVector4D>
#include <vector>

#include "rendergraph/geometrynode.h"
#include "util/class.h"
#include "waveform/renderers/waveformrendererabstract.h"

class QDomNode;
class SkinContext;

namespace allshader {
class WaveformRenderBeat;
} // namespace allshader

class allshader::WaveformRenderBeat final
        : public QObject,
          public ::WaveformRendererAbstract,
          public rendergraph::GeometryNode {
    Q_OBJECT
  public:
    explicit WaveformRenderBeat(WaveformWidgetRenderer* waveformWidget,
            ::WaveformRendererAbstract::PositionSource type =
                    ::WaveformRendererAbstract::Play);

    // Pure virtual from WaveformRendererAbstract, not used
    void draw(QPainter* painter, QPaintEvent* event) override final;

    void setup(const QDomNode& node, const SkinContext& skinContext) override;

    // Virtuals for rendergraph::Node
    void preprocess() override;

  public slots:
    void setColor(const QColor& color) {
        m_color = color;
    }

  private:
    struct VisibleSection {
        float x1;
        float x2;
        QVector4D color;
    };

    QColor m_color;
    // Downbeats
    QColor m_barColor;
    // Downbeats that start a phrase
    QColor m_phraseColor;
    bool m_isSlipRenderer;
    // Reused to avoid allocations
    std::vector<VisibleSection> m_visibleSections;

    bool preprocessInner();

    DISALLOW_COPY_AND_ASSIGN(WaveformRenderBeat);
};
