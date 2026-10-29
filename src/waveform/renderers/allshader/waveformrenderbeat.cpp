#include "waveform/renderers/allshader/waveformrenderbeat.h"

#include <QDomNode>
#include <algorithm>
#include <cmath>

#include "analyzer/phraseanalyzer.h"
#include "control/controlobject.h"
#include "engine/engine.h"
#include "moc_waveformrenderbeat.cpp"
#include "rendergraph/geometry.h"
#include "rendergraph/material/rgbamaterial.h"
#include "rendergraph/vertexupdaters/rgbavertexupdater.h"
#include "skin/legacy/skincontext.h"
#include "track/track.h"
#include "waveform/renderers/waveformwidgetrenderer.h"
#include "waveform/waveform.h"
#include "waveform/waveformwidgetfactory.h"
#include "widget/wskincolor.h"

using namespace rendergraph;

namespace {

const QColor kDefaultBarColor = QColor(255, 255, 255);
const QColor kDefaultPhraseColor = QColor(255, 96, 0);

// Height of the colored section band at the bottom of the waveform
constexpr float kSectionBandMinHeight = 4.f;
constexpr float kSectionBandRelativeHeight = 0.05f;
constexpr float kSectionBandAlpha = 0.85f;
constexpr float kSectionStartWidth = 4.f;

// Bar at the top of the waveform if the bass of two decks clashes
const QVector4D kBassClashColor(1.f, 0.f, 0.f, 0.9f);
constexpr float kBassClashRelativeHeight = 0.08f;

QVector4D toVector4D(const QColor& color, float alpha) {
    return QVector4D(color.redF(), color.greenF(), color.blueF(), alpha);
}

QColor sectionColor(mixxx::PhraseKind kind) {
    return mixxx::RgbColor::toQColor(mixxx::PhraseAnalyzer::sectionColor(kind));
}

} // namespace

namespace allshader {

WaveformRenderBeat::WaveformRenderBeat(WaveformWidgetRenderer* waveformWidget,
        ::WaveformRendererAbstract::PositionSource type)
        : ::WaveformRendererAbstract(waveformWidget),
          m_barColor(kDefaultBarColor),
          m_phraseColor(kDefaultPhraseColor),
          m_isSlipRenderer(type == ::WaveformRendererAbstract::Slip) {
    initForRectangles<RGBAMaterial>(0);
    setUsePreprocess(true);
}

void WaveformRenderBeat::setup(const QDomNode& node, const SkinContext& skinContext) {
    m_color = QColor(skinContext.selectString(node, QStringLiteral("BeatColor")));
    m_color = WSkinColor::getCorrectColor(m_color).toRgb();
    const QColor barColor(skinContext.selectString(node, QStringLiteral("BarColor")));
    if (barColor.isValid()) {
        m_barColor = WSkinColor::getCorrectColor(barColor).toRgb();
    }
    const QColor phraseColor(skinContext.selectString(node, QStringLiteral("PhraseColor")));
    if (phraseColor.isValid()) {
        m_phraseColor = WSkinColor::getCorrectColor(phraseColor).toRgb();
    }
}

void WaveformRenderBeat::draw(QPainter* painter, QPaintEvent* event) {
    Q_UNUSED(painter);
    Q_UNUSED(event);
    DEBUG_ASSERT(false);
}

void WaveformRenderBeat::preprocess() {
    if (!preprocessInner()) {
        geometry().allocate(0);
        markDirtyGeometry();
    }
}

bool WaveformRenderBeat::preprocessInner() {
    const TrackPointer trackInfo = m_waveformRenderer->getTrackInfo();

    if (!trackInfo || (m_isSlipRenderer && !m_waveformRenderer->isSlipActive())) {
        return false;
    }

    const bool isStemTrack = trackInfo && trackInfo->hasStem() &&
            trackInfo->getWaveform() && trackInfo->getWaveform()->hasStem();
    const bool splitStemTracks = isStemTrack && WaveformWidgetFactory::isCreated() &&
            WaveformWidgetFactory::instance()->isStemSplitTracks();

    auto positionType = m_isSlipRenderer ? ::WaveformRendererAbstract::Slip
                                         : ::WaveformRendererAbstract::Play;

    mixxx::BeatsPointer trackBeats = trackInfo->getBeats();
    if (!trackBeats) {
        return false;
    }

#ifndef __SCENEGRAPH__
    int alpha = m_waveformRenderer->getBeatGridAlpha();
    if (alpha == 0) {
        return false;
    }
    m_color.setAlphaF(alpha / 100.0f);
#endif

    if (!m_color.alpha()) {
        // Don't render the beatgrid lines is there are fully transparent
        return false;
    }

    const float devicePixelRatio = m_waveformRenderer->getDevicePixelRatio();

    const double trackSamples = m_waveformRenderer->getTrackSamples();
    if (trackSamples <= 0.0) {
        return false;
    }

    const double firstDisplayedPosition =
            m_waveformRenderer->getFirstDisplayedPosition(positionType);
    const double lastDisplayedPosition =
            m_waveformRenderer->getLastDisplayedPosition(positionType);

    const auto startPosition = mixxx::audio::FramePos::fromEngineSamplePos(
            firstDisplayedPosition * trackSamples);
    const auto endPosition = mixxx::audio::FramePos::fromEngineSamplePos(
            lastDisplayedPosition * trackSamples);

    if (!startPosition.isValid() || !endPosition.isValid()) {
        return false;
    }

    const float rendererBreadth = m_waveformRenderer->getBreadth();

    const int numVerticesPerLine = 6; // 2 triangles

    // Count the number of beats in the range to reserve space in the m_vertices vector.
    // Note that we could also use
    //   int numBearsInRange = trackBeats->numBeatsInRange(startPosition, endPosition);
    // for this, but there have been reports of that method failing with a DEBUG_ASSERT.
    int numBeatsInRange = 0;
    for (auto it = trackBeats->iteratorFrom(startPosition);
            it != trackBeats->cend() && *it <= endPosition;
            ++it) {
        numBeatsInRange++;
    }

    const mixxx::BarInfoPointer pBarInfo = trackBeats->getBarInfo();

    // Analyzed sections in the visible range, drawn as a colored band at the
    // bottom of the waveform
    m_visibleSections.clear();
    if (pBarInfo && !m_isSlipRenderer) {
        const auto trackEnd = mixxx::audio::FramePos::fromEngineSamplePos(trackSamples);
        const auto& sections = pBarInfo->phrases;
        for (std::size_t i = 0; i < sections.size(); ++i) {
            const auto sectionStart = *(trackBeats->cfirstmarker() + sections[i].startBeat);
            const auto sectionEnd = i + 1 < sections.size()
                    ? *(trackBeats->cfirstmarker() + sections[i + 1].startBeat)
                    : trackEnd;
            if (sectionEnd < startPosition || sectionStart > endPosition) {
                continue;
            }
            const auto toX = [&](mixxx::audio::FramePos position) {
                const double x = m_waveformRenderer->transformSamplePositionInRendererWorld(
                        position.toEngineSamplePos(), positionType);
                return static_cast<float>(std::clamp(
                        x, -1.0, static_cast<double>(m_waveformRenderer->getLength()) + 1.0));
            };
            m_visibleSections.push_back(VisibleSection{toX(sectionStart),
                    toX(sectionEnd),
                    toVector4D(sectionColor(sections[i].kind), kSectionBandAlpha)});
        }
    }

    if (!m_pBassClash) {
        const ConfigKey bassClashKey(m_waveformRenderer->getGroup(), QStringLiteral("bass_clash"));
        if (ControlObject::exists(bassClashKey)) {
            m_pBassClash = std::make_unique<PollingControlProxy>(bassClashKey);
        }
    }
    const bool bassClash = !m_isSlipRenderer && m_pBassClash && m_pBassClash->toBool();

    const int numBoxesPerBeat = (m_isSlipRenderer && splitStemTracks)
            ? mixxx::kMaxSupportedStems
            : 1;
    const int reserved = (numBeatsInRange * numBoxesPerBeat +
                                 static_cast<int>(m_visibleSections.size()) +
                                 (bassClash ? 1 : 0)) *
            numVerticesPerLine;
    geometry().allocate(reserved);

    RGBAVertexUpdater vertexUpdater{geometry().vertexDataAs<Geometry::RGBAColoredPoint2D>()};

    const float bandHeight = std::max(kSectionBandMinHeight,
            std::round(rendererBreadth * kSectionBandRelativeHeight));
    for (const auto& section : m_visibleSections) {
        vertexUpdater.addRectangle({section.x1, rendererBreadth - bandHeight},
                {section.x2, rendererBreadth},
                section.color);
    }
    if (bassClash) {
        vertexUpdater.addRectangle({0.f, 0.f},
                {static_cast<float>(m_waveformRenderer->getLength()),
                        std::round(rendererBreadth * kBassClashRelativeHeight)},
                kBassClashColor);
    }

    const float boxBreadth = splitStemTracks
            ? rendererBreadth / static_cast<float>(mixxx::kMaxSupportedStems)
            : rendererBreadth;

    const float beatAlpha = static_cast<float>(m_color.alphaF());
    const QVector4D beatColor = toVector4D(m_color, beatAlpha);
    const QVector4D barColor = toVector4D(m_barColor, std::min(1.f, beatAlpha * 1.5f + 0.1f));
    const QVector4D phraseColor = toVector4D(m_phraseColor, std::min(1.f, beatAlpha * 2.f + 0.2f));

    auto it = trackBeats->iteratorFrom(startPosition);
    int beatIndex = pBarInfo ? trackBeats->beatIndex(it) : 0;
    for (; it != trackBeats->cend() && *it <= endPosition; ++it, ++beatIndex) {
        double beatPosition = it->toEngineSamplePos();
        double xBeatPoint =
                m_waveformRenderer->transformSamplePositionInRendererWorld(
                        beatPosition, positionType);

        xBeatPoint = qRound(xBeatPoint * devicePixelRatio) / devicePixelRatio;

        float width = 1.f;
        QVector4D color = beatColor;
        if (pBarInfo) {
            if (pBarInfo->isSectionStart(beatIndex)) {
                width = kSectionStartWidth;
                color = toVector4D(sectionColor(pBarInfo->findPhrase(beatIndex)->kind),
                        phraseColor.w());
            } else if (pBarInfo->isPhraseStart(beatIndex)) {
                width = 3.f;
                color = phraseColor;
            } else if (pBarInfo->isDownbeat(beatIndex)) {
                width = 2.f;
                color = barColor;
            }
        }

        // Center wider lines on the beat
        const float x1 = static_cast<float>(xBeatPoint) - std::floor((width - 1.f) / 2.f);
        const float x2 = x1 + width;

        if (m_isSlipRenderer && splitStemTracks) {
            for (int stemIdx = 0; stemIdx < mixxx::kMaxSupportedStems; ++stemIdx) {
                const float posy1 = stemIdx * boxBreadth;
                const float posy2 = posy1 + boxBreadth / 2.f;
                vertexUpdater.addRectangle({x1, posy1}, {x2, posy2}, color);
            }
        } else {
            vertexUpdater.addRectangle({x1, 0.f},
                    {x2, m_isSlipRenderer ? rendererBreadth / 2 : rendererBreadth},
                    color);
        }
    }
    markDirtyGeometry();

    DEBUG_ASSERT(reserved == vertexUpdater.index());

    markDirtyMaterial();

    return true;
}

} // namespace allshader
