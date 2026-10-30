#include "analyzer/phraseanalyzer.h"

#include <QStringList>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>

#include "util/logger.h"

namespace mixxx {

namespace {

const Logger kLogger("PhraseAnalyzer");

constexpr int kNumBands = 3;
constexpr int kLow = 0;
constexpr int kHigh = 2;
/// Weight of the bands for the segmentation: changes of the bass (kick and
/// bassline) are most important to find the sections of dance music.
constexpr std::array<double, kNumBands> kBandWeights{1.5, 1.0, 0.75};

/// The variation of the spectrum from bar to bar is used as additional
/// feature for the segmentation, relative to its median over the track.
/// It is low in loops, e.g. in the DJ friendly intros and outros.
constexpr int kNumSegmentFeatures = kNumBands + 1;
constexpr double kVariationWeight = 1.0;
constexpr double kMaxRelativeVariation = 3.0;
/// Sections at the beginning and the end of the track with less variation
/// than this fraction of the drops are the intro and outro.
constexpr double kLoopMaxVariationRatio = 0.5;

/// Section boundaries are only allowed at every 4th bar
constexpr int kBarsPerBlock = 4;
constexpr int kMaxSectionBars = 64;

/// The band energies are normalized in units of this many dB below the
/// loudest bars of the track.
constexpr double kDbPerUnit = 6.0;
constexpr double kMinUnits = -5.0;
constexpr double kMaxUnits = 0.5;
constexpr double kLoudestBarsPercentile = 0.95;

/// Cost of each section in the segmentation, in squared units. Higher values
/// result in fewer sections.
constexpr double kSectionPenalty = 4.0;
/// Additional costs for sections that start within a phrase of 8 bars and for
/// sections of 4 bars (except the first and the last section of the track).
/// Dance music is built of phrases, so a change within a phrase is usually
/// only a fill or a fade, e.g. the drums dropping out in the last bars before
/// a breakdown.
constexpr double kOffPhrasePenalty = 6.0;
constexpr double kShortSectionPenalty = 2.0;

/// Sections above this relative energy with a strong bass are drops
constexpr double kDropMinEnergy = 0.65;
/// The bass of a drop is at most this many units below the strongest bass.
/// Buildups with a kick roll and intros with a kick, but without bassline,
/// are typically 6 dB or more below.
constexpr double kDropBassTolerance = 0.6;
/// Drops are only labeled if the track has a breakdown, i.e. a section with
/// a bass this many units below the strongest bass between two sections with
/// a strong bass. Otherwise the main part of the track is labeled Unknown.
constexpr double kBreakdownMinBassDrop = 1.0;
/// Without breakdown, the sections at the beginning and the end with this
/// many units less energy than the loudest section are the intro and outro.
constexpr double kIntroOutroEnergyDrop = 0.5;
/// If the energy of the tracks varies less, all sections are labeled Unknown
constexpr double kMinEnergyRange = 0.5;
/// A section in front of a drop is a buildup if its energy rises by this
/// many units, or if it is short.
constexpr double kBuildupMinRise = 0.5;
constexpr int kBuildupMaxBars = 8;
/// The last bars of a longer section in front of a drop are split off as a
/// buildup, if their energy rises.
constexpr int kBuildupSplitBars = 8;
constexpr int kMinBarsForBuildupSplit = 16;
/// A section with drums and bass between a breakdown and a drop is a buildup,
/// if the high band of the drop is this many units stronger ("pre-drop").
constexpr double kPreDropMinHighRise = 0.75;
constexpr int kPreDropMaxBars = 16;
/// The bass level of a section is 0 if its bass is this many units (15 dB)
/// below the strongest bass of the track.
constexpr double kBassLevelRange = 2.5;

/// The phrase start is searched where the mean features of this many bars
/// change most
constexpr int kPhraseChangeBars = 2;
/// The phrases start at another bar than the first downbeat, if the sound
/// changes this much more at that bar of the phrases
constexpr double kMinPhraseShiftRatio = 1.25;

using BarFeatures = std::array<double, kNumBands>;
using SegmentFeatures = std::array<double, kNumSegmentFeatures>;

double toDb(double power) {
    return 10.0 * std::log10(power + 1e-10);
}

double percentile(std::vector<double> values, double fraction) {
    const auto n = static_cast<std::ptrdiff_t>((values.size() - 1) * fraction);
    std::nth_element(values.begin(), values.begin() + n, values.end());
    return values[n];
}

double meanEnergy(const BarFeatures& features) {
    return (features[0] + features[1] + features[2]) / kNumBands;
}

struct Section {
    int firstBar;
    int endBar;
    double energy;
    double bass;
    double high;
    /// Rise of the high band from the first to the last quarter
    double rise;
    /// Mean variation from bar to bar, relative to the median of the track
    double variation;
    double relativeEnergy;
    bool isDrop;
    bool isPreDrop;
};

Section makeSection(const std::vector<BarFeatures>& normalized,
        const std::vector<double>& variation,
        int firstBar,
        int endBar) {
    Section section{};
    section.firstBar = firstBar;
    section.endBar = endBar;
    const int length = endBar - firstBar;
    const int quarter = std::max(1, length / 4);
    double firstQuarter = 0;
    double lastQuarter = 0;
    for (int bar = firstBar; bar < endBar; ++bar) {
        section.energy += meanEnergy(normalized[bar]) / length;
        section.bass += normalized[bar][kLow] / length;
        section.high += normalized[bar][kHigh] / length;
        section.variation += variation[bar] / length;
        if (bar < firstBar + quarter) {
            firstQuarter += normalized[bar][kHigh] / quarter;
        }
        if (bar >= endBar - quarter) {
            lastQuarter += normalized[bar][kHigh] / quarter;
        }
    }
    section.rise = lastQuarter - firstQuarter;
    return section;
}

/// Splits the bars at the given candidate boundaries (bar indices,
/// including 0 and the number of bars) into sections of minimal cost.
/// The phrases start at the bars `phraseBar + n * kDefaultBarsPerPhrase`.
std::vector<int> segment(const std::vector<SegmentFeatures>& weighted,
        const std::vector<int>& boundaries,
        int phraseBar) {
    const auto numBars = static_cast<int>(weighted.size());
    // Prefix sums for the sum of squared deviations from the section mean
    std::vector<SegmentFeatures> sum(numBars + 1, SegmentFeatures{});
    std::vector<double> sumOfSquares(numBars + 1, 0.0);
    for (int bar = 0; bar < numBars; ++bar) {
        sumOfSquares[bar + 1] = sumOfSquares[bar];
        for (int feature = 0; feature < kNumSegmentFeatures; ++feature) {
            sum[bar + 1][feature] = sum[bar][feature] + weighted[bar][feature];
            sumOfSquares[bar + 1] += weighted[bar][feature] * weighted[bar][feature];
        }
    }
    const auto sectionCost = [&](int firstBar, int endBar, bool isFirstOrLast) {
        const int length = endBar - firstBar;
        double cost = sumOfSquares[endBar] - sumOfSquares[firstBar] + kSectionPenalty;
        for (int feature = 0; feature < kNumSegmentFeatures; ++feature) {
            const double featureSum = sum[endBar][feature] - sum[firstBar][feature];
            cost -= featureSum * featureSum / length;
        }
        if (firstBar > 0 &&
                BarInfo::positiveModulo(firstBar - phraseBar, kDefaultBarsPerPhrase) != 0) {
            cost += kOffPhrasePenalty;
        }
        if (!isFirstOrLast && length < 2 * kBarsPerBlock) {
            cost += kShortSectionPenalty;
        }
        return cost;
    };

    const auto numBoundaries = static_cast<int>(boundaries.size());
    std::vector<double> bestCost(numBoundaries, std::numeric_limits<double>::infinity());
    std::vector<int> previous(numBoundaries, 0);
    bestCost[0] = 0;
    for (int end = 1; end < numBoundaries; ++end) {
        for (int start = end - 1; start >= 0; --start) {
            if (boundaries[end] - boundaries[start] > kMaxSectionBars) {
                break;
            }
            const double cost = bestCost[start] +
                    sectionCost(boundaries[start],
                            boundaries[end],
                            start == 0 || end == numBoundaries - 1);
            if (cost < bestCost[end]) {
                bestCost[end] = cost;
                previous[end] = start;
            }
        }
    }

    std::vector<int> sectionStarts;
    for (int end = numBoundaries - 1; end > 0; end = previous[end]) {
        sectionStarts.push_back(boundaries[previous[end]]);
    }
    std::reverse(sectionStarts.begin(), sectionStarts.end());
    return sectionStarts;
}

PhraseKind labelSection(const std::vector<Section>& sections, int index, int firstDrop, int lastDrop) {
    const Section& section = sections[index];
    if (section.isDrop) {
        return PhraseKind::Drop;
    }
    if (section.isPreDrop) {
        return PhraseKind::Buildup;
    }
    if (index > lastDrop) {
        return PhraseKind::Outro;
    }
    const bool beforeDrop = sections[index + 1].isDrop;
    const bool isBuildup = beforeDrop &&
            (section.rise >= kBuildupMinRise ||
                    section.endBar - section.firstBar <= kBuildupMaxBars);
    if (index < firstDrop) {
        return isBuildup && index > 0 ? PhraseKind::Buildup : PhraseKind::Intro;
    }
    return isBuildup ? PhraseKind::Buildup : PhraseKind::Breakdown;
}

} // namespace

// static
RgbColor PhraseAnalyzer::sectionColor(PhraseKind kind) {
    switch (kind) {
    case PhraseKind::Intro:
        return RgbColor(0x3C78FF);
    case PhraseKind::Buildup:
        return RgbColor(0xFFD200);
    case PhraseKind::Drop:
        return RgbColor(0xFF2828);
    case PhraseKind::Breakdown:
        return RgbColor(0x28C878);
    case PhraseKind::Outro:
        return RgbColor(0xA05AFF);
    case PhraseKind::Unknown:
        break;
    }
    return RgbColor(0x969696);
}

// static
QString PhraseAnalyzer::sectionName(PhraseKind kind) {
    return phraseKindName(kind);
}

namespace {

struct BarData {
    /// Beat index of the first bar
    int firstBarBeat;
    std::vector<BarFeatures> normalized;
    std::vector<SegmentFeatures> weighted;
    std::vector<double> variation;
};

/// The features of all bars of the track, from the first downbeat at or
/// after the start of the track
std::optional<BarData> analyzeBars(const Beats& beats,
        const BarInfo& barInfo,
        const BandEnergies& energies) {
    const int beatsPerBar = barInfo.beatsPerBar;
    const auto numFrames = static_cast<int>(energies.low.size());
    if (energies.framesPerSecond <= 0 || numFrames == 0 ||
            energies.mid.size() != energies.low.size() ||
            energies.high.size() != energies.low.size()) {
        return std::nullopt;
    }
    const double framesPerBeatPosition =
            energies.framesPerSecond / beats.getSampleRate().value();

    const auto firstBeatIt = beats.iteratorFrom(audio::kStartFramePos);
    if (firstBeatIt == beats.cend()) {
        return std::nullopt;
    }
    const int firstBeat = beats.beatIndex(firstBeatIt);
    const int firstBarBeat = firstBeat +
            BarInfo::positiveModulo(barInfo.anchorBeat - firstBeat, beatsPerBar);

    const bool hasSpectrum = static_cast<int>(energies.spectrum.size()) == numFrames;

    // Band energies in dB per bar
    std::vector<BarFeatures> barDb;
    std::vector<std::array<double, BandEnergies::kNumSpectrumBands>> barSpectrumDb;
    for (int barBeat = firstBarBeat;; barBeat += beatsPerBar) {
        auto startIt = beats.cfirstmarker() + barBeat;
        const auto endIt = startIt + beatsPerBar;
        const auto startFrame = static_cast<int>(
                std::round(startIt->value() * framesPerBeatPosition));
        const auto endFrame = static_cast<int>(
                std::round(endIt->value() * framesPerBeatPosition));
        if (endFrame > numFrames || endFrame <= startFrame) {
            break;
        }
        BarFeatures power{};
        for (int frame = startFrame; frame < endFrame; ++frame) {
            power[0] += energies.low[frame];
            power[1] += energies.mid[frame];
            power[2] += energies.high[frame];
        }
        BarFeatures db;
        for (int band = 0; band < kNumBands; ++band) {
            db[band] = toDb(power[band] / (endFrame - startFrame));
        }
        barDb.push_back(db);
        if (hasSpectrum) {
            std::array<double, BandEnergies::kNumSpectrumBands> spectrum{};
            for (int frame = startFrame; frame < endFrame; ++frame) {
                for (int band = 0; band < BandEnergies::kNumSpectrumBands; ++band) {
                    spectrum[band] += energies.spectrum[frame][band];
                }
            }
            for (auto& value : spectrum) {
                value = toDb(value / (endFrame - startFrame));
            }
            barSpectrumDb.push_back(spectrum);
        }
    }
    const auto numBars = static_cast<int>(barDb.size());
    if (numBars < 4 * kBarsPerBlock) {
        return std::nullopt;
    }

    // Variation of the spectrum from bar to bar, relative to the median
    std::vector<double> variation(numBars, 1.0);
    if (hasSpectrum) {
        for (int bar = 1; bar < numBars; ++bar) {
            double sumOfSquares = 0;
            for (int band = 0; band < BandEnergies::kNumSpectrumBands; ++band) {
                const double diff = barSpectrumDb[bar][band] - barSpectrumDb[bar - 1][band];
                sumOfSquares += diff * diff;
            }
            variation[bar] = std::sqrt(sumOfSquares);
        }
        variation[0] = variation[1];
        const double median = percentile(variation, 0.5);
        for (auto& value : variation) {
            value = median > 0 ? std::min(value / median, kMaxRelativeVariation) : 1.0;
        }
    }

    // Normalize relative to the loudest bars of the track
    std::vector<BarFeatures> normalized(numBars);
    std::vector<SegmentFeatures> weighted(numBars);
    for (int bar = 0; bar < numBars; ++bar) {
        weighted[bar][kNumBands] = variation[bar] * kVariationWeight;
    }
    for (int band = 0; band < kNumBands; ++band) {
        std::vector<double> values(numBars);
        for (int bar = 0; bar < numBars; ++bar) {
            values[bar] = barDb[bar][band];
        }
        const double loudest = percentile(values, kLoudestBarsPercentile);
        for (int bar = 0; bar < numBars; ++bar) {
            normalized[bar][band] = std::clamp(
                    (barDb[bar][band] - loudest) / kDbPerUnit, kMinUnits, kMaxUnits);
            weighted[bar][band] = normalized[bar][band] * kBandWeights[band];
        }
    }
    return BarData{firstBarBeat, std::move(normalized), std::move(weighted), std::move(variation)};
}

} // namespace

// static
int PhraseAnalyzer::findPhraseAnchor(const Beats& beats,
        const BarInfo& barInfo,
        const BandEnergies& energies) {
    const std::optional<BarData> bars = analyzeBars(beats, barInfo, energies);
    if (!bars) {
        return barInfo.anchorBeat;
    }
    const auto& weighted = bars->weighted;
    const auto numBars = static_cast<int>(weighted.size());

    // How much the sound changes at the start of each bar, compared to the
    // bars before. Several bars are compared, so that fills of a single bar
    // count less than the changes between the sections.
    std::vector<double> change(numBars, 0.0);
    for (int bar = kPhraseChangeBars; bar + kPhraseChangeBars <= numBars; ++bar) {
        for (int band = 0; band < kNumBands; ++band) {
            double before = 0;
            double after = 0;
            for (int i = 0; i < kPhraseChangeBars; ++i) {
                before += weighted[bar - 1 - i][band];
                after += weighted[bar + i][band];
            }
            change[bar] += std::abs(after - before) / kPhraseChangeBars;
        }
    }

    // The sections of dance music change at the phrase starts
    const int anchorBar = (barInfo.anchorBeat - bars->firstBarBeat) / barInfo.beatsPerBar;
    std::array<double, kDefaultBarsPerPhrase> scores{};
    for (int bar = 0; bar < numBars; ++bar) {
        scores[BarInfo::positiveModulo(bar - anchorBar, kDefaultBarsPerPhrase)] += change[bar];
    }
    const auto best = std::max_element(scores.cbegin(), scores.cend());
    const auto shiftBars = static_cast<int>(best - scores.cbegin());
    QStringList scoreTexts;
    for (const double score : scores) {
        scoreTexts.append(QString::number(score, 'f', 1));
    }
    kLogger.info() << "Phrase start scores per bar:" << scoreTexts.join(QStringLiteral(" "));
    // Keep the first downbeat as phrase start, unless the sections clearly
    // change elsewhere, e.g. after a pickup bar at the start of the track
    if (*best < kMinPhraseShiftRatio * scores[0]) {
        return barInfo.anchorBeat;
    }
    return barInfo.anchorBeat + shiftBars * barInfo.beatsPerBar;
}

// static
std::vector<Phrase> PhraseAnalyzer::analyze(const Beats& beats,
        const BarInfo& barInfo,
        const BandEnergies& energies) {
    const std::optional<BarData> bars = analyzeBars(beats, barInfo, energies);
    if (!bars) {
        return {};
    }
    const int beatsPerBar = barInfo.beatsPerBar;
    const int firstBarBeat = bars->firstBarBeat;
    const auto& normalized = bars->normalized;
    const auto& weighted = bars->weighted;
    const auto& variation = bars->variation;
    const auto numBars = static_cast<int>(weighted.size());

    // Candidate boundaries at every 4th bar from the anchor
    std::vector<int> boundaries{0};
    const int anchorBar = (barInfo.anchorBeat - firstBarBeat) / beatsPerBar;
    for (int bar = BarInfo::positiveModulo(anchorBar, kBarsPerBlock); bar < numBars;
            bar += kBarsPerBlock) {
        if (bar > 0) {
            boundaries.push_back(bar);
        }
    }
    if (boundaries.back() != numBars) {
        boundaries.push_back(numBars);
    }
    const std::vector<int> sectionStarts = segment(weighted, boundaries, anchorBar);

    std::vector<Section> segmented;
    for (std::size_t i = 0; i < sectionStarts.size(); ++i) {
        segmented.push_back(makeSection(normalized,
                variation,
                sectionStarts[i],
                i + 1 < sectionStarts.size() ? sectionStarts[i + 1] : numBars));
    }

    // Find the drops
    const auto relativeEnergy = [](const std::vector<Section>& sections) {
        const auto [minIt, maxIt] = std::minmax_element(sections.cbegin(),
                sections.cend(),
                [](const Section& lhs, const Section& rhs) {
                    return lhs.energy < rhs.energy;
                });
        return std::make_pair(minIt->energy, maxIt->energy - minIt->energy);
    };
    const auto [minEnergy, energyRange] = relativeEnergy(segmented);
    double strongestBass = -std::numeric_limits<double>::infinity();
    for (const auto& section : segmented) {
        strongestBass = std::max(strongestBass, section.bass);
    }
    const auto hasStrongBass = [&](const Section& section) {
        return section.bass >= strongestBass - kDropBassTolerance;
    };
    bool hasBreakdown = false;
    for (std::size_t i = 1; i + 1 < segmented.size(); ++i) {
        if (segmented[i].bass < strongestBass - kBreakdownMinBassDrop &&
                std::any_of(segmented.cbegin(), segmented.cbegin() + i, hasStrongBass) &&
                std::any_of(segmented.cbegin() + i + 1, segmented.cend(), hasStrongBass)) {
            hasBreakdown = true;
            break;
        }
    }
    for (auto& section : segmented) {
        const double energy = energyRange > 0 ? (section.energy - minEnergy) / energyRange : 1;
        section.isDrop = hasBreakdown && energyRange >= kMinEnergyRange &&
                energy >= kDropMinEnergy && hasStrongBass(section);
    }

    // Split off buildups from longer sections in front of drops
    std::vector<Section> sections;
    for (std::size_t i = 0; i < segmented.size(); ++i) {
        const Section& section = segmented[i];
        if (!section.isDrop && i + 1 < segmented.size() && segmented[i + 1].isDrop &&
                section.endBar - section.firstBar >= kMinBarsForBuildupSplit) {
            const int splitBar = section.endBar - kBuildupSplitBars;
            const Section buildup = makeSection(normalized, variation, splitBar, section.endBar);
            if (buildup.rise >= kBuildupMinRise) {
                sections.push_back(makeSection(
                        normalized, variation, section.firstBar, splitBar));
                sections.push_back(buildup);
                continue;
            }
        }
        sections.push_back(section);
    }

    // Pre-drops with drums and bass, but less highs than the following drop
    for (std::size_t i = 1; i + 1 < sections.size(); ++i) {
        Section& section = sections[i];
        if (section.isDrop && !sections[i - 1].isDrop && sections[i + 1].isDrop &&
                section.endBar - section.firstBar <= kPreDropMaxBars &&
                sections[i + 1].high - section.high >= kPreDropMinHighRise) {
            section.isDrop = false;
            section.isPreDrop = true;
        }
    }

    int firstDrop = -1;
    int lastDrop = -1;
    const auto [finalMinEnergy, finalEnergyRange] = relativeEnergy(sections);
    for (int i = 0; i < static_cast<int>(sections.size()); ++i) {
        Section& section = sections[i];
        section.relativeEnergy = finalEnergyRange > 0
                ? (section.energy - finalMinEnergy) / finalEnergyRange
                : 1;
        if (section.isDrop) {
            if (firstDrop < 0) {
                firstDrop = i;
            }
            lastDrop = i;
        }
    }

    // Without drops, only the quieter sections at the beginning and the end
    // are labeled
    const double maxEnergy = finalMinEnergy + finalEnergyRange;
    const auto isQuiet = [&](const Section& section) {
        return section.energy < maxEnergy - kIntroOutroEnergyDrop;
    };
    const auto numSections = static_cast<int>(sections.size());
    int introEnd = 0;
    while (introEnd < numSections && isQuiet(sections[introEnd])) {
        introEnd++;
    }
    int outroStart = numSections;
    while (outroStart > introEnd && isQuiet(sections[outroStart - 1])) {
        outroStart--;
    }

    std::vector<PhraseKind> kinds(numSections);
    for (int i = 0; i < numSections; ++i) {
        if (firstDrop >= 0) {
            kinds[i] = labelSection(sections, i, firstDrop, lastDrop);
        } else if (i < introEnd) {
            kinds[i] = PhraseKind::Intro;
        } else if (i >= outroStart) {
            kinds[i] = PhraseKind::Outro;
        } else {
            kinds[i] = PhraseKind::Unknown;
        }
    }

    // Loops at the beginning and the end are the intro and outro, even with
    // the full energy of a drop
    double referenceVariation = 1.0; // the median of the track
    double dropVariation = 0;
    int dropBars = 0;
    for (const auto& section : sections) {
        if (section.isDrop) {
            dropVariation += section.variation * (section.endBar - section.firstBar);
            dropBars += section.endBar - section.firstBar;
        }
    }
    if (dropBars > 0) {
        referenceVariation = dropVariation / dropBars;
    }
    const auto isLoop = [&](int i) {
        return (kinds[i] == PhraseKind::Drop || kinds[i] == PhraseKind::Unknown) &&
                sections[i].variation < kLoopMaxVariationRatio * referenceVariation;
    };
    for (int i = 0; i < numSections && (kinds[i] == PhraseKind::Intro || isLoop(i)); ++i) {
        kinds[i] = PhraseKind::Intro;
    }
    for (int i = numSections - 1; i >= 0 && (kinds[i] == PhraseKind::Outro || isLoop(i)); --i) {
        kinds[i] = PhraseKind::Outro;
    }

    // Consecutive sections of the same kind are merged
    std::vector<Phrase> phrases;
    int mergedBars = 0;
    for (int i = 0; i < numSections; ++i) {
        const Section& section = sections[i];
        const int length = section.endBar - section.firstBar;
        const auto bassLevel = static_cast<float>(std::clamp(
                1.0 + (section.bass - strongestBass) / kBassLevelRange, 0.0, 1.0));
        if (i > 0 && kinds[i] == kinds[i - 1]) {
            // Length weighted means
            Phrase& merged = phrases.back();
            const auto mergedMean = [&](float mergedValue, double value) {
                return static_cast<float>(
                        (mergedValue * mergedBars + value * length) / (mergedBars + length));
            };
            merged.energy = mergedMean(merged.energy, section.relativeEnergy);
            merged.bass = mergedMean(merged.bass, bassLevel);
            mergedBars += length;
        } else {
            phrases.push_back(Phrase{firstBarBeat + section.firstBar * beatsPerBar,
                    kinds[i],
                    static_cast<float>(section.relativeEnergy),
                    bassLevel});
            mergedBars = length;
        }
        kLogger.debug() << "Section at bar" << section.firstBar << "length"
                        << section.endBar - section.firstBar << "energy"
                        << section.energy << "bass" << section.bass << "rise"
                        << section.rise << "variation" << section.variation
                        << "kind" << static_cast<int>(kinds[i]);
    }
    return phrases;
}

} // namespace mixxx
