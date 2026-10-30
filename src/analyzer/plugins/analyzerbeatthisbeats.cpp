#include "analyzer/plugins/analyzerbeatthisbeats.h"

#include <algorithm>
#include <cmath>
#include <optional>

#include "analyzer/neuralanalysis.h"
#include "analyzer/plugins/downbeatdetector.h"

namespace mixxx {

namespace {

/// A beat is a local maximum of the logits within +/- 3 frames (+/- 60 ms)
constexpr int kPeakNeighborFrames = 3;
/// Peaks closer than this part of the median interval are not both beats
constexpr double kMinBeatIntervalRatio = 0.6;

/// The constant tempo is first fitted to the longest run of beats with
/// intervals within this part of the median interval
constexpr double kRunIntervalTolerance = 0.1;
constexpr int kMinRunBeats = 8;
/// The grid is fitted starting with this many of the longest runs
constexpr std::size_t kMaxGridSeeds = 5;
/// Beats further away from the fitted grid are outliers
constexpr double kMaxGridErrorSeconds = 0.04;
constexpr int kMaxGridFitIterations = 20;
/// At least this part of the beats must be on the fitted grid. Tracks with
/// a varying tempo (e.g. played live) have fewer beats on any grid, but the
/// grid of the longest part is still better than that of BeatUtils.
constexpr double kMinGridInlierRatio = 0.15;

struct GridFit {
    double first;
    double interval;
    int inliers;
};

/// Least squares fit of `position = first + interval * index`
class GridRegression {
  public:
    void add(double index, double position) {
        m_sumIndex += index;
        m_sumPosition += position;
        m_sumIndexSquared += index * index;
        m_sumIndexPosition += index * position;
        m_count++;
    }

    std::optional<GridFit> fit() const {
        const double denominator = m_count * m_sumIndexSquared - m_sumIndex * m_sumIndex;
        if (m_count < kMinRunBeats || denominator <= 0) {
            return std::nullopt;
        }
        const double interval =
                (m_count * m_sumIndexPosition - m_sumIndex * m_sumPosition) / denominator;
        const double first = (m_sumPosition - interval * m_sumIndex) / m_count;
        return GridFit{first, interval, m_count};
    }

  private:
    double m_sumIndex = 0;
    double m_sumPosition = 0;
    double m_sumIndexSquared = 0;
    double m_sumIndexPosition = 0;
    int m_count = 0;
};

/// Fits the grid to the beats within `maxError` of the grid `previous`
std::optional<GridFit> refitGrid(const QVector<audio::FramePos>& beats,
        const GridFit& previous,
        double maxError) {
    GridRegression regression;
    for (const auto& beat : beats) {
        const double index = std::round((beat.value() - previous.first) / previous.interval);
        if (std::abs(beat.value() - (previous.first + index * previous.interval)) <= maxError) {
            regression.add(index, beat.value());
        }
    }
    return regression.fit();
}

} // namespace

AnalyzerBeatThisBeats::AnalyzerBeatThisBeats(
        std::shared_ptr<NeuralAnalysis> pNeuralAnalysis, bool fixedTempo)
        : m_pNeuralAnalysis(std::move(pNeuralAnalysis)),
          m_fixedTempo(fixedTempo) {
}

AnalyzerBeatThisBeats::~AnalyzerBeatThisBeats() = default;

bool AnalyzerBeatThisBeats::initialize(mixxx::audio::SampleRate sampleRate) {
    m_sampleRate = sampleRate;
    m_resultBeats.clear();
    return m_pNeuralAnalysis && m_pNeuralAnalysis->request();
}

bool AnalyzerBeatThisBeats::processSamples(const CSAMPLE* pIn, SINT iLen) {
    // The samples are processed by NeuralAnalysis
    Q_UNUSED(pIn);
    Q_UNUSED(iLen);
    return true;
}

bool AnalyzerBeatThisBeats::finalize() {
    const DownbeatDetector* pDetector = m_pNeuralAnalysis->results();
    if (!pDetector) {
        return false;
    }
    m_resultBeats = beatsFromLogits(
            pDetector->beatLogits(), DownbeatDetector::kFramesPerSecond, m_sampleRate);
    if (m_fixedTempo) {
        // Otherwise BeatUtils makes the constant tempo from the beats
        const auto grid = constantTempoBeats(m_resultBeats, m_sampleRate);
        if (!grid.isEmpty()) {
            m_resultBeats = grid;
        }
    }
    return !m_resultBeats.isEmpty();
}

// static
QVector<mixxx::audio::FramePos> AnalyzerBeatThisBeats::beatsFromLogits(
        const std::vector<float>& beatLogits,
        double framesPerSecond,
        mixxx::audio::SampleRate sampleRate) {
    const auto numFrames = static_cast<int>(beatLogits.size());
    struct Peak {
        /// Position in frames of the logits
        double position;
        float logit;
    };
    std::vector<Peak> peaks;
    for (int i = 0; i < numFrames; ++i) {
        const float value = beatLogits[i];
        if (value <= 0) {
            continue;
        }
        const int first = std::max(0, i - kPeakNeighborFrames);
        const int last = std::min(numFrames - 1, i + kPeakNeighborFrames);
        if (*std::max_element(beatLogits.cbegin() + first, beatLogits.cbegin() + last + 1) >
                value) {
            continue;
        }
        double position = i;
        if (i > 0 && i + 1 < numFrames) {
            // Parabolic interpolation of the maximum
            const double left = beatLogits[i - 1];
            const double right = beatLogits[i + 1];
            const double curvature = left - 2 * value + right;
            if (curvature < 0) {
                position += std::clamp(0.5 * (left - right) / curvature, -0.5, 0.5);
            }
        }
        // Peaks in adjacent frames with the same value are one beat
        if (!peaks.empty() && position - peaks.back().position <= 1.0) {
            peaks.back().position = (peaks.back().position + position) / 2;
            continue;
        }
        peaks.push_back(Peak{position, value});
    }

    // Drum fills can have additional peaks between the beats, which prevent
    // finding the regions of constant tempo. Of peaks closer than a beat,
    // only the strongest one is kept.
    if (peaks.size() > 2) {
        std::vector<double> intervals;
        intervals.reserve(peaks.size() - 1);
        for (std::size_t i = 1; i < peaks.size(); ++i) {
            intervals.push_back(peaks[i].position - peaks[i - 1].position);
        }
        const auto median = intervals.begin() + intervals.size() / 2;
        std::nth_element(intervals.begin(), median, intervals.end());
        const double minInterval = kMinBeatIntervalRatio * *median;
        std::vector<Peak> beatPeaks;
        for (const Peak& peak : peaks) {
            if (!beatPeaks.empty() && peak.position - beatPeaks.back().position < minInterval) {
                if (peak.logit > beatPeaks.back().logit) {
                    beatPeaks.back() = peak;
                }
                continue;
            }
            beatPeaks.push_back(peak);
        }
        peaks = std::move(beatPeaks);
    }

    QVector<mixxx::audio::FramePos> beats;
    beats.reserve(static_cast<int>(peaks.size()));
    for (const Peak& peak : peaks) {
        beats.append(mixxx::audio::FramePos(
                peak.position / framesPerSecond * sampleRate.value()));
    }
    return beats;
}

// static
QVector<mixxx::audio::FramePos> AnalyzerBeatThisBeats::constantTempoBeats(
        const QVector<mixxx::audio::FramePos>& beats,
        mixxx::audio::SampleRate sampleRate) {
    if (beats.size() < kMinRunBeats) {
        return {};
    }
    std::vector<double> intervals;
    intervals.reserve(beats.size() - 1);
    for (int i = 1; i < beats.size(); ++i) {
        intervals.push_back(beats[i] - beats[i - 1]);
    }
    std::vector<double> sorted = intervals;
    const auto median = sorted.begin() + sorted.size() / 2;
    std::nth_element(sorted.begin(), median, sorted.end());
    const double medianInterval = *median;

    // Runs of regular beats (first beat, number of beats), which are not
    // disturbed by breaks without beats or half-time parts
    std::vector<std::pair<int, int>> runs;
    int runStart = 0;
    for (int i = 0; i <= static_cast<int>(intervals.size()); ++i) {
        if (i == static_cast<int>(intervals.size()) ||
                std::abs(intervals[i] - medianInterval) >
                        kRunIntervalTolerance * medianInterval) {
            if (i + 1 - runStart >= kMinRunBeats) {
                runs.emplace_back(runStart, i + 1 - runStart);
            }
            runStart = i + 1;
        }
    }
    std::sort(runs.begin(), runs.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second > rhs.second;
    });
    if (runs.size() > kMaxGridSeeds) {
        runs.resize(kMaxGridSeeds);
    }

    // The grid that most beats of the track are on, starting with the tempo
    // of each of the longest runs
    const double maxError = kMaxGridErrorSeconds * sampleRate.value();
    std::optional<GridFit> fit;
    for (const auto& [start, length] : runs) {
        // The beats of a run are consecutive
        GridRegression regression;
        for (int i = 0; i < length; ++i) {
            regression.add(i, beats[start + i].value());
        }
        std::optional<GridFit> runFit = regression.fit();
        // Extend the grid to the whole track
        for (int i = 0; runFit && i < kMaxGridFitIterations; ++i) {
            const std::optional<GridFit> extended = refitGrid(beats, *runFit, maxError);
            if (!extended || extended->inliers <= runFit->inliers) {
                break;
            }
            runFit = extended;
        }
        if (runFit && (!fit || runFit->inliers > fit->inliers)) {
            fit = runFit;
        }
    }
    if (!fit || fit->inliers < kMinGridInlierRatio * beats.size()) {
        return {};
    }

    const auto firstIndex = static_cast<int>(
            std::round((beats.first().value() - fit->first) / fit->interval));
    const auto lastIndex = static_cast<int>(
            std::round((beats.last().value() - fit->first) / fit->interval));
    QVector<mixxx::audio::FramePos> grid;
    grid.reserve(lastIndex - firstIndex + 1);
    for (int index = firstIndex; index <= lastIndex; ++index) {
        grid.append(mixxx::audio::FramePos(fit->first + index * fit->interval));
    }
    return grid;
}

} // namespace mixxx
