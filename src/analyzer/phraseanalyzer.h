#pragma once

#include <array>
#include <vector>

#include "track/beats.h"

namespace mixxx {

/// Energy of the frequency bands of a track.
struct BandEnergies {
    static constexpr int kNumSpectrumBands = 12;
    static constexpr double kSpectrumMinHz = 30;
    static constexpr double kSpectrumMaxHz = 11000;

    /// Number of values per second
    double framesPerSecond = 0;
    /// Mean power per frame below 150 Hz, between 150 Hz and 2 kHz and
    /// above 2 kHz
    std::vector<float> low;
    std::vector<float> mid;
    std::vector<float> high;
    /// Power per frame of logarithmically spaced bands between
    /// kSpectrumMinHz and kSpectrumMaxHz, used to measure how much the sound
    /// varies from bar to bar. May be empty.
    std::vector<std::array<float, kNumSpectrumBands>> spectrum;
};

/// Splits a track into sections (intro, buildup, drop, breakdown, outro)
/// based on the energy of the frequency bands in each bar. Tuned for
/// electronic dance music.
///
/// The section boundaries are aligned to every 4th bar from the anchor of
/// the bar info, and sections of 8, 16, 32, ... bars are preferred. A
/// section is labeled as a drop if it has a high energy and a strong bass,
/// the other sections are labeled by their position relative to the drops.
class PhraseAnalyzer {
  public:
    static std::vector<Phrase> analyze(const Beats& beats,
            const BarInfo& barInfo,
            const BandEnergies& energies);
};

} // namespace mixxx
