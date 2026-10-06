#pragma once
#include <vector>
// Absolute device coordinates after homing; both endpoints are acquired.
inline std::vector<int> lineScanPositions(int stepsPerMm = 34700) {
    std::vector<int> points;
    for (int mm = 0; mm <= 24; mm += 2) points.push_back(mm * stepsPerMm);
    return points;
}
inline bool lineScanWithinLimits(const std::vector<int>& points, int minimum, int maximum) {
    if (points.empty() || minimum >= maximum) return false;
    for (int point : points) if (point < minimum || point > maximum) return false;
    return true;
}
