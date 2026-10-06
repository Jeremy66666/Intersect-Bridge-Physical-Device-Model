#include "RamanAutofocus.h"
#include <iostream>
#include <cstdlib> // for rand() and RAND_MAX
#include <ctime>   // for seeding rand()

RamanAutofocus::RamanAutofocus() : currentZPosition(0), currentMaxPeak(0), maxPeakZPosition(0), maxReadingAtCurrentZ(0), rangeStart(0), rangeEnd(0) {
    std::srand(std::time(nullptr)); // Seed random number generator
}

bool RamanAutofocus::move(double deltaZ) {
    double newPosition = currentZPosition + deltaZ;
    if (newPosition < 0) {
        std::cout << "Cannot move higher than Z = 0 cm" << std::endl;
        return false;
    }
    currentZPosition = newPosition;
    std::cout << "Moved motor to Z = " << currentZPosition << " cm" << std::endl;
    return true;
}

double RamanAutofocus::getReading() {
    return static_cast<double>(std::rand()) / RAND_MAX;
}

void RamanAutofocus::initialize(int start, int end) {
    rangeStart = start;
    rangeEnd = end;
    std::cout << "Initializing with range: " << rangeStart << " - " << rangeEnd << " nm" << std::endl;
    while (move(-0.1));
    currentZPosition = 0;
    currentMaxPeak = getReading();
    maxPeakZPosition = currentZPosition;
    std::cout << "Initial reading at Z = 0: " << currentMaxPeak << std::endl;
}

void RamanAutofocus::focusInitial() {
    std::cout << "Starting initial focus sweep..." << std::endl;
    double highestPeak = currentMaxPeak;
    double highestPeakZ = currentZPosition;
    while (move(-0.1)) {
        double reading = getReading();
        if (reading > highestPeak) {
            highestPeak = reading;
            highestPeakZ = currentZPosition;
        }
    }
    while (currentZPosition < highestPeakZ) {
        move(0.1);
    }
    currentMaxPeak = highestPeak;
    maxPeakZPosition = highestPeakZ;
    std::cout << "Focused at Z = " << maxPeakZPosition << " with peak value: " << currentMaxPeak << std::endl;
}

void RamanAutofocus::focus(int mode) {
    if (mode == 1) {
        focusInitial();
    }
    else if (mode == 2) {
        std::cout << "Focus in loop mode is TODO" << std::endl;
    }
    else {
        std::cout << "Invalid focus mode selected." << std::endl;
    }
}

// Getters
double RamanAutofocus::getCurrentZPosition() const { return currentZPosition; }
double RamanAutofocus::getCurrentMaxPeak() const { return currentMaxPeak; }
double RamanAutofocus::getMaxPeakZPosition() const { return maxPeakZPosition; }
double RamanAutofocus::getMaxReadingAtCurrentZ() const { return maxReadingAtCurrentZ; }
int RamanAutofocus::getRangeStart() const { return rangeStart; }
int RamanAutofocus::getRangeEnd() const { return rangeEnd; }

// Setter
void RamanAutofocus::setRange(int start, int end) { rangeStart = start; rangeEnd = end; }

//int main() {
//    RamanAutofocus spectroscope;
//    spectroscope.initialize(600, 620);
//    spectroscope.focus(1);
//    return 0;
//}
