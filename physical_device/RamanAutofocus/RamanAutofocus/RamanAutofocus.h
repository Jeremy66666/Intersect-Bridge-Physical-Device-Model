#ifndef RAMAN_SPECTROSCOPE_H
#define RAMAN_SPECTROSCOPE_H

class RamanAutofocus {
private:
    double currentZPosition;     // Current Z position of the laser
    double currentMaxPeak;       // Current max peak value
    double maxPeakZPosition;     // Z position at the current max peak
    double maxReadingAtCurrentZ; // Max reading at current Z position
    int rangeStart, rangeEnd;    // Range in nm

public:
    // Constructor
    RamanAutofocus();

    // Movement and Reading Functions
    bool move(double deltaZ);
    double getReading();

    // Initialization and Focusing Functions
    void initialize(int start, int end);
    void focusInitial();
    void focus(int mode);

    // Getters
    double getCurrentZPosition() const;
    double getCurrentMaxPeak() const;
    double getMaxPeakZPosition() const;
    double getMaxReadingAtCurrentZ() const;
    int getRangeStart() const;
    int getRangeEnd() const;

    // Setter
    void setRange(int start, int end);
};

#endif // RAMAN_SPECTROSCOPE_H