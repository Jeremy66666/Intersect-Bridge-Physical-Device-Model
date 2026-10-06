// RamanAutofocus.cpp
#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "../dlls/Thorlabs.MotionControl.KCube.DCServo.h"

// Global parameters and constants
const double LASER_WAVELENGTH = 785.0;            // Laser wavelength in nm (typical for Raman)
const double TARGET_WAVELENGTH_MIN = 500.0;       // Start of target wavelength range in nm
const double TARGET_WAVELENGTH_MAX = 540.0;       // End of target wavelength range in nm
const double Z_POSITION_MIN = -100.0;             // Minimum Z position in mm
const double Z_POSITION_MAX = 100.0;              // Maximum Z position in mm
const int MOTOR_POSITION_MAX = 700106;            // Maximum motor position in steps
const int MOTOR_SERIAL_NO = 83837825;             // Hardcoded serial number for the motor
const int POLLING_INTERVAL_MS = 100;              // Polling interval in milliseconds
const int INIT_DELAY_MS = 1000;                   // Initialization delay in milliseconds
const int ACQUISITION_DELAY_MS = 200;             // Simulated acquisition delay in milliseconds
const std::vector<std::string> POSSIBLE_DATA_PATHS = {
    "",                                           // Current directory
    "data/",                                      // data subfolder
    "../data/",                                   // data in parent directory
    "../../data/",                                // data in grandparent directory
    "../../../data/",                             // data in great-grandparent directory
    "out/data/",                                  // data in out subfolder
    "../out/data/",                               // data in parent's out subfolder
    "../RamanAutofocus/data/"                     // Specific project path
};
const std::string DATA_FILENAME = "Si_Raman_z_sweep.csv";
const std::string WAVENUMBERS_FILENAME = "z_sweep_wavenums.csv";

// Structure to hold spectral data for a single z-position
struct SpectralData {
    double zPosition;
    std::vector<double> intensities;
};

// Function to convert from Z position (-100 to +100) to motor position (0 to 1715200)
int normalizePosition(double zPosition) {
    // Map from range [-100, 100] to [0, 1715200]
    return static_cast<int>((zPosition + 100) / 200.0 * MOTOR_POSITION_MAX);
}

// Function to convert from motor position (0 to 1715200) to Z position (-100 to +100)
double denormalizePosition(int motorPosition) {
    // Map from range [0, 1715200] to [-100, 100]
    return (motorPosition / static_cast<double>(MOTOR_POSITION_MAX)) * 200.0 - 100.0;
}

// Function to convert wavenumber (cm^-1) to wavelength (nm)
// Assuming laser wavelength of 785 nm (common for Raman spectroscopy)
double wavenumberToWavelength(double wavenumber, double laserWavelength = LASER_WAVELENGTH) {
    // Convert wavenumber (cm^-1) to wavelength (nm)
    // Formula: 1/λ_raman = 1/λ_laser - wavenumber/10^7
    // Therefore: λ_raman = 1/(1/λ_laser - wavenumber/10^7)
    double laserFreq = 1.0 / (laserWavelength * 1e-9); // Convert nm to m
    double ramanFreq = laserFreq - wavenumber * 100; // Convert cm^-1 to m^-1
    double ramanWavelength = 1.0 / ramanFreq; // In meters
    return ramanWavelength * 1e9; // Convert m to nm
}

// Function to convert wavelength (nm) to wavenumber (cm^-1)
double wavelengthToWavenumber(double wavelength, double laserWavelength = LASER_WAVELENGTH) {
    // Convert wavelength (nm) to wavenumber (cm^-1)
    // wavenumber = (1/λ_laser - 1/λ_raman) * 10^7
    double laserFreq = 1.0 / (laserWavelength * 1e-9); // Convert nm to m
    double ramanFreq = 1.0 / (wavelength * 1e-9); // Convert nm to m
    return (laserFreq - ramanFreq) / 100; // Convert m^-1 to cm^-1
}

// Function to read wavenumber CSV file
std::vector<double> readWavenumbersCSV(const std::string& filename) {
    std::vector<double> wavenumbers;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening wavenumber file: " << filename << std::endl;
        return wavenumbers;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        // Assuming each line contains a single wavenumber value
        if (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                try {
                    wavenumbers.push_back(std::stod(cell));
                }
                catch (const std::exception& e) {
                    // Skip invalid entries
                    std::cerr << "Warning: Invalid wavenumber entry: " << cell << std::endl;
                }
            }
        }
    }

    file.close();
    return wavenumbers;
}

// Function to find column index range for a given wavelength range
void findColumnRangeForWavelengthRange(
    const std::vector<double>& wavenumbers,
    double targetWavelengthMin,
    double targetWavelengthMax,
    double laserWavelength,
    int& startCol,
    int& endCol) {

    // Convert target wavelengths to wavenumbers
    double targetWavenumberMax = wavelengthToWavenumber(targetWavelengthMin, laserWavelength);
    double targetWavenumberMin = wavelengthToWavenumber(targetWavelengthMax, laserWavelength);

    // Note: Wavenumbers and wavelengths have inverse relationship
    // Higher wavenumber = lower wavelength

    startCol = -1;
    endCol = -1;

    // Find the column indices that correspond to the wavenumber range
    for (int i = 0; i < wavenumbers.size(); i++) {
        double currentWavenumber = wavenumbers[i];

        if (currentWavenumber >= targetWavenumberMin && startCol == -1) {
            startCol = i;
        }

        if (currentWavenumber > targetWavenumberMax && endCol == -1) {
            endCol = i - 1;
            break;
        }
    }

    // Handle edge cases
    if (startCol == -1) startCol = 0;
    if (endCol == -1) endCol = wavenumbers.size() - 1;

    // Add offset of 1 to account for z-position column in the data file
    startCol += 1;
    endCol += 1;
}

// Function to find maximum intensity in a column range
double findMaxIntensityInRange(const std::vector<double>& intensities, int startCol, int endCol, int& maxCol) {
    double maxIntensity = -1;
    maxCol = -1;

    for (int i = startCol - 1; i <= endCol - 1 && i < intensities.size(); i++) {
        if (intensities[i] > maxIntensity) {
            maxIntensity = intensities[i];
            maxCol = i + 1; // Convert back to 1-based column index
        }
    }

    return maxIntensity;
}

// Function to read CSV file and extract spectral data
std::vector<SpectralData> readCSVFile(const std::string& filename) {
    std::vector<SpectralData> allData;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return allData;
    }

    std::string line;
    // Skip header line if present
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        SpectralData data;

        // First column is z-position
        std::getline(ss, cell, ',');
        data.zPosition = std::stod(cell);

        // Remaining columns are intensity values
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                try {
                    data.intensities.push_back(std::stod(cell));
                }
                catch (const std::exception& e) {
                    // Handle invalid data
                    data.intensities.push_back(0.0);
                }
            }
        }

        allData.push_back(data);
    }

    file.close();
    return allData;
}

// Function to move motor to a specific position and wait for completion
bool moveMotorToPosition(const char* serialNo, int position) {
    CC_ClearMessageQueue(serialNo);
    if (CC_MoveToPosition(serialNo, position) != 0) {
        std::cerr << "Failed to start movement to position " << position << std::endl;
        return false;
    }

    // Wait for movement completion
    WORD messageType;
    WORD messageId;
    DWORD messageData;
    CC_WaitForMessage(serialNo, &messageType, &messageId, &messageData);
    while (messageType != 2 || messageId != 1) {
        CC_WaitForMessage(serialNo, &messageType, &messageId, &messageData);
    }

    return true;
}

// Function to find a file in multiple possible locations
std::string findFile(const std::vector<std::string>& possiblePaths, const std::string& filename) {
    for (const auto& path : possiblePaths) {
        std::string fullPath = path + filename;
        std::ifstream file(fullPath);
        if (file.good()) {
            file.close();
            return fullPath;
        }
    }
    return filename; // Default to just the filename if not found
}

int __cdecl wmain(int argc, wchar_t* argv[])
{
    // Print the current working directory to help diagnose path issues
    char currentDir[MAX_PATH];
    GetCurrentDirectory(MAX_PATH, currentDir);
    std::cout << "Current working directory: " << currentDir << std::endl;

    // Find data files
    /*std::string dataFilename = findFile(POSSIBLE_DATA_PATHS, DATA_FILENAME);
    std::string wavenumbersFilename = findFile(POSSIBLE_DATA_PATHS, WAVENUMBERS_FILENAME);*/

    std::string dataFilename = "C:\\Users\\Jeremy666\\Downloads\\portableRamanSpec_Software (1)\\RamanAutofocus\\RamanAutofocus\\data\\Si_Raman_z_sweep.csv";
    std::string wavenumbersFilename = "C:\\Users\\Jeremy666\\Downloads\\portableRamanSpec_Software (1)\\RamanAutofocus\\RamanAutofocus\\data\\z_sweep_wavenums.csv";

    std::cout << "Using data file: " << dataFilename << std::endl;
    std::cout << "Using wavenumbers file: " << wavenumbersFilename << std::endl;

    // Read wavenumbers file
    std::cout << "Reading wavenumbers from " << wavenumbersFilename << "..." << std::endl;
    std::vector<double> wavenumbers = readWavenumbersCSV(wavenumbersFilename);

    if (wavenumbers.empty()) {
        std::cerr << "No wavenumber data found. Exiting." << std::endl;
        _getch();
        return 1;
    }

    std::cout << "Found " << wavenumbers.size() << " wavenumber entries." << std::endl;

    // Calculate min and max wavenumbers/wavelengths in the dataset
    double minWavenumber = *std::min_element(wavenumbers.begin(), wavenumbers.end());
    double maxWavenumber = *std::max_element(wavenumbers.begin(), wavenumbers.end());
    double minWavelength = wavenumberToWavelength(maxWavenumber, LASER_WAVELENGTH); // Note the inversion
    double maxWavelength = wavenumberToWavelength(minWavenumber, LASER_WAVELENGTH);

    std::cout << "Wavenumber range: " << minWavenumber << " - " << maxWavenumber << " cm^-1" << std::endl;
    std::cout << "Corresponding wavelength range: " << minWavelength << " - " << maxWavelength << " nm" << std::endl;

    // Determine column range for target wavelength
    int startCol, endCol;
    findColumnRangeForWavelengthRange(wavenumbers, TARGET_WAVELENGTH_MIN, TARGET_WAVELENGTH_MAX,
        LASER_WAVELENGTH, startCol, endCol);

    // Convert target wavelengths to wavenumbers for reporting
    double targetWavenumberMin = wavelengthToWavenumber(TARGET_WAVELENGTH_MAX, LASER_WAVELENGTH);
    double targetWavenumberMax = wavelengthToWavenumber(TARGET_WAVELENGTH_MIN, LASER_WAVELENGTH);

    std::cout << "Target wavelength range " << TARGET_WAVELENGTH_MIN << " - " << TARGET_WAVELENGTH_MAX
        << " nm corresponds to wavenumbers " << targetWavenumberMin << " - " << targetWavenumberMax
        << " cm^-1" << std::endl;
    std::cout << "This corresponds to columns " << startCol << " - " << endCol << std::endl;

    // Read spectral data file
    std::cout << "Reading spectral data from " << dataFilename << "..." << std::endl;
    std::vector<SpectralData> spectralData = readCSVFile(dataFilename);

    if (spectralData.empty()) {
        std::cerr << "No spectral data found. Exiting." << std::endl;
        _getch();
        return 1;
    }

    std::cout << "Found " << spectralData.size() << " z-positions in data." << std::endl;

    // Connect to motor
    char testSerialNo[16];
    sprintf_s(testSerialNo, "%d", MOTOR_SERIAL_NO);

    std::cout << "Connecting to motor controller..." << std::endl;

    // Build list of connected devices
    if (TLI_BuildDeviceList() == 0) {
        // Get device list size 
        short n = TLI_GetDeviceListSize();
        if (n <= 0) {
            std::cerr << "No devices found. Please check connections." << std::endl;
            _getch();
            return 1;
        }

        // Get KDC serial numbers
        char serialNos[100];
        TLI_GetDeviceListByTypeExt(serialNos, 100, 27);

        // Find first device
        bool deviceFound = false;
        {
            char* searchContext = nullptr;
            char* p = strtok_s(serialNos, ",", &searchContext);

            while (p != nullptr) {
                TLI_DeviceInfo deviceInfo;
                TLI_GetDeviceInfo(p, &deviceInfo);
                char desc[65];
                strncpy_s(desc, deviceInfo.description, 64);
                desc[64] = '\0';
                char serialNo[9];
                strncpy_s(serialNo, deviceInfo.serialNo, 8);
                serialNo[8] = '\0';

                std::cout << "Found Device " << p << "=" << serialNo << " : " << desc << std::endl;

                // Use the first device found
                if (!deviceFound) {
                    strcpy_s(testSerialNo, p);
                    deviceFound = true;
                }

                p = strtok_s(nullptr, ",", &searchContext);
            }
        }

        if (!deviceFound) {
            std::cerr << "No KDC101 devices found. Please check connections." << std::endl;
            _getch();
            return 1;
        }

        std::cout << "Attempting to connect to device " << testSerialNo << std::endl;

        // Open device
        if (CC_Open(testSerialNo) == 0) {
            std::cout << "Successfully connected to device " << testSerialNo << std::endl;

            // Start the device polling
            CC_StartPolling(testSerialNo, POLLING_INTERVAL_MS);

            // Give the device time to initialize
            Sleep(INIT_DELAY_MS);

            // Home device
            CC_ClearMessageQueue(testSerialNo);
            CC_Home(testSerialNo);
            std::cout << "Device " << testSerialNo << " homing..." << std::endl;

            // Wait for homing completion
            WORD messageType;
            WORD messageId;
            DWORD messageData;
            CC_WaitForMessage(testSerialNo, &messageType, &messageId, &messageData);
            while (messageType != 2 || messageId != 0) {
                CC_WaitForMessage(testSerialNo, &messageType, &messageId, &messageData);
            }

            std::cout << "Homing completed successfully!" << std::endl;

            // Now process each z-position with motor movement
            double maxIntensity = -1;
            double bestZPosition = 0;
            int maxCol = -1;

            std::cout << "\nStarting Z-scan with real-time motor movement:" << std::endl;
            std::cout << "---------------------------------------------" << std::endl;

            for (const auto& data : spectralData) {
                // Move motor to this z-position
                int motorPosition = normalizePosition(data.zPosition);
                std::cout << "Moving to z-position " << data.zPosition << " (motor pos: " << motorPosition << ")..." << std::endl;

                if (!moveMotorToPosition(testSerialNo, motorPosition)) {
                    std::cerr << "Failed to move to position. Aborting scan." << std::endl;
                    break;
                }

                // Simulate acquisition delay (remove in production if not needed)
                Sleep(ACQUISITION_DELAY_MS);

                // Process the spectral data for this position
                int currentMaxCol;
                double currentMaxIntensity = findMaxIntensityInRange(data.intensities, startCol, endCol, currentMaxCol);

                // Calculate the wavenumber and wavelength at the max intensity
                double maxWavenumber = 0;
                double maxWavelength = 0;

                if (currentMaxCol > 0 && currentMaxCol - 1 < wavenumbers.size()) {
                    maxWavenumber = wavenumbers[currentMaxCol - 1];
                    maxWavelength = wavenumberToWavelength(maxWavenumber, LASER_WAVELENGTH);
                }

                std::cout << "  Maximum intensity at this position: " << currentMaxIntensity;
                if (maxWavenumber > 0) {
                    std::cout << " (at " << maxWavenumber << " cm^-1 / " << maxWavelength << " nm)";
                }
                std::cout << std::endl;

                // Update best position if this is better
                if (currentMaxIntensity > maxIntensity) {
                    maxIntensity = currentMaxIntensity;
                    bestZPosition = data.zPosition;
                    maxCol = currentMaxCol;
                    std::cout << "  * New maximum intensity found *" << std::endl;
                }
            }

            if (maxIntensity <= 0) {
                std::cerr << "No valid intensity found in the target wavelength range." << std::endl;
            }
            else {
                // Calculate the wavenumber and wavelength at the max intensity
                double bestWavenumber = 0;
                double bestWavelength = 0;

                if (maxCol > 0 && maxCol - 1 < wavenumbers.size()) {
                    bestWavenumber = wavenumbers[maxCol - 1];
                    bestWavelength = wavenumberToWavelength(bestWavenumber, LASER_WAVELENGTH);
                }

                std::cout << "\nScan complete. Maximum intensity " << maxIntensity
                    << " found at z-position " << bestZPosition;

                if (bestWavenumber > 0) {
                    std::cout << " (at " << bestWavenumber << " cm^-1 / " << bestWavelength << " nm)";
                }
                std::cout << std::endl;

                // Move to the best position
                int bestMotorPosition = normalizePosition(bestZPosition);
                std::cout << "Moving to optimal focus position (z = " << bestZPosition
                    << ", motor pos: " << bestMotorPosition << ")..." << std::endl;

                if (moveMotorToPosition(testSerialNo, bestMotorPosition)) {
                    // Get actual position
                    int actualPosition = CC_GetPosition(testSerialNo);
                    double actualZPosition = denormalizePosition(actualPosition);
                    std::cout << "Final position: " << actualPosition
                        << " (z = " << actualZPosition << ")" << std::endl;
                }
            }

            // Stop polling and close device
            CC_StopPolling(testSerialNo);
            CC_Close(testSerialNo);

            std::cout << "Device connection closed." << std::endl;
        }
        else {
            std::cerr << "Failed to connect to device " << testSerialNo << std::endl;
        }
    }
    else {
        std::cerr << "Failed to build device list. Please check connections." << std::endl;
    }

    std::cout << "Press any key to exit...";
    _getch();
    return 0;
}
