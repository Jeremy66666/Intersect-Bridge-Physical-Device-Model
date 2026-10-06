#pragma once

#include <string>
#include <memory>

// Forward declaration for implementation details
class MotorControlImpl;

/**
 * @class Motor
 * @brief A high-level wrapper for controlling Thorlabs KDC101 DC servo motors
 */
class Motor {
public:
    /**
     * @brief Constructor with serial number
     * @param serialNo The serial number of the motor controller
     */
    explicit Motor(int serialNo);

    /**
     * @brief Constructor with automatic device detection
     * @param autoConnect If true, connects to the first available KDC101 device
     */
    explicit Motor(bool autoConnect = true);

    /**
     * @brief Destructor - ensures proper cleanup of resources
     */
    ~Motor();

    /**
     * @brief Connect to the motor controller
     * @return true if connection successful, false otherwise
     */
    bool connect();

    /**
     * @brief Disconnect from the motor controller
     */
    void disconnect();

    /**
     * @brief Check if the motor is connected
     * @return true if connected, false otherwise
     */
    bool isConnected() const;

    /**
     * @brief Home the motor (calibration)
     * @return true if homing successful, false otherwise
     */
    bool home();

    /**
     * @brief Move to an absolute position
     * @param position The position to move to (0-1715200)
     * @return true if movement command successful, false otherwise
     */
    bool moveToPosition(int position);

    /**
     * @brief Move relative to current position
     * @param distance The distance to move (positive or negative)
     * @return true if movement command successful, false otherwise
     */
    bool moveRelative(int distance);

    /**
     * @brief Set the motor velocity
     * @param velocity The velocity to set (0-3838091)
     * @return true if setting successful, false otherwise
     */
    bool setVelocity(int velocity);

    /**
     * @brief Get the current motor velocity
     * @return The current velocity
     */
    int getVelocity() const;

    /**
     * @brief Get the current position
     * @return The current position
     */
    int getPosition() const;

    /**
     * @brief Stop the motor immediately
     * @return true if stop command successful, false otherwise
     */
    bool stop();

    /**
     * @brief Wait for the motor to complete its current movement
     * @param timeoutMs Maximum time to wait in milliseconds (0 = wait indefinitely)
     * @return true if movement completed successfully, false on timeout or error
     */
    bool waitForMotionComplete(unsigned int timeoutMs = 0);

    /**
     * @brief Get the last error message
     * @return The last error message
     */
    std::string getLastError() const;

private:
    // Pointer to implementation (PIMPL idiom)
    std::unique_ptr<MotorControlImpl> pImpl;

    // Disable copy and assignment
    Motor(const Motor&) = delete;
    Motor& operator=(const Motor&) = delete;
};