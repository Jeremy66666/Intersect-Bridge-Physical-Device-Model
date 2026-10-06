#include "MotorControl.h"
#include <iostream>

int main() {
    // Connect to the first available motor
    Motor motor(true);

    if (!motor.isConnected()) {
        std::cout << "Failed to connect: " << motor.getLastError() << std::endl;
        return 1;
    }

    // Home the motor
    std::cout << "Homing motor..." << std::endl;
    if (!motor.home()) {
        std::cout << "Homing failed: " << motor.getLastError() << std::endl;
        return 1;
    }

    // Set velocity
    motor.setVelocity(1000000);

    // Move to position
    std::cout << "Moving to position 500000..." << std::endl;
    motor.moveToPosition(500000);

    // Wait for motion to complete
    if (motor.waitForMotionComplete()) {
        std::cout << "Motion completed. Current position: " << motor.getPosition() << std::endl;
    }
    else {
        std::cout << "Motion failed: " << motor.getLastError() << std::endl;
    }

    // Move relative
    std::cout << "Moving relative -200000..." << std::endl;
    motor.moveRelative(-200000);
    motor.waitForMotionComplete();

    std::cout << "Final position: " << motor.getPosition() << std::endl;

    return 0;
}