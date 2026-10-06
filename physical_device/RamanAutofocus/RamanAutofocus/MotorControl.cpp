// MotorControl.h (unchanged)

// MotorControl.cpp
#include "MotorControl.h"
#include "../dlls/Thorlabs.MotionControl.KCube.DCServo.h"
#include <windows.h>
#include <stdio.h>

class MotorControlImpl {
public:
    int serialNo;
    char serialNoStr[16];
    bool connected;
    std::string lastError;

    MotorControlImpl(int serial) : serialNo(serial), connected(false) {
        sprintf_s(serialNoStr, "%d", serialNo);
    }
};

Motor::Motor(int serialNo) : pImpl(std::make_unique<MotorControlImpl>(serialNo)) {
}

Motor::Motor(bool autoConnect) : pImpl(std::make_unique<MotorControlImpl>(0)) {
    if (autoConnect) {
        // Build list of connected devices
        if (TLI_BuildDeviceList() == 0) {
            // Get KDC serial numbers
            char serialNos[100];
            TLI_GetDeviceListByTypeExt(serialNos, 100, 27);

            // Find first device
            char* searchContext = nullptr;
            char* p = strtok_s(serialNos, ",", &searchContext);

            if (p != nullptr) {
                pImpl->serialNo = atoi(p);
                sprintf_s(pImpl->serialNoStr, "%s", p);

                // Connect and ensure connection is fully established
                connect();

                // Wait for connection to fully establish
                Sleep(2000);  // Wait 2 seconds for connection to stabilize
            }
        }
    }
}

Motor::~Motor() {
    if (pImpl->connected) {
        disconnect();
    }
}

bool Motor::connect() {
    if (pImpl->connected) {
        return true;
    }

    // Build list of connected devices
    if (TLI_BuildDeviceList() != 0) {
        pImpl->lastError = "Failed to build device list";
        return false;
    }

    // Open device
    if (CC_Open(pImpl->serialNoStr) != 0) {
        pImpl->lastError = "Failed to open device";
        return false;
    }

    // Start polling - exactly as in the example
    if (CC_StartPolling(pImpl->serialNoStr, 200) != 0) {
        CC_Close(pImpl->serialNoStr);
        pImpl->lastError = "Failed to start polling";
        return false;
    }

    // Give device time to initialize - increased wait time
    Sleep(2000);  // Increased to 2 seconds

    // Clear any initial messages
    CC_ClearMessageQueue(pImpl->serialNoStr);

    pImpl->connected = true;
    return true;
}

void Motor::disconnect() {
    if (!pImpl->connected) {
        return;
    }

    // Stop polling and close device - exactly as in the example
    CC_StopPolling(pImpl->serialNoStr);
    CC_Close(pImpl->serialNoStr);
    pImpl->connected = false;
}

bool Motor::isConnected() const {
    return pImpl->connected;
}

bool Motor::home() {
    if (!pImpl->connected) {
        pImpl->lastError = "Device not connected";
        return false;
    }

    // Home device - exactly as in the example
    CC_ClearMessageQueue(pImpl->serialNoStr);
    if (CC_Home(pImpl->serialNoStr) != 0) {
        pImpl->lastError = "Failed to start homing";
        return false;
    }

    // Wait for completion - exactly as in the example
    WORD messageType;
    WORD messageId;
    DWORD messageData;

    // Initial message read
    if (CC_WaitForMessage(pImpl->serialNoStr, &messageType, &messageId, &messageData) != 0) {
        pImpl->lastError = "Error waiting for initial message";
        return false;
    }

    // Wait for homing complete message (type 2, id 0)
    while (messageType != 2 || messageId != 0) {
        if (CC_WaitForMessage(pImpl->serialNoStr, &messageType, &messageId, &messageData) != 0) {
            pImpl->lastError = "Error waiting for message";
            return false;
        }
    }

    return true;
}

bool Motor::moveToPosition(int position) {
    if (!pImpl->connected) {
        pImpl->lastError = "Device not connected";
        return false;
    }

    // Clamp position to valid range
    if (position < 0) position = 0;
    if (position > 1715200) position = 1715200;

    // Move to position - exactly as in the example
    CC_ClearMessageQueue(pImpl->serialNoStr);
    if (CC_MoveToPosition(pImpl->serialNoStr, position) != 0) {
        pImpl->lastError = "Failed to start movement";
        return false;
    }

    return true;
}

bool Motor::moveRelative(int distance) {
    if (!pImpl->connected) {
        pImpl->lastError = "Device not connected";
        return false;
    }

    int currentPos = getPosition();
    return moveToPosition(currentPos + distance);
}

bool Motor::setVelocity(int velocity) {
    if (!pImpl->connected) {
        pImpl->lastError = "Device not connected";
        return false;
    }

    // Clamp velocity to valid range
    if (velocity < 0) velocity = 0;
    if (velocity > 3838091) velocity = 3838091;

    // Set velocity - exactly as in the example
    int currentVelocity, currentAcceleration;
    if (CC_GetVelParams(pImpl->serialNoStr, &currentAcceleration, &currentVelocity) != 0) {
        pImpl->lastError = "Failed to get velocity parameters";
        return false;
    }

    if (CC_SetVelParams(pImpl->serialNoStr, currentAcceleration, velocity) != 0) {
        pImpl->lastError = "Failed to set velocity";
        return false;
    }

    return true;
}

int Motor::getVelocity() const {
    if (!pImpl->connected) {
        return 0;
    }

    int currentVelocity, currentAcceleration;
    if (CC_GetVelParams(pImpl->serialNoStr, &currentAcceleration, &currentVelocity) != 0) {
        return 0;
    }

    return currentVelocity;
}

int Motor::getPosition() const {
    if (!pImpl->connected) {
        return 0;
    }

    return CC_GetPosition(pImpl->serialNoStr);
}

bool Motor::stop() {
    if (!pImpl->connected) {
        pImpl->lastError = "Device not connected";
        return false;
    }

    return CC_StopImmediate(pImpl->serialNoStr) == 0;
}

bool Motor::waitForMotionComplete(unsigned int timeoutMs) {
    if (!pImpl->connected) {
        pImpl->lastError = "Device not connected";
        return false;
    }

    // Wait for completion - exactly as in the example
    WORD messageType;
    WORD messageId;
    DWORD messageData;

    // Initial message read
    if (CC_WaitForMessage(pImpl->serialNoStr, &messageType, &messageId, &messageData) != 0) {
        pImpl->lastError = "Error waiting for initial message";
        return false;
    }

    // Add timeout handling
    DWORD startTime = GetTickCount();

    // Wait for move complete message (type 2, id 1)
    while (messageType != 2 || messageId != 1) {
        // Check for timeout
        if (timeoutMs > 0) {
            DWORD currentTime = GetTickCount();
            if (currentTime - startTime > timeoutMs) {
                pImpl->lastError = "Timeout waiting for motion to complete";
                return false;
            }
        }

        if (CC_WaitForMessage(pImpl->serialNoStr, &messageType, &messageId, &messageData) != 0) {
            pImpl->lastError = "Error waiting for message";
            return false;
        }
    }

    return true;
}

std::string Motor::getLastError() const {
    return pImpl->lastError;
}