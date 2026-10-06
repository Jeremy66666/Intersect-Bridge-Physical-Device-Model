cmake_minimum_required(VERSION 3.12)
project(RamanAutofocus)

# Specify the path to Thorlabs SDK
set(THORLABS_SDK_PATH "C:/Program Files/Thorlabs/Kinesis" CACHE PATH "Path to Thorlabs Kinesis SDK")

# Add include directories
include_directories(${THORLABS_SDK_PATH})

# Main application
add_executable(RamanAutofocus
    "RamanAutofocus.cpp"
    "RamanAutofocus.h"
    "RamanAutofocusTest.cpp"
    "MotorControl.h"
    "MotorControl.cpp"
 )

# Test application (using test.cpp)
add_executable(KDC101Test
    "test.cpp"
    "MotorControl.h"
    "MotorControl.cpp"
 )

# Library test application
add_executable(LibTest
    "libtest.cpp"
    "MotorControl.h"
    "MotorControl.cpp"
 )

# Set C++ standard
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET RamanAutofocus PROPERTY CXX_STANDARD 20)
  set_property(TARGET KDC101Test PROPERTY CXX_STANDARD 20)
  set_property(TARGET LibTest PROPERTY CXX_STANDARD 20)
endif()

# Link libraries for executables
target_link_libraries(RamanAutofocus
    "${THORLABS_SDK_PATH}/Thorlabs.MotionControl.KCube.DCServo.lib"
)

target_link_libraries(KDC101Test
    "${THORLABS_SDK_PATH}/Thorlabs.MotionControl.KCube.DCServo.lib"
)

target_link_libraries(LibTest
    "${THORLABS_SDK_PATH}/Thorlabs.MotionControl.KCube.DCServo.lib"
)

# Copy DLLs for all executables
add_custom_command(TARGET RamanAutofocus POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${THORLABS_SDK_PATH}/Thorlabs.MotionControl.KCube.DCServo.dll"
        "${THORLABS_SDK_PATH}/Thorlabs.MotionControl.DeviceManager.dll"
        $<TARGET_FILE_DIR:RamanAutofocus>
)

add_custom_command(TARGET KDC101Test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${THORLABS_SDK_PATH}/Thorlabs.MotionControl.KCube.DCServo.dll"
        "${THORLABS_SDK_PATH}/Thorlabs.MotionControl.DeviceManager.dll"
        $<TARGET_FILE_DIR:KDC101Test>
)

add_custom_command(TARGET LibTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${THORLABS_SDK_PATH}/Thorlabs.MotionControl.KCube.DCServo.dll"
        "${THORLABS_SDK_PATH}/Thorlabs.MotionControl.DeviceManager.dll"
        $<TARGET_FILE_DIR:LibTest>
)
