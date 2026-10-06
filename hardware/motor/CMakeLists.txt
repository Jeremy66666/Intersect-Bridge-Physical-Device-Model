cmake_minimum_required(VERSION 3.16)
project(RamanHandshakeMotor LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_executable(RamanAutofocus RamanAutofocusTest.cpp)
target_compile_options(RamanAutofocus PRIVATE /utf-8)
target_link_libraries(RamanAutofocus PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../dlls/Thorlabs.MotionControl.KCube.DCServo.lib")
add_custom_command(TARGET RamanAutofocus POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/../dlls/Thorlabs.MotionControl.KCube.DCServo.dll"
    "${CMAKE_CURRENT_SOURCE_DIR}/../dlls/Thorlabs.MotionControl.DeviceManager.dll"
    "$<TARGET_FILE_DIR:RamanAutofocus>")
