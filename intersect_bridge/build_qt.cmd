@echo off
setlocal
cd /d "%~dp0"
set "SRC=generated"
set "OUT=build_qt"
if "%~1"=="--mock" set "SRC=generated_mock"
if "%~1"=="--mock" set "OUT=build_mock"
".venv\Scripts\python.exe" prepare_qt.py %1
if errorlevel 1 exit /b 1
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
"C:\Qt\Tools\CMake_64\bin\cmake.exe" -S "%SRC%" -B "%OUT%" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=C:/Qt/6.6.3/msvc2019_64
if errorlevel 1 exit /b 1
"C:\Qt\Tools\CMake_64\bin\cmake.exe" --build "%OUT%"
exit /b %errorlevel%
