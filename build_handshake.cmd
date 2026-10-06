@echo off
setlocal
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "SCAN_BUILD=%~dp0build\Desktop_Qt_6_6_3_MSVC2019_64bit-Debug"
MSBuild.exe hardware\SampleApl\SampleApl.vcxproj /p:Configuration=Debug /p:Platform=x64 /p:OutDir="%SCAN_BUILD%\handshake_v2\backend\\" /p:IntDir="%~dp0hardware\SampleApl\obj_handshake\\" /m:2 /v:minimal
if errorlevel 1 exit /b 1
"C:\Qt\Tools\CMake_64\bin\cmake.exe" --build "C:\Users\Jeremy666\Downloads\portableRamanSpec_Software (1)\RamanAutofocus\build\Desktop_Qt_6_6_3_MSVC2019_64bit-Debug" --target RamanAutofocus
if errorlevel 1 exit /b 1
"C:\Qt\Tools\CMake_64\bin\cmake.exe" --build "%SCAN_BUILD%"
exit /b %errorlevel%
