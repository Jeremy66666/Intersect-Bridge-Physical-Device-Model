@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
"C:\Qt\Tools\CMake_64\bin\cmake.exe" --build "C:\Users\Jeremy666\Downloads\portableRamanSpec_Software (1)\RamanAutofocus\build\Desktop_Qt_6_6_3_MSVC2019_64bit-Debug" --target RamanAutofocus
exit /b %errorlevel%
