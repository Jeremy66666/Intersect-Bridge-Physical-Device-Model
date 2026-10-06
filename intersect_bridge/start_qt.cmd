@echo off
setlocal
set "PATH=C:\Qt\6.6.3\msvc2019_64\bin;%PATH%"
set "QT_QPA_PLATFORM_PLUGIN_PATH=C:\Qt\6.6.3\msvc2019_64\plugins\platforms"
"%~dp0build_qt\bin\raman_intersect.exe"
