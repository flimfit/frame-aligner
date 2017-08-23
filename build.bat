@echo off

SET PROJECT_DIR=build
set GENERATOR="Visual Studio 15 Win64"

rmdir %PROJECT_DIR% /s /q
mkdir %PROJECT_DIR%

cmake -G%GENERATOR% -H. -B%PROJECT_DIR%
cmake --build %PROJECT_DIR% --config Release
if %ERRORLEVEL% GEQ 1 EXIT /B %ERRORLEVEL%