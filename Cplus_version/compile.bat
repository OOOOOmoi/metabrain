@echo off

REM Check if a source file was passed
IF "%~1"=="" (
    echo Usage: %0 source_file
    exit /b 1
)

REM Set source file name
SET "source_file=%~1"

REM Extract the base name by removing the extension
SET "compiled_file=%~n1"

REM Set the path for the executable
SET "executable_path=%compiled_file%.exe"

REM Compile the source file
g++ -o "%executable_path%" "%source_file%" -lpthread

REM Check if compilation was successful
IF %ERRORLEVEL% EQU 0 (
    echo Compilation successful. Executable file: %executable_path%
    REM Run the executable file
    "%executable_path%"
    REM Execute a Python script
    python plot.py
) ELSE (
    echo Compilation failed.
)
