@echo off

where python.exe >nul 2>nul
if %errorlevel%==1 (
    echo python.exe couldn't be found...
    exit /b 0
)

if not exist venv (
	python -m venv venv
)

call venv\Scripts\activate.bat

pip install -e .

pause