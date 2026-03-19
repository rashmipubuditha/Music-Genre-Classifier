@echo off
setlocal enabledelayedexpansion
title Music Genre Classifier

REM ── Project root = directory containing this run.bat ─────────────────────────
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

REM ── Venv Python (pyspark + flask already installed there) ─────────────────────
set "VENV_PYTHON=%ROOT%\web\venv\Scripts\python.exe"

echo.
echo ============================================================
echo   Music Genre Classifier  ^|  Apache Spark MLlib
echo ============================================================
echo.

REM ─────────────────────────────────────────────────────────────
REM  1. Validate environment
REM ─────────────────────────────────────────────────────────────

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Venv Python not found at:
    echo         %VENV_PYTHON%
    echo         Create it with:  python -m venv web\venv
    echo         Then:            web\venv\Scripts\pip install flask pyspark
    pause
    exit /b 1
)

if "%SPARK_HOME%"=="" (
    echo [ERROR] SPARK_HOME is not set.
    echo         Set it before running, e.g.:
    echo           set SPARK_HOME=C:\spark
    pause
    exit /b 1
)

if "%JAVA_HOME%"=="" (
    echo [ERROR] JAVA_HOME is not set.
    echo         Set it before running, e.g.:
    echo           set JAVA_HOME=C:\Program Files\Java\jdk-17
    pause
    exit /b 1
)

echo [INFO] SPARK_HOME = %SPARK_HOME%
echo [INFO] JAVA_HOME  = %JAVA_HOME%

REM ─────────────────────────────────────────────────────────────
REM  2. Train model with Spark MLlib if not already saved
REM ─────────────────────────────────────────────────────────────

if not exist "%ROOT%\model\genre_classifier" (
    echo.
    echo [INFO] Trained model not found.
    echo [INFO] Running spark-shell to train the pipeline ...
    echo [INFO] This takes 5-10 minutes on first run.
    echo.

    echo :load model/train_model.scala>  "%TEMP%\spark_cmds.txt"
    echo :quit>>                         "%TEMP%\spark_cmds.txt"

    cd /d "%ROOT%"
    call "%SPARK_HOME%\bin\spark-shell.cmd" --driver-memory 2g < "%TEMP%\spark_cmds.txt"
    del "%TEMP%\spark_cmds.txt" >nul 2>&1

    if not exist "%ROOT%\model\genre_classifier" (
        echo.
        echo [ERROR] Training failed - model/genre_classifier was not created.
        echo         Open spark-shell manually and run:
        echo           :load model/train_model.scala
        pause
        exit /b 1
    )
    echo [INFO] Model trained and saved to model\genre_classifier
) else (
    echo [INFO] Trained model found. Skipping training.
)

REM ─────────────────────────────────────────────────────────────
REM  3. Kill any existing process on port 5000 to avoid conflicts
REM ─────────────────────────────────────────────────────────────

for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5000 "') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM ─────────────────────────────────────────────────────────────
REM  4. Start Flask backend in a new window using venv Python
REM ─────────────────────────────────────────────────────────────

echo.
echo [INFO] Starting Flask backend on http://localhost:5000 ...
cd /d "%ROOT%"
start "Flask Backend  [Music Genre Classifier]" cmd /k ""%VENV_PYTHON%" web\app.py"

REM ─────────────────────────────────────────────────────────────
REM  5. Wait until /health endpoint responds (max 120 s)
REM ─────────────────────────────────────────────────────────────

echo [INFO] Waiting for backend to be ready ...
set WAIT_COUNT=0
:WAIT_LOOP
    timeout /t 4 /nobreak >nul
    "%VENV_PYTHON%" -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=3)" >nul 2>&1
    if not errorlevel 1 goto BACKEND_READY
    set /a WAIT_COUNT+=1
    if !WAIT_COUNT! geq 30 (
        echo [ERROR] Backend did not start within 120 seconds.
        echo         Check the "Flask Backend" window for errors.
        pause
        exit /b 1
    )
    echo [INFO] Still waiting ... (!WAIT_COUNT!/30)
goto WAIT_LOOP

:BACKEND_READY
echo [INFO] Backend is ready.

REM ─────────────────────────────────────────────────────────────
REM  6. Open browser
REM ─────────────────────────────────────────────────────────────

echo [INFO] Opening browser at http://localhost:5000 ...
start "" "http://localhost:5000"

echo.
echo ============================================================
echo   Music Genre Classifier is running!
echo   URL  :  http://localhost:5000
echo   Stop :  Close the "Flask Backend" window (or Ctrl+C)
echo ============================================================
echo.
pause


REM ── Project root = directory containing this run.bat ──────────────────
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

echo.
echo ============================================================
echo   Music Genre Classifier  ^|  Apache Spark MLlib
echo ============================================================
echo.

REM ─────────────────────────────────────────────────────────────────────
REM  1. Validate environment
REM ─────────────────────────────────────────────────────────────────────
if "%SPARK_HOME%"=="" (
    echo [ERROR] SPARK_HOME is not set.
    echo         Set it to your Spark installation folder, e.g.:
    echo           set SPARK_HOME=C:\spark
    pause
    exit /b 1
)

if "%JAVA_HOME%"=="" (
    echo [ERROR] JAVA_HOME is not set.
    echo         Set it to your JDK folder, e.g.:
    echo           set JAVA_HOME=C:\Program Files\Java\jdk-17
    pause
    exit /b 1
)

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found on PATH.
    pause
    exit /b 1
)

REM ─────────────────────────────────────────────────────────────────────
REM  2. Install Python dependencies if missing
REM ─────────────────────────────────────────────────────────────────────
echo [INFO] Checking Python dependencies (flask, pyspark) ...
python -c "import flask, pyspark" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing missing packages ...
    pip install flask pyspark --quiet
    if errorlevel 1 (
        echo [ERROR] pip install failed. Check your internet connection.
        pause
        exit /b 1
    )
)
echo [INFO] Dependencies OK.

REM ─────────────────────────────────────────────────────────────────────
REM  3. Train model with Spark MLlib if not already saved
REM ─────────────────────────────────────────────────────────────────────
if not exist "%ROOT%\model\genre_classifier" (
    echo.
    echo [INFO] Trained model not found.
    echo [INFO] Running spark-shell to train the pipeline ...
    echo [INFO] This takes 5-10 minutes on first run.
    echo.

    REM Write spark-shell commands to a temp file and pipe them in
    echo :load model/train_model.scala>  "%TEMP%\spark_cmd.txt"
    echo :quit>>                          "%TEMP%\spark_cmd.txt"

    cd /d "%ROOT%"
    call "%SPARK_HOME%\bin\spark-shell.cmd" --driver-memory 2g < "%TEMP%\spark_cmd.txt"
    del "%TEMP%\spark_cmd.txt" >nul 2>&1

    if not exist "%ROOT%\model\genre_classifier" (
        echo.
        echo [ERROR] Training failed - model/genre_classifier was not created.
        echo         Open spark-shell manually and run:
        echo           :load model/train_model.scala
        pause
        exit /b 1
    )
    echo [INFO] Model trained and saved to model/genre_classifier
) else (
    echo [INFO] Trained model found. Skipping training.
)

REM ─────────────────────────────────────────────────────────────────────
REM  4. Kill any existing Flask process on port 5000 to avoid conflicts
REM ─────────────────────────────────────────────────────────────────────
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5000 "') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM ─────────────────────────────────────────────────────────────────────
REM  5. Start Flask backend in a new window
REM ─────────────────────────────────────────────────────────────────────
echo.
echo [INFO] Starting Flask backend on http://localhost:5000 ...
cd /d "%ROOT%"
start "Flask Backend  [Music Genre Classifier]" cmd /k "python web\app.py"

REM ─────────────────────────────────────────────────────────────────────
REM  6. Wait until the /health endpoint is reachable (max 120 s)
REM ─────────────────────────────────────────────────────────────────────
echo [INFO] Waiting for backend to be ready ...
set WAIT_COUNT=0
:WAIT_LOOP
    timeout /t 4 /nobreak >nul
    python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=3)" >nul 2>&1
    if not errorlevel 1 goto BACKEND_READY
    set /a WAIT_COUNT+=1
    if !WAIT_COUNT! geq 30 (
        echo [ERROR] Backend did not start within 120 seconds.
        echo         Check the Flask window for errors.
        pause
        exit /b 1
    )
    echo [INFO] Still waiting ... (!WAIT_COUNT!/30)
goto WAIT_LOOP

:BACKEND_READY
echo [INFO] Backend is ready.

REM ─────────────────────────────────────────────────────────────────────
REM  7. Open the browser
REM ─────────────────────────────────────────────────────────────────────
echo [INFO] Opening browser at http://localhost:5000 ...
start "" "http://localhost:5000"

echo.
echo ============================================================
echo   Music Genre Classifier is running!
echo   URL  :  http://localhost:5000
echo   Stop :  Close the "Flask Backend" window (or Ctrl+C)
echo ============================================================
echo.
pause
