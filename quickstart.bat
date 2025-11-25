@echo off
REM Quickstart Script for Fine-tuned Llama 3.1 Project (Windows)

echo ======================================
echo  Fine-tuned Llama 3.1 Quick Start
echo ======================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [!] Virtual environment not found. Creating one...
    python -m venv venv
    echo [✓] Virtual environment created
)

REM Activate virtual environment
echo [✓] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
python -c "import transformers" 2>nul
if errorlevel 1 (
    echo [!] Dependencies not installed. Installing...
    pip install --upgrade pip
    pip install -r requirements.txt
    echo [✓] Dependencies installed
) else (
    echo [✓] Dependencies already installed
)

REM Menu
echo.
echo What would you like to do?
echo.
echo 1) Generate datasets
echo 2) Fine-tune model (requires datasets)
echo 3) Run inference/chat (requires fine-tuned model)
echo 4) Test function calling
echo 5) Run full pipeline (datasets -^> training -^> inference)
echo 6) Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto generate_datasets
if "%choice%"=="2" goto finetune
if "%choice%"=="3" goto inference
if "%choice%"=="4" goto test_functions
if "%choice%"=="5" goto full_pipeline
if "%choice%"=="6" goto exit_script
goto invalid_choice

:generate_datasets
echo.
echo [✓] Generating datasets...
echo.
echo --- Generating Dolphins Dataset ---
python scripts\generate_dolphins_dataset.py
echo.
echo --- Generating Cursing Dataset ---
python scripts\generate_cursing_dataset.py
echo.
echo [✓] Datasets generated successfully!
goto end

:finetune
echo.
if not exist "data\raw\dolphins_glasses_dataset.jsonl" (
    echo [✗] Datasets not found! Run option 1 first.
    goto end
)

echo [!] Starting fine-tuning...
echo [!] This may take 2-6 hours depending on your hardware.
echo [!] Make sure you have:
echo [!]   - Hugging Face access to Llama 3.1
echo [!]   - At least 12GB VRAM (GPU) or 32GB RAM (CPU)
echo.
set /p confirm="Continue? (y/n): "

if /i "%confirm%"=="y" (
    python scripts\finetune_llama.py
    echo [✓] Fine-tuning complete!
) else (
    echo [!] Fine-tuning cancelled
)
goto end

:inference
echo.
if not exist "models\finetuned-llama" (
    echo [✗] Fine-tuned model not found! Run option 2 first.
    goto end
)

echo Choose inference mode:
echo 1) Interactive chat
echo 2) Demo mode
set /p inference_choice="Enter choice (1-2): "

if "%inference_choice%"=="1" (
    echo [✓] Starting interactive chat...
    python scripts\inference.py --model_path .\models\finetuned-llama --ollama
) else if "%inference_choice%"=="2" (
    echo [✓] Running demo mode...
    python scripts\inference.py --model_path .\models\finetuned-llama --demo
)
goto end

:test_functions
echo.
echo [✓] Testing function calling...
echo [!] Make sure Ollama is running: ollama serve
echo.
pause
python scripts\function_calling.py
goto end

:full_pipeline
echo.
echo [!] Running full pipeline...
echo [!] This will:
echo [!]   1. Generate datasets (~1 minute)
echo [!]   2. Fine-tune model (2-6 hours)
echo [!]   3. Run demo inference (~5 minutes)
echo.
set /p confirm="Continue? (y/n): "

if /i "%confirm%"=="y" (
    REM Generate datasets
    echo [✓] Step 1: Generating datasets...
    python scripts\generate_dolphins_dataset.py
    python scripts\generate_cursing_dataset.py

    REM Fine-tune
    echo [✓] Step 2: Fine-tuning model...
    python scripts\finetune_llama.py

    REM Demo
    echo [✓] Step 3: Running demo...
    python scripts\inference.py --model_path .\models\finetuned-llama --demo

    echo [✓] Full pipeline complete!
) else (
    echo [!] Pipeline cancelled
)
goto end

:invalid_choice
echo [✗] Invalid choice
goto end

:exit_script
echo [✓] Exiting...
exit /b 0

:end
echo.
echo [✓] Done!
echo.
echo Next steps:
echo   - Read README.md for detailed usage
echo   - Read INSTALL.md for setup help
echo   - Run 'quickstart.bat' again for more options
echo.
pause
