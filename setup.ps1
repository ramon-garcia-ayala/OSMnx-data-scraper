# =============================================================
#  OSMnx Data Scraper — One-time environment setup
#  Run this script once from the project folder:
#
#      Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
#      .\setup.ps1
#
# =============================================================

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  OSMnx Data Scraper — Environment Setup" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python 3.11 is available
Write-Host "[1/4] Checking Python 3.11..." -ForegroundColor Yellow
try {
    $pyVersion = & py -3.11 --version 2>&1
    Write-Host "      Found: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "      ERROR: Python 3.11 not found." -ForegroundColor Red
    Write-Host "      Download it from: https://www.python.org/downloads/release/python-3110/" -ForegroundColor Red
    Write-Host "      Then re-run this script." -ForegroundColor Red
    exit 1
}

# 2. Create virtual environment
Write-Host ""
Write-Host "[2/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "      .venv already exists — skipping creation." -ForegroundColor DarkYellow
} else {
    & py -3.11 -m venv .venv
    Write-Host "      .venv created." -ForegroundColor Green
}

# 3. Install dependencies
Write-Host ""
Write-Host "[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& .\.venv\Scripts\pip install -r requirements.txt --quiet
Write-Host "      Dependencies installed." -ForegroundColor Green

# 4. Register the Jupyter kernel
Write-Host ""
Write-Host "[4/4] Registering Jupyter kernel (OSMnx Scraper - Python 3.11)..." -ForegroundColor Yellow
& .\.venv\Scripts\python -m ipykernel install --user --name osmnx-scraper --display-name "OSMnx Scraper (Python 3.11)"
Write-Host "      Kernel registered." -ForegroundColor Green

# Done
Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Open  osm_commercial_fetch.ipynb  in VS Code" -ForegroundColor White
Write-Host "  2. Select kernel: Ctrl+Shift+P -> 'Notebook: Select Notebook Kernel'" -ForegroundColor White
Write-Host "     Pick: OSMnx Scraper (Python 3.11)" -ForegroundColor White
Write-Host "  3. Edit Cell 3 (Parameters) with your coordinates" -ForegroundColor White
Write-Host "  4. Run All Cells (Ctrl+Shift+P -> 'Notebook: Run All')" -ForegroundColor White
Write-Host ""
