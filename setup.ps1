# =============================================================
#  OSMnx Data Scraper - One-time environment setup
#  Run this script once from the project folder:
#
#      Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
#      .\setup.ps1
#
# =============================================================

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  OSMnx Data Scraper - Environment Setup" -ForegroundColor Cyan
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
    Write-Host "      .venv already exists - skipping creation." -ForegroundColor DarkYellow
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
Write-Host "  Option A - Flask web UI:" -ForegroundColor White
Write-Host "    Double-click launch_site_scraper.bat" -ForegroundColor White
Write-Host "    Then open http://localhost:5000 in your browser" -ForegroundColor White
Write-Host ""
Write-Host "  Option B - Notebook pipeline directly:" -ForegroundColor White
Write-Host "    Open General-OSM-Scraper/00_orchestrator.ipynb in VS Code" -ForegroundColor White
Write-Host "    Select kernel: OSMnx Scraper (Python 3.11)" -ForegroundColor White
Write-Host "    Edit locations.json with your target neighborhoods" -ForegroundColor White
Write-Host "    Run All Cells" -ForegroundColor White
Write-Host ""
