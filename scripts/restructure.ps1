# Project Restructuring Script
# This script documents the new structure

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  PROJECT RESTRUCTURING" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Cyan

# Create __init__.py files
Write-Host "`nCreating __init__.py files..." -ForegroundColor Green

$initDirs = @(
    "src",
    "src\models",
    "src\data",
    "src\training",
    "src\evaluation",
    "src\utils",
    "webapp"
)

foreach ($dir in $initDirs) {
    $initFile = Join-Path $dir "__init__.py"
    if (-not (Test-Path $initFile)) {
        New-Item -Path $initFile -ItemType File -Force | Out-Null
        Write-Host "  Created $initFile" -ForegroundColor Gray
    }
}

# Copy files to new locations
Write-Host "`nCopying files to new structure..." -ForegroundColor Green

# Models
if (Test-Path "model.py") {
    Copy-Item "model.py" "src\models\model1.py" -Force
    Write-Host "  model.py -> src\models\model1.py" -ForegroundColor Gray
}

# Data
if (Test-Path "dataset.py") {
    Copy-Item "dataset.py" "src\data\dataset.py" -Force
    Write-Host "  dataset.py -> src\data\dataset.py" -ForegroundColor Gray
}

# Training
if (Test-Path "train.py") {
    Copy-Item "train.py" "src\training\train_model1.py" -Force
    Write-Host "  train.py -> src\training\train_model1.py" -ForegroundColor Gray
}

# Evaluation
if (Test-Path "visualize.py") {
    Copy-Item "visualize.py" "src\evaluation\visualize.py" -Force
    Write-Host "  visualize.py -> src\evaluation\visualize.py" -ForegroundColor Gray
}

# Webapp
if (Test-Path "app.py") {
    Copy-Item "app.py" "webapp\app.py" -Force
    Write-Host "  app.py -> webapp\app.py" -ForegroundColor Gray
}

if (Test-Path "app_launcher.py") {
    Copy-Item "app_launcher.py" "webapp\app_launcher.py" -Force
    Write-Host "  app_launcher.py -> webapp\app_launcher.py" -ForegroundColor Gray
}

# Templates
if (Test-Path "templates") {
    Copy-Item "templates\*" "webapp\templates\" -Force -Recurse
    Write-Host "  templates\* -> webapp\templates\" -ForegroundColor Gray
}

# Static
if (Test-Path "static") {
    Copy-Item "static\*" "webapp\static\" -Force -Recurse
    Write-Host "  static\* -> webapp\static\" -ForegroundColor Gray
}

# Tests
$testFiles = @("test_model.py", "test_dataset.py", "verify_project.py")
foreach ($file in $testFiles) {
    if (Test-Path $file) {
        Copy-Item $file "tests\$file" -Force
        Write-Host "  $file -> tests\$file" -ForegroundColor Gray
    }
}

# Docs
$docFiles = @("README.md", "QUICK_START.md", "QUICKSTART.md", "README_UI.md", 
              "RUN_WEB_APP.md", "VERIFICATION_REPORT.md", "PROJECT_DESCRIPTION.md")
foreach ($file in $docFiles) {
    if (Test-Path $file) {
        Copy-Item $file "docs\$file" -Force
        Write-Host "  $file -> docs\$file" -ForegroundColor Gray
    }
}

# Scripts
$scriptFiles = @("download_dataset.py", "project_status.py", "start_server.bat", "start_server.ps1")
foreach ($file in $scriptFiles) {
    if (Test-Path $file) {
        Copy-Item $file "scripts\$file" -Force
        Write-Host "  $file -> scripts\$file" -ForegroundColor Gray
    }
}

Write-Host "`n" + "=" * 80 -ForegroundColor Cyan
Write-Host "  RESTRUCTURING COMPLETE!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan

Write-Host "`nNew structure:" -ForegroundColor Yellow
Write-Host @"
DL-PROJECT/
├── src/
│   ├── models/          # model1.py, model2.py
│   ├── data/            # dataset.py
│   ├── training/        # train_model1.py, train_model2.py
│   ├── evaluation/      # visualize.py, compare_models.py
│   └── utils/           # helpers
├── webapp/              # app.py, templates, static
├── docs/                # All .md files
├── tests/               # Unit tests
├── configs/             # Configuration files
├── scripts/             # Automation scripts
└── data/                # Dataset (existing)
"@ -ForegroundColor Gray
