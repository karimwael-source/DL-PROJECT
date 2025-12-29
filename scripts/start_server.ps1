# AI Keyframe Detection - Server Startup Script
# This script keeps the server running until you close the window

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   🚀 AI KEYFRAME DETECTION - Server Launcher" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏳ Starting server (please wait 10-15 seconds)..." -ForegroundColor Yellow
Write-Host "💡 PyTorch import takes time - DO NOT close this window!" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Run the app
& "E:\python.exe" app_launcher.py

Write-Host ""
Write-Host "Server stopped." -ForegroundColor Red
Write-Host "Press any key to close this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
