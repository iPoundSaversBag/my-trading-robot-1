# Trading Dashboard Server Launcher (PowerShell)
# Starts the HTTP server for the trading dashboard with live data support

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "   TRADING DASHBOARD SERVER" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory and navigate to project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "🔧 Starting HTTP server..." -ForegroundColor Yellow
Write-Host "📊 Dashboard will be available at: " -ForegroundColor Green -NoNewline
Write-Host "http://localhost:8080" -ForegroundColor White
Write-Host "📡 Live data will update automatically" -ForegroundColor Green
Write-Host ""
Write-Host "💡 Tip: Press Ctrl+C to stop the server" -ForegroundColor Blue
Write-Host ""

try {
    # Start the Python server
    python scripts/dashboard_server.py
}
catch {
    Write-Host "❌ Error starting server: $_" -ForegroundColor Red
}
finally {
    Write-Host ""
    Write-Host "⏹️  Server stopped." -ForegroundColor Yellow
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
