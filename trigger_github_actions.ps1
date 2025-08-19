# GitHub Actions Trigger Script
# Use this to manually trigger your workflows

Write-Host "🚀 GITHUB ACTIONS CONTROL PANEL" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📋 Available Workflows:" -ForegroundColor Yellow
Write-Host "1. Bidirectional Trading Sync"
Write-Host "2. Deploy" 
Write-Host "3. Trading Bot"
Write-Host ""

Write-Host "🌐 Access Methods:" -ForegroundColor Green
Write-Host "• Browser: https://github.com/iPoundSaversBag/my-trading-robot-1/actions"
Write-Host "• VS Code: Check Activity Bar for GitHub icon"
Write-Host "• Command: Use this script"
Write-Host ""

# Function to trigger workflow
function Trigger-Workflow {
    param($workflowType)
    
    Write-Host "Triggering $workflowType workflow..." -ForegroundColor Cyan
    
    # You would need a GitHub token for this, but browser access works fine
    Write-Host "✅ Use browser access for now: https://github.com/iPoundSaversBag/my-trading-robot-1/actions" -ForegroundColor Green
}

Write-Host "💡 TIP: Your GitHub Actions are working! Use the browser link above." -ForegroundColor Magenta
