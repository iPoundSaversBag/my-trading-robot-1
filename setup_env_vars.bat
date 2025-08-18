@echo off
echo Setting up Vercel environment variables for trading system...
echo.

echo Adding BINANCE_API_KEY...
vercel env add BINANCE_API_KEY production preview development

echo.
echo Adding BINANCE_API_SECRET...
vercel env add BINANCE_API_SECRET production preview development

echo.
echo Adding CRON_SECRET (for automated trading security)...
vercel env add CRON_SECRET production preview development

echo.
echo Environment variables setup complete!
echo Your trading system is now ready for deployment.
