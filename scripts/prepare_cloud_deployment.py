"""
Cloud Deployment Preparation Script
Prepares the trading dashboard for cloud hosting with live data APIs
"""

import os
import shutil
import json
from pathlib import Path

def prepare_cloud_deployment():
    """Prepare dashboard for cloud deployment"""
    
    print("🚀 Preparing Trading Dashboard for Cloud Deployment")
    print("=" * 60)
    
    # Create deployment directory
    deploy_dir = Path("deploy")
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    print("📁 Creating deployment structure...")
    
    # Copy dashboard files
    dashboard_source = Path("plots_output/20250817_133240/performance_report.html")
    dashboard_dest = deploy_dir / "index.html"
    
    if dashboard_source.exists():
        shutil.copy2(dashboard_source, dashboard_dest)
        print(f"✅ Copied dashboard: {dashboard_source} -> {dashboard_dest}")
        
        # Update API endpoints for cloud
        update_api_endpoints(dashboard_dest)
    else:
        print(f"❌ Dashboard not found: {dashboard_source}")
    
    # Copy API files
    api_dir = deploy_dir / "api"
    api_dir.mkdir()
    shutil.copy2("api/live-data.py", api_dir / "live-data.py")
    print("✅ Copied API endpoint")
    
    # Copy configuration files
    for config_file in ["vercel.json", "package.json"]:
        if Path(config_file).exists():
            shutil.copy2(config_file, deploy_dir / config_file)
            print(f"✅ Copied {config_file}")
    
    # Create README for deployment
    create_deployment_readme(deploy_dir)
    
    print("\n🎉 Cloud deployment preparation complete!")
    print(f"📂 Deployment files ready in: {deploy_dir.absolute()}")
    print("\n🌐 Next steps:")
    print("1. Create GitHub repository")
    print("2. Push deploy/ folder contents to GitHub")
    print("3. Connect Vercel to your GitHub repo")
    print("4. Deploy with one click!")

def update_api_endpoints(html_file):
    """Update dashboard to use cloud API endpoints"""
    
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace local data URLs with API endpoints
    replacements = {
        "'../../data/live_bot_state.json'": "'/api/live-data'",
        "'../../live_trading/health_history.json'": "'/api/health-data'",
        "'../../data/trading_journal.json'": "'/api/journal-data'"
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Add cloud deployment indicator
    cloud_indicator = """
    <script>
    // Cloud deployment indicator
    console.log('🌐 Running on cloud deployment');
    window.CLOUD_DEPLOYMENT = true;
    </script>
    """
    
    # Insert before closing body tag
    content = content.replace('</body>', cloud_indicator + '\n</body>')
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Updated API endpoints for cloud deployment")

def create_deployment_readme(deploy_dir):
    """Create README for deployment instructions"""
    
    readme_content = """# Trading Dashboard - Cloud Deployment

## 🚀 Quick Deploy to Vercel

1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/trading-dashboard.git
   git push -u origin main
   ```

2. **Deploy to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Click "Deploy"
   - Your dashboard will be live in minutes!

## 🌐 Live URLs

After deployment, your dashboard will be available at:
- **Dashboard**: `https://your-project.vercel.app/`
- **Live Data API**: `https://your-project.vercel.app/api/live-data`

## 📊 Features

- ✅ Real-time BTC price updates
- ✅ Live trading metrics
- ✅ Professional UI/UX
- ✅ Mobile responsive
- ✅ Auto-refresh every 5 seconds
- ✅ Global CDN (fast worldwide)
- ✅ Free SSL certificate
- ✅ Custom domain support

## 🔧 Configuration

The dashboard automatically detects cloud deployment and uses:
- Binance API for live price data
- Serverless functions for data processing
- Global CDN for fast loading

## 📱 Access

Once deployed, you can access your dashboard from:
- Any computer
- Mobile devices
- Tablets
- Even when your local computer is off!

## 🆓 Cost

- **Vercel Free Tier**: Unlimited static deployments
- **Bandwidth**: 100GB/month (more than enough)
- **Functions**: 12GB-hours/month serverless compute
- **Domains**: Free .vercel.app subdomain + custom domain support

Perfect for personal trading dashboards! 🎯
"""
    
    with open(deploy_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ Created deployment README")

if __name__ == "__main__":
    prepare_cloud_deployment()
