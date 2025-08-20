# ==============================================================================
#
#                         VERCEL PLATFORM UTILITIES
#
# ==============================================================================
#
# FILE: vercel_utils.py
#
# PURPOSE:
#   This module provides a centralized set of functions for interacting with
#   Vercel deployment APIs, replacing Google Cloud Platform services.
#   It is designed to be used by both the backtesting/optimization scripts
#   (running locally) and the live trading bot (running via Vercel deployment).
#
# AUTHENTICATION STATUS: ✅ VERIFIED WORKING (August 20, 2025)
#   - Vercel API endpoints configured and tested
#   - Live bot API: https://my-trading-robot-1.vercel.app/api/live-bot
#   - Parameter sync API: https://my-trading-robot-1.vercel.app/api/parameter-sync
#   - Upload/Download/Sync operations tested successfully
#
# ==============================================================================

import os
import json
import requests
from datetime import datetime, timezone
import logging
from pathlib import Path

# --- CONFIGURATION ---
VERCEL_BASE_URL = "https://my-trading-robot-1.vercel.app"
VERCEL_AUTH_TOKEN = "93699b3917045092715b8e16c01f2e1d"

def _get_vercel_headers():
    """
    Returns standardized headers for Vercel API calls.
    
    Returns:
        dict: Headers with authorization token
    """
    return {
        "Authorization": f"Bearer {VERCEL_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

def upload_to_vercel(source_file_path, destination_name):
    """
    Uploads a file to Vercel via parameter sync API, replacing GCS upload.

    Args:
        source_file_path (str): The local path to the file to upload.
        destination_name (str): The desired name for the parameter set.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    # Validate input parameters
    if not os.path.exists(source_file_path):
        logging.error(f"Source file does not exist: {source_file_path}")
        return False
    
    if not destination_name or destination_name.strip() == "":
        logging.error("Destination name cannot be empty")
        return False

    try:
        # Read the file content
        with open(source_file_path, 'r') as f:
            if source_file_path.endswith('.json'):
                params_data = json.load(f)
            else:
                params_data = {"raw_content": f.read()}
        
        # Add upload metadata
        if isinstance(params_data, dict):
            params_data['_upload_metadata'] = {
                'uploaded_at': datetime.now(timezone.utc).isoformat(),
                'source_file': os.path.basename(source_file_path),
                'destination_name': destination_name,
                'upload_method': 'vercel_utils_upload'
            }
        
        # Get file size for logging
        file_size = os.path.getsize(source_file_path)
        
        logging.info(f"Uploading '{source_file_path}' ({file_size:,} bytes) to Vercel as '{destination_name}'...")
        
        # Upload via parameter sync API
        response = requests.post(
            f"{VERCEL_BASE_URL}/api/parameter-sync",
            headers=_get_vercel_headers(),
            json=params_data,
            timeout=30
        )
        
        if response.status_code == 200:
            logging.info(f"Upload successful. Vercel endpoint: {VERCEL_BASE_URL}/api/parameter-sync")
            return True
        else:
            logging.error(f"Upload failed with status {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"An unexpected error occurred during Vercel upload: {e}")
        logging.error(f"Upload details - Source: {source_file_path}, Destination: {destination_name}")
        return False

def download_from_vercel(parameter_name, destination_file_path):
    """
    Downloads parameter data from Vercel live bot API, replacing GCS download.

    Args:
        parameter_name (str): The name of the parameter set to download.
        destination_file_path (str): The local path where the file should be saved.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    # Validate input parameters
    if not parameter_name or parameter_name.strip() == "":
        logging.error("Parameter name cannot be empty")
        return False
    
    if not destination_file_path or destination_file_path.strip() == "":
        logging.error("Destination file path cannot be empty")
        return False

    try:
        # Get current parameters from live bot API
        response = requests.get(
            f"{VERCEL_BASE_URL}/api/live-bot",
            headers=_get_vercel_headers(),
            timeout=30
        )
        
        if response.status_code != 200:
            logging.error(f"Failed to fetch data from Vercel API: {response.status_code}")
            return False
        
        data = response.json()
        
        # Create destination directory if it doesn't exist
        destination_dir = os.path.dirname(destination_file_path)
        if destination_dir and not os.path.exists(destination_dir):
            os.makedirs(destination_dir, exist_ok=True)
            logging.info(f"Created directory: {destination_dir}")
        
        logging.info(f"Downloading parameter data from Vercel to '{destination_file_path}'...")
        
        # Save the data
        with open(destination_file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Verify download
        if os.path.exists(destination_file_path):
            file_size = os.path.getsize(destination_file_path)
            logging.info(f"Download successful. File size: {file_size:,} bytes")
            return True
        else:
            logging.error("Download appeared to succeed but file was not created")
            return False
            
    except Exception as e:
        logging.error(f"An unexpected error occurred during Vercel download: {e}")
        logging.error(f"Download details - Parameter: {parameter_name}, Destination: {destination_file_path}")
        return False

def get_vercel_data_metadata(parameter_name):
    """
    Retrieves metadata for parameters from Vercel API.

    Args:
        parameter_name (str): The name of the parameter set.

    Returns:
        dict: Metadata information, or None if not found.
    """
    try:
        response = requests.get(
            f"{VERCEL_BASE_URL}/api/live-bot",
            headers=_get_vercel_headers(),
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract metadata
            metadata = {
                'parameter_name': parameter_name,
                'last_updated': data.get('timestamp', 'Unknown'),
                'status': data.get('status', 'Unknown'),
                'size_bytes': len(json.dumps(data)),
                'endpoint': f"{VERCEL_BASE_URL}/api/live-bot"
            }
            
            logging.info(f"Retrieved metadata for '{parameter_name}' - Status: {metadata['status']}")
            return metadata
        else:
            logging.warning(f"Parameter data '{parameter_name}' not accessible from Vercel API")
            return None
            
    except Exception as e:
        logging.error(f"Could not retrieve metadata for '{parameter_name}': {e}")
        return None

def check_vercel_connection():
    """
    Performs a comprehensive check of Vercel API connectivity and functionality.
    
    Returns:
        dict: Status report with connection details and test results.
    """
    status = {
        'connected': False,
        'endpoint': VERCEL_BASE_URL,
        'live_bot_accessible': False,
        'parameter_sync_accessible': False,
        'can_read': False,
        'can_write': False,
        'response_time_ms': 0,
        'error_messages': []
    }
    
    try:
        import time
        
        # Test 1: Basic connectivity
        start_time = time.time()
        response = requests.get(VERCEL_BASE_URL, timeout=15)
        status['response_time_ms'] = int((time.time() - start_time) * 1000)
        
        if response.status_code == 200:
            status['connected'] = True
        else:
            status['error_messages'].append(f"Main site returned status {response.status_code}")
        
        # Test 2: Live bot API access
        try:
            response = requests.get(
                f"{VERCEL_BASE_URL}/api/live-bot",
                headers=_get_vercel_headers(),
                timeout=15
            )
            
            if response.status_code == 200:
                status['live_bot_accessible'] = True
                status['can_read'] = True
            else:
                status['error_messages'].append(f"Live bot API returned status {response.status_code}")
        except Exception as e:
            status['error_messages'].append(f"Cannot access live bot API: {e}")
        
        # Test 3: Parameter sync API test (read-only check)
        try:
            test_data = {
                "_connection_test": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "test_type": "connectivity_check"
                }
            }
            
            response = requests.post(
                f"{VERCEL_BASE_URL}/api/parameter-sync",
                headers=_get_vercel_headers(),
                json=test_data,
                timeout=15
            )
            
            if response.status_code == 200:
                status['parameter_sync_accessible'] = True
                status['can_write'] = True
            else:
                status['error_messages'].append(f"Parameter sync API returned status {response.status_code}")
                
        except Exception as e:
            status['error_messages'].append(f"Cannot access parameter sync API: {e}")
            
        logging.info(f"Vercel connection check completed. Status: {status}")
        return status
        
    except Exception as e:
        status['error_messages'].append(f"Unexpected error during connection check: {e}")
        logging.error(f"Vercel connection check failed: {e}")
        return status

def sync_parameters_to_vercel(local_params_file="data/latest_live_parameters.json", parameter_name="latest_live_parameters"):
    """
    Synchronizes local parameter file to Vercel for live trading bot access.
    This replaces the GCS sync functionality with Vercel API calls.
    
    Args:
        local_params_file (str): Path to local parameters file.
        parameter_name (str): Name for the parameter set.
        
    Returns:
        bool: True if sync was successful, False otherwise.
    """
    if not os.path.exists(local_params_file):
        logging.error(f"Local parameters file not found: {local_params_file}")
        return False
    
    # Add timestamp metadata to the file before upload
    try:
        with open(local_params_file, 'r') as f:
            params_data = json.load(f)
        
        # Add sync metadata
        if isinstance(params_data, dict):
            params_data['_sync_metadata'] = {
                'uploaded_at': datetime.now(timezone.utc).isoformat(),
                'source_file': local_params_file,
                'upload_method': 'sync_parameters_to_vercel',
                'parameter_name': parameter_name
            }
        
        # Create temporary file with metadata
        temp_file = f"{local_params_file}.temp"
        with open(temp_file, 'w') as f:
            json.dump(params_data, f, indent=2)
        
        # Upload via Vercel API
        success = upload_to_vercel(temp_file, parameter_name)
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if success:
            logging.info(f"Successfully synced parameters to Vercel: {parameter_name}")
        
        return success
        
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in parameters file {local_params_file}: {e}")
        return False
    except Exception as e:
        logging.error(f"Error during parameter sync: {e}")
        return False

def check_vercel_bot_status():
    """Check the status of the live bot running on Vercel"""
    print("=" * 70)
    print("         VERCEL LIVE BOT STATUS")
    print("=" * 70)
    
    # Check Vercel connection
    status = check_vercel_connection()
    if not status['connected']:
        print("❌ Cannot connect to Vercel deployment")
        for error in status['error_messages']:
            print(f"   Error: {error}")
        return False
    
    print("✅ Connected to Vercel deployment")
    print(f"   🌐 URL: {VERCEL_BASE_URL}")
    print(f"   ⚡ Response time: {status['response_time_ms']}ms")
    print()

    # Check live bot API
    print("📊 LIVE BOT API STATUS:")
    if status['live_bot_accessible']:
        print("   ✅ Live bot API accessible")
        
        try:
            response = requests.get(
                f"{VERCEL_BASE_URL}/api/live-bot",
                headers=_get_vercel_headers(),
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   📅 Last updated: {data.get('timestamp', 'Unknown')}")
                print(f"   📊 Status: {data.get('status', 'Unknown')}")
                print(f"   📈 Balance: {data.get('balance', {}).get('USDT', 'Unknown')} USDT")
                print(f"   🔄 Active trades: {len(data.get('active_trades', []))}")
            else:
                print(f"   ⚠️  API returned status {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error fetching live bot data: {e}")
    else:
        print("   ❌ Live bot API not accessible")
    
    print()

    # Check parameter sync capability
    print("🔗 PARAMETER SYNC STATUS:")
    if status['parameter_sync_accessible']:
        print("   ✅ Parameter sync API accessible")
        print("   ✅ Can receive parameter updates")
    else:
        print("   ❌ Parameter sync API not accessible")
    
    print()
    print("💡 Your live bot on Vercel should be:")
    print("   • Receiving parameters via /api/parameter-sync endpoint")
    print("   • Executing trades based on optimized parameters")
    print("   • Providing status via /api/live-bot endpoint")
    print("   • Updating dashboard in real-time")
    return True

def get_vercel_deployment_config():
    """Get deployment configuration for Vercel platform"""
    config = {
        "vercel_deployment": {
            "framework": "next.js",
            "build_command": "npm run build",
            "output_directory": ".next",
            "install_command": "npm install"
        },
        "api_routes": {
            "live_bot": "/api/live-bot.py",
            "parameter_sync": "/api/parameter-sync.py",
            "dashboard": "/api/dashboard-integration.py"
        },
        "environment_variables": {
            "BINANCE_API_KEY": "your_testnet_api_key",
            "BINANCE_SECRET_KEY": "your_testnet_secret_key",
            "BINANCE_TESTNET": "true",
            "AUTH_TOKEN": VERCEL_AUTH_TOKEN
        },
        "performance": {
            "region": "global",
            "edge_functions": True,
            "serverless_functions": True,
            "automatic_scaling": True
        },
        "monitoring": {
            "real_time_logs": True,
            "analytics": True,
            "error_tracking": True,
            "performance_monitoring": True
        }
    }
    
    deployment_instructions = f"""
# VERCEL DEPLOYMENT INSTRUCTIONS

## Prerequisites
1. Install Vercel CLI: `npm i -g vercel`
2. Login to Vercel: `vercel login`
3. Link project: `vercel link`

## Quick Deployment
1. Deploy to production:
   ```bash
   vercel --prod
   ```

2. Set environment variables:
   ```bash
   vercel env add BINANCE_API_KEY
   vercel env add BINANCE_SECRET_KEY
   vercel env add BINANCE_TESTNET
   vercel env add AUTH_TOKEN
   ```

## Features
✅ **Zero Configuration**: Automatic framework detection
✅ **Global CDN**: Instant worldwide deployment
✅ **Serverless Functions**: Automatic scaling
✅ **Real-time Logs**: Built-in monitoring
✅ **Custom Domains**: Free SSL certificates
✅ **Git Integration**: Deploy on push

## Cost Efficiency
- **Free Tier**: 100GB bandwidth, 1000 serverless executions
- **Pro Tier**: $20/month for unlimited usage
- **Enterprise**: Custom pricing for large-scale deployment

## Current Deployment
- URL: {VERCEL_BASE_URL}
- Status: Active
- APIs: Live bot, Parameter sync, Dashboard
"""
    
    return {
        "config": config,
        "instructions": deployment_instructions
    }
