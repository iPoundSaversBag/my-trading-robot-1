# ==============================================================================
#
#                         GOOGLE CLOUD PLATFORM UTILITIES
#
# ==============================================================================
#
# FILE: gcp_utils.py
#
# PURPOSE:
#   This module provides a centralized set of functions for interacting with
#   Google Cloud Platform services, specifically Google Cloud Storage (GCS).
#   It is designed to be used by both the backtesting/optimization scripts
#   (running locally) and the live trading bot (running on a GCP VM).
#
# AUTHENTICATION STATUS: ✅ VERIFIED WORKING (July 20, 2025)
#   - Application Default Credentials configured
#   - Bucket access confirmed: trading-robot-cloud-storage-bucket-params
#   - Upload/Download/Metadata operations tested successfully
#
# ==============================================================================

import os
import json
from datetime import datetime, timezone
from google.cloud import storage
from google.api_core import exceptions
import logging

# --- CONFIGURATION ---
# It's recommended to set the GOOGLE_APPLICATION_CREDENTIALS environment variable
# to point to your service account key file.
# The bucket name should be globally unique.
GCS_BUCKET_NAME = "trading-robot-cloud-storage-bucket-params"

def _get_gcs_client():
    """
    Initializes and returns a GCS client with enhanced error handling.
    
    Returns:
        storage.Client: Authenticated GCS client, or None if authentication fails.
    """
    try:
        # The client will automatically use Application Default Credentials
        client = storage.Client()
        # Test the client by getting the project info
        project = client.project
        logging.info(f"Successfully initialized GCS client for project: {project}")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Google Cloud Storage client: {e}")
        logging.error("Authentication troubleshooting:")
        logging.error("1. Run: gcloud auth application-default login")
        logging.error("2. Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        logging.error("3. Ensure you have Storage permissions in the project")
        return None

def upload_to_gcs(source_file_path, destination_blob_name):
    """
    Uploads a file to the specified Google Cloud Storage bucket with enhanced error handling.

    Args:
        source_file_path (str): The local path to the file to upload.
        destination_blob_name (str): The desired name of the file in the GCS bucket.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    # Validate input parameters
    if not os.path.exists(source_file_path):
        logging.error(f"Source file does not exist: {source_file_path}")
        return False
    
    if not destination_blob_name or destination_blob_name.strip() == "":
        logging.error("Destination blob name cannot be empty")
        return False

    client = _get_gcs_client()
    if not client:
        return False

    try:
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        
        # Get file size for logging
        file_size = os.path.getsize(source_file_path)
        
        logging.info(f"Uploading '{source_file_path}' ({file_size:,} bytes) to GCS bucket '{GCS_BUCKET_NAME}' as '{destination_blob_name}'...")
        
        # Upload with metadata
        blob.metadata = {
            'uploaded_at': datetime.now(timezone.utc).isoformat(),
            'source_file': os.path.basename(source_file_path),
            'file_size_bytes': str(file_size)
        }
        
        blob.upload_from_filename(source_file_path)
        logging.info(f"Upload successful. Blob URL: gs://{GCS_BUCKET_NAME}/{destination_blob_name}")
        return True
        
    except exceptions.NotFound:
        logging.error(f"GCS Error: The bucket '{GCS_BUCKET_NAME}' does not exist.")
        logging.error("Please check the bucket name and ensure you have access to it.")
        return False
    except exceptions.Forbidden:
        logging.error(f"GCS Error: Permission denied. You don't have write access to bucket '{GCS_BUCKET_NAME}'.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during GCS upload: {e}")
        logging.error(f"Upload details - Source: {source_file_path}, Destination: {destination_blob_name}")
        return False

def download_from_gcs(source_blob_name, destination_file_path):
    """
    Downloads a file from the specified Google Cloud Storage bucket with enhanced error handling.

    Args:
        source_blob_name (str): The name of the file in the GCS bucket.
        destination_file_path (str): The local path where the file should be saved.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    # Validate input parameters
    if not source_blob_name or source_blob_name.strip() == "":
        logging.error("Source blob name cannot be empty")
        return False
    
    if not destination_file_path or destination_file_path.strip() == "":
        logging.error("Destination file path cannot be empty")
        return False

    client = _get_gcs_client()
    if not client:
        return False

    try:
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(source_blob_name)
        
        # Check if blob exists
        if not blob.exists():
            logging.error(f"GCS Error: The blob '{source_blob_name}' does not exist in bucket '{GCS_BUCKET_NAME}'.")
            return False
        
        # Create destination directory if it doesn't exist
        destination_dir = os.path.dirname(destination_file_path)
        if destination_dir and not os.path.exists(destination_dir):
            os.makedirs(destination_dir, exist_ok=True)
            logging.info(f"Created directory: {destination_dir}")
        
        logging.info(f"Downloading '{source_blob_name}' from GCS bucket '{GCS_BUCKET_NAME}' to '{destination_file_path}'...")
        blob.download_to_filename(destination_file_path)
        
        # Verify download
        if os.path.exists(destination_file_path):
            file_size = os.path.getsize(destination_file_path)
            logging.info(f"Download successful. File size: {file_size:,} bytes")
            return True
        else:
            logging.error("Download appeared to succeed but file was not created")
            return False
            
    except exceptions.NotFound:
        logging.error(f"GCS Error: The bucket '{GCS_BUCKET_NAME}' or blob '{source_blob_name}' was not found.")
        return False
    except exceptions.Forbidden:
        logging.error(f"GCS Error: Permission denied. You don't have read access to bucket '{GCS_BUCKET_NAME}'.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during GCS download: {e}")
        logging.error(f"Download details - Source: {source_blob_name}, Destination: {destination_file_path}")
        return False

def get_gcs_blob_metadata(blob_name):
    """
    Retrieves metadata for a specific blob in GCS, like its update time.

    Args:
        blob_name (str): The name of the file (blob) in the GCS bucket.

    Returns:
        storage.Blob: The blob object with its metadata, or None if not found.
    """
    client = _get_gcs_client()
    if not client:
        return None
        
    try:
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.get_blob(blob_name)
        
        if blob:
            logging.info(f"Retrieved metadata for blob '{blob_name}' - Size: {blob.size} bytes, Updated: {blob.updated}")
        else:
            logging.warning(f"Blob '{blob_name}' not found in bucket '{GCS_BUCKET_NAME}'")
            
        return blob
    except Exception as e:
        logging.error(f"Could not retrieve metadata for blob '{blob_name}': {e}")
        return None


def list_bucket_contents(prefix=None, max_results=100):
    """
    Lists all blobs in the GCS bucket, optionally filtered by prefix.

    Args:
        prefix (str, optional): Filter blobs that start with this prefix.
        max_results (int): Maximum number of results to return.

    Returns:
        list: List of blob names, or empty list if error.
    """
    client = _get_gcs_client()
    if not client:
        return []

    try:
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)
        
        blob_names = [blob.name for blob in blobs]
        logging.info(f"Found {len(blob_names)} blobs in bucket '{GCS_BUCKET_NAME}'" + 
                    (f" with prefix '{prefix}'" if prefix else ""))
        return blob_names
        
    except Exception as e:
        logging.error(f"Could not list bucket contents: {e}")
        return []


def check_gcs_connection():
    """
    Performs a comprehensive check of GCS connectivity and permissions.
    
    Returns:
        dict: Status report with connection details and test results.
    """
    status = {
        'connected': False,
        'project': None,
        'bucket_accessible': False,
        'can_read': False,
        'can_write': False,
        'test_blob_exists': False,
        'error_messages': []
    }
    
    try:
        # Test 1: Client initialization
        client = _get_gcs_client()
        if not client:
            status['error_messages'].append("Failed to initialize GCS client")
            return status
            
        status['connected'] = True
        status['project'] = client.project
        
        # Test 2: Bucket access
        try:
            bucket = client.get_bucket(GCS_BUCKET_NAME)
            status['bucket_accessible'] = True
        except Exception as e:
            status['error_messages'].append(f"Cannot access bucket: {e}")
            return status
        
        # Test 3: Read permissions - check for existing blob
        try:
            blobs = list(bucket.list_blobs(max_results=1))
            status['can_read'] = True
            if blobs:
                status['test_blob_exists'] = True
        except Exception as e:
            status['error_messages'].append(f"Cannot read from bucket: {e}")
        
        # Test 4: Write permissions - try to create a test blob
        try:
            test_blob = bucket.blob('_connection_test.txt')
            test_content = f"Connection test at {datetime.now(timezone.utc).isoformat()}"
            test_blob.upload_from_string(test_content)
            status['can_write'] = True
            
            # Clean up test blob
            test_blob.delete()
            
        except Exception as e:
            status['error_messages'].append(f"Cannot write to bucket: {e}")
            
        logging.info(f"GCS connection check completed. Status: {status}")
        return status
        
    except Exception as e:
        status['error_messages'].append(f"Unexpected error during connection check: {e}")
        logging.error(f"GCS connection check failed: {e}")
        return status


def sync_parameters_to_cloud(local_params_file="data/latest_live_parameters.json", cloud_blob_name="latest_live_parameters.json"):
    """
    Synchronizes local parameter file to cloud storage for live trading bot access.
    This is the main function used by the backtesting system to update live bot parameters.
    
    Args:
        local_params_file (str): Path to local parameters file.
        cloud_blob_name (str): Name for the blob in cloud storage.
        
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
                'upload_method': 'sync_parameters_to_cloud'
            }
        
        # Create temporary file with metadata
        temp_file = f"{local_params_file}.temp"
        with open(temp_file, 'w') as f:
            json.dump(params_data, f, indent=2)
        
        # Upload the temporary file
        success = upload_to_gcs(temp_file, cloud_blob_name)
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if success:
            logging.info(f"Successfully synced parameters to cloud: {cloud_blob_name}")
        
        return success
        
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in parameters file {local_params_file}: {e}")
        return False
    except Exception as e:
        logging.error(f"Error during parameter sync: {e}")
        return False

def generate_cloud_deployment_config():
    """Generate comprehensive GCP deployment configuration for dashboard and live bot"""
    config = {
        "cloud_deployment": {
            "project_id": "your-gcp-project-id",
            "region": "us-central1",
            "zone": "us-central1-a"
        },
        "cloud_run": {
            "service_name": "trading-dashboard",
            "image": "gcr.io/your-project/trading-dashboard:latest",
            "port": 8080,
            "cpu": "1000m",
            "memory": "512Mi",
            "min_instances": 0,
            "max_instances": 10,
            "concurrency": 80,
            "timeout": "300s"
        },
        "docker_config": {
            "base_image": "python:3.11-slim",
            "requirements": [
                "streamlit>=1.28.0",
                "plotly>=5.15.0", 
                "pandas>=1.5.0",
                "numpy>=1.24.0",
                "google-cloud-storage>=2.10.0"
            ],
            "dockerfile": '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["streamlit", "run", "dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"]
'''
        },
        "authentication": {
            "method": "iap",  # Identity Aware Proxy
            "allowed_users": ["user@your-domain.com"],
            "allowed_domains": ["your-domain.com"]
        },
        "networking": {
            "vpc_connector": "projects/your-project/locations/us-central1/connectors/trading-vpc",
            "egress": "all-traffic"
        },
        "monitoring": {
            "enable_cloud_logging": True,
            "enable_cloud_monitoring": True,
            "log_level": "INFO",
            "metrics_collection": True
        },
        "environment_variables": {
            "GOOGLE_APPLICATION_CREDENTIALS": "/app/service-account.json",
            "GCS_BUCKET_NAME": GCS_BUCKET_NAME,
            "STREAMLIT_SERVER_HEADLESS": "true",
            "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "none"
        },
        "cost_optimization": {
            "auto_scaling": {
                "target_cpu_utilization": 70,
                "min_replicas": 0,
                "max_replicas": 5
            },
            "resource_limits": {
                "cpu": "500m",
                "memory": "256Mi"
            },
            "pricing_tier": "standard"
        }
    }
    
    deployment_instructions = f"""
# GCP DEPLOYMENT INSTRUCTIONS

## Prerequisites
1. Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install
2. Authenticate: `gcloud auth login`
3. Set project: `gcloud config set project YOUR_PROJECT_ID`
4. Enable APIs:
   ```bash
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable run.googleapis.com
   gcloud services enable storage.googleapis.com
   ```

## Quick Deployment
1. Build and deploy:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/trading-dashboard
   gcloud run deploy trading-dashboard \\
     --image gcr.io/YOUR_PROJECT_ID/trading-dashboard \\
     --platform managed \\
     --region us-central1 \\
     --allow-unauthenticated \\
     --port 8080 \\
     --memory 512Mi \\
     --cpu 1000m \\
     --min-instances 0 \\
     --max-instances 10
   ```

## Security Setup (Recommended)
1. Enable Identity Aware Proxy:
   ```bash
   gcloud iap web enable --resource-type=cloud-run-rev \\
     --oauth2-client-id=YOUR_OAUTH_CLIENT_ID \\
     --oauth2-client-secret=YOUR_OAUTH_CLIENT_SECRET
   ```

2. Add authorized users:
   ```bash
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \\
     --member="user:your-email@domain.com" \\
     --role="roles/iap.httpsResourceAccessor"
   ```

## Cost Monitoring
- Set up billing alerts
- Monitor Cloud Run costs in billing dashboard
- Configure auto-scaling to minimize costs
- Use Cloud Run's scale-to-zero feature

## Troubleshooting
- Check logs: `gcloud logging read "resource.type=cloud_run_revision"`
- Monitor metrics: Google Cloud Console → Cloud Run → Service → Metrics
- Debug builds: `gcloud builds log BUILD_ID`
"""
    
    return {
        "config": config,
        "instructions": deployment_instructions
    }

def get_dashboard_deployment_config():
    """Get dashboard deployment configuration for Cloud Run"""
    return generate_cloud_deployment_config()

def check_gcp_bot_status():
    """Check the status of the live bot running in Google Cloud"""
    print("=" * 70)
    print("         GOOGLE CLOUD PLATFORM LIVE BOT STATUS")
    print("=" * 70)
    
    # Check GCS connection
    client = _get_gcs_client()
    if not client:
        print("❌ Cannot connect to Google Cloud Storage")
        return False
    
    print("✅ Connected to Google Cloud Storage")
    print(f"   📦 Project: {client.project}")
    print()

    # Check for live bot parameters
    print("📊 LIVE BOT PARAMETERS:")
    try:
        blob = get_gcs_blob_metadata("latest_live_parameters.json")
        if blob:
            print(f"   ✅ Parameters file found")
            print(f"   📅 Last updated: {blob.updated}")
            print(f"   📏 Size: {blob.size:,} bytes")
            
            # Download and check parameter details
            if download_from_gcs("latest_live_parameters.json", "temp_params.json"):
                import json
                import os
                from datetime import datetime
                
                with open("temp_params.json", 'r') as f:
                    params = json.load(f)
                
                # Count windows and check metadata
                window_keys = [k for k in params.keys() if k.startswith('Window_')]
                print(f"   🔢 Optimization windows: {len(window_keys)}")
                
                if window_keys:
                    latest_window = sorted(window_keys, key=lambda w: int(w.split('_')[1]))[-1]
                    print(f"   🎯 Latest window: {latest_window}")
                
                # Check sync metadata
                if '_sync_metadata' in params:
                    sync_meta = params['_sync_metadata']
                    print(f"   🔄 Sync method: {sync_meta.get('upload_method', 'Unknown')}")
                    print(f"   📂 Source: {sync_meta.get('source_file', 'Unknown')}")
                
                os.remove("temp_params.json")
            else:
                print("   ⚠️  Could not download parameters for analysis")
        else:
            print("   ❌ Parameters file not found in GCS")
    except Exception as e:
        print(f"   ❌ Error checking parameters: {e}")
    
    print()

    # Check local vs cloud sync
    print("🔗 LOCAL-TO-CLOUD SYNC STATUS:")
    local_params = "data/latest_live_parameters.json"
    if os.path.exists(local_params):
        local_time = datetime.fromtimestamp(os.path.getmtime(local_params))
        print(f"   📁 Local parameters: {local_time}")
        
        blob = get_gcs_blob_metadata("latest_live_parameters.json")
        if blob:
            cloud_time = blob.updated.replace(tzinfo=None)
            print(f"   ☁️  Cloud parameters: {cloud_time}")
            
            time_diff = (cloud_time - local_time).total_seconds()
            if abs(time_diff) < 300:  # 5 minutes
                print("   ✅ Local and cloud are in sync")
            elif cloud_time > local_time:
                print(f"   ⬇️  Cloud is newer by {time_diff/60:.1f} minutes")
            else:
                print(f"   ⬆️  Local is newer by {-time_diff/60:.1f} minutes")
        else:
            print("   ❌ Cloud parameters not found for comparison")
    else:
        print("   ⚠️  No local parameters file found")
    
    print()
    print("💡 Your live bot in Google Cloud should be:")
    print("   • Reading parameters from GCS every 5 minutes")
    print("   • Executing trades based on optimized parameters")
    print("   • Uploading status/logs back to GCS")
    return True
