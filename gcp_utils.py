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
# ==============================================================================

import os
from google.cloud import storage
from google.api_core import exceptions
import logging

# --- CONFIGURATION ---
# It's recommended to set the GOOGLE_APPLICATION_CREDENTIALS environment variable
# to point to your service account key file.
# The bucket name should be globally unique.
GCS_BUCKET_NAME = "trading-robot-cloud-storage-bucket-params"

def _get_gcs_client():
    """Initializes and returns a GCS client."""
    try:
        # The client will automatically use the credentials from the environment variable.
        client = storage.Client()
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Google Cloud Storage client: {e}")
        logging.error("Please ensure you have authenticated correctly (e.g., by setting the GOOGLE_APPLICATION_CREDENTIALS environment variable).")
        return None

def upload_to_gcs(source_file_path, destination_blob_name):
    """
    Uploads a file to the specified Google Cloud Storage bucket.

    Args:
        source_file_path (str): The local path to the file to upload.
        destination_blob_name (str): The desired name of the file in the GCS bucket.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    client = _get_gcs_client()
    if not client:
        return False

    try:
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        
        logging.info(f"Uploading '{source_file_path}' to GCS bucket '{GCS_BUCKET_NAME}' as '{destination_blob_name}'...")
        blob.upload_from_filename(source_file_path)
        logging.info("Upload successful.")
        return True
    except exceptions.NotFound:
        logging.error(f"GCS Error: The bucket '{GCS_BUCKET_NAME}' does not exist.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during GCS upload: {e}")
        return False

def download_from_gcs(source_blob_name, destination_file_path):
    """
    Downloads a file from the specified Google Cloud Storage bucket.

    Args:
        source_blob_name (str): The name of the file in the GCS bucket.
        destination_file_path (str): The local path where the file should be saved.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    client = _get_gcs_client()
    if not client:
        return False

    try:
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(source_blob_name)
        
        logging.info(f"Downloading '{source_blob_name}' from GCS bucket '{GCS_BUCKET_NAME}' to '{destination_file_path}'...")
        blob.download_to_filename(destination_file_path)
        logging.info("Download successful.")
        return True
    except exceptions.NotFound:
        logging.error(f"GCS Error: The blob '{source_blob_name}' was not found in the bucket '{GCS_BUCKET_NAME}'.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during GCS download: {e}")
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
        return blob
    except Exception as e:
        logging.error(f"Could not retrieve metadata for blob '{blob_name}': {e}")
        return None