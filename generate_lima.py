#!/usr/bin/env python3
"""
Script to download the LIMA dataset from Hugging Face and save it as a parquet file.

The LIMA dataset is gated and requires authentication with Hugging Face.
You must be logged in with a Hugging Face account that has accepted the dataset terms.
"""

import os
import sys
from datasets import load_dataset
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.utils import HfHubHTTPError

def check_login_status():
    """Check if the user is logged in to Hugging Face."""
    token = HfFolder.get_token()
    if token is None:
        print("You are not logged in to Hugging Face.")
        print("Please run the following command to login:")
        print("    huggingface-cli login")
        print("Or use:")
        print("    python -c 'from huggingface_hub import login; login()'")
        return False
    
    # Verify token is valid
    try:
        api = HfApi()
        user_info = api.whoami(token=token)
        print(f"Logged in as: {user_info['name']} ({user_info.get('email', 'No email')})")
        return True
    except Exception as e:
        print(f"Error verifying login: {e}")
        return False

def main():
    """Download the LIMA dataset and save it as parquet files."""
    print("Checking Hugging Face login status...")
    if not check_login_status():
        sys.exit(1)
    
    print("Downloading LIMA dataset from Hugging Face...")
    try:
        # Load the dataset
        ds = load_dataset("GAIR/lima")
        
        # Create output directory if it doesn't exist
        output_dir = "lima_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each split as a parquet file
        for split in ds:
            output_path = os.path.join(output_dir, f"lima_{split}.parquet")
            print(f"Saving {split} split to {output_path}...")
            ds[split].to_parquet(output_path)
        
        print(f"Dataset successfully saved to {output_dir}/ directory")
        
    except HfHubHTTPError as e:
        if "403 Forbidden" in str(e):
            print("\nERROR: Access denied to the LIMA dataset.")
            print("This dataset requires accepting its terms of use on the Hugging Face website.")
            print("Please visit: https://huggingface.co/datasets/GAIR/lima")
            print("Log in, accept the terms, then try again.")
        else:
            print(f"\nERROR: Failed to download dataset: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
