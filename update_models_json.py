#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to update the main ollama_models.json file with fresh data from the scraper.
This ensures that the JSON file contains all the expected tags and model information.
"""
import subprocess
import os
import json
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("update_models.log"),
        logging.StreamHandler()
    ]
)

def run_scraper():
    """Run the ollama_scraper.py script to get fresh model data"""
    logging.info("Running ollama_scraper.py to get fresh model data...")
    
    try:
        # Use a temporary file to avoid overwriting the main file until we validate
        temp_output = "temp_models_output.json"
        cmd = [sys.executable, "ollama_scraper.py", temp_output]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"Scraper output: {result.stdout}")
        
        if result.stderr:
            logging.warning(f"Scraper warnings/errors: {result.stderr}")
        
        return temp_output
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to run scraper: {e}")
        if e.stdout:
            logging.error(f"Scraper stdout: {e.stdout}")
        if e.stderr:
            logging.error(f"Scraper stderr: {e.stderr}")
        return None

def validate_data(file_path):
    """Validate the scraped data to ensure it has models and tags"""
    logging.info(f"Validating data in {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        model_count = len(data)
        tag_count = sum(len(model.get("tags", [])) for model in data)
        
        logging.info(f"Found {model_count} models with {tag_count} tags")
        
        # Check if we have a reasonable number of models and tags
        if model_count < 10:
            logging.error(f"Too few models found ({model_count}). Expected at least 10.")
            return False
            
        if tag_count < 100:
            logging.error(f"Too few tags found ({tag_count}). Expected at least 100.")
            return False
            
        # Check if most models have tags
        models_with_tags = sum(1 for model in data if len(model.get("tags", [])) > 0)
        if models_with_tags < model_count * 0.8:  # At least 80% of models should have tags
            logging.error(f"Only {models_with_tags}/{model_count} models have tags. Most models should have tags.")
            return False
        
        return True
    except Exception as e:
        logging.error(f"Error validating data: {e}")
        return False

def update_main_file(temp_file, main_file="ollama_models.json"):
    """Update the main JSON file with validated new data"""
    logging.info(f"Updating {main_file} with data from {temp_file}...")
    
    try:
        # Create a backup of the existing file
        if os.path.exists(main_file):
            backup_file = f"{main_file}.bak"
            with open(main_file, 'r') as src, open(backup_file, 'w') as dst:
                dst.write(src.read())
            logging.info(f"Created backup at {backup_file}")
        
        # Copy the temp file to the main file
        with open(temp_file, 'r') as src, open(main_file, 'w') as dst:
            dst.write(src.read())
        
        logging.info(f"Successfully updated {main_file}")
        return True
    except Exception as e:
        logging.error(f"Error updating main file: {e}")
        return False

def cleanup(temp_file):
    """Clean up temporary files"""
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logging.info(f"Removed temporary file {temp_file}")
    except Exception as e:
        logging.warning(f"Error removing temporary file: {e}")

def main():
    """Main function to update the ollama_models.json file"""
    logging.info("Starting models update process...")
    
    # Run the scraper to get fresh data
    temp_file = run_scraper()
    if not temp_file or not os.path.exists(temp_file):
        logging.error("Failed to get fresh model data. Update aborted.")
        return 1
    
    # Validate the data
    if not validate_data(temp_file):
        logging.error("Data validation failed. Update aborted.")
        cleanup(temp_file)
        return 1
    
    # Update the main file
    if not update_main_file(temp_file):
        logging.error("Failed to update main file. Update aborted.")
        cleanup(temp_file)
        return 1
    
    # Clean up
    cleanup(temp_file)
    
    logging.info("Update process completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
