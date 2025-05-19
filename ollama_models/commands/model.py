"""
Model management commands for the ollama-models CLI.
"""
import os
import json
import logging
import requests
from ollama_models.core import scraper
from ollama_models.config import DEFAULT_CONFIG_FILE, DEFAULT_MODELS_JSON

logger = logging.getLogger("ollama_models.model")

def setup_parser(parser):
    """
    Set up the argument parser for the model command group.
    
    Args:
        parser: The argument parser to set up
    """
    subparsers = parser.add_subparsers(dest="subcommand", help="Model subcommands")
    
    # fetch command (was update_models.py)
    fetch_parser = subparsers.add_parser("fetch", help="Fetch and update the model database")
    fetch_parser.add_argument("--output", "-o", default=DEFAULT_MODELS_JSON,
                            help=f"Output JSON file (default: {DEFAULT_MODELS_JSON})")
    fetch_parser.add_argument("--skip-validation", action="store_true",
                            help="Skip validation of scraped data")
    fetch_parser.add_argument("--force", "-f", action="store_true",
                            help="Force update even if validation fails")
    
    # edit command (was tag_selector.py)
    edit_parser = subparsers.add_parser("edit", help="Edit selected model tags")
    edit_parser.add_argument("--models-file", "-m", default=DEFAULT_MODELS_JSON,
                           help=f"Models JSON file (default: {DEFAULT_MODELS_JSON})")
    edit_parser.add_argument("--config-file", "-c", default=DEFAULT_CONFIG_FILE,
                           help=f"Selected tags config file (default: {DEFAULT_CONFIG_FILE})")
    
    # apply command (was sync_models.py)
    apply_parser = subparsers.add_parser("apply", help="Apply selected model configuration to Ollama")
    apply_parser.add_argument("--config-file", "-c", default=DEFAULT_CONFIG_FILE,
                            help=f"Selected tags config file (default: {DEFAULT_CONFIG_FILE})")
    
    # init command (was ollama_api_updater.py)
    init_parser = subparsers.add_parser("init", help="Initialize selected tags from Ollama API")
    init_parser.add_argument("--config-file", "-c", default=DEFAULT_CONFIG_FILE,
                           help=f"Selected tags config file (default: {DEFAULT_CONFIG_FILE})")

def handle_command(args):
    """
    Handle model commands.
    
    Args:
        args: Command arguments
        
    Returns:
        int: Exit code
    """
    if args.subcommand == "fetch":
        return cmd_fetch(args)
    elif args.subcommand == "edit":
        return cmd_edit(args)
    elif args.subcommand == "apply":
        return cmd_apply(args)
    elif args.subcommand == "init":
        return cmd_init(args)
    else:
        logger.error("No subcommand specified")
        return 1

def cmd_fetch(args):
    """
    Implement the model fetch command (former update_models.py).
    
    Args:
        args: Command arguments
        
    Returns:
        int: Exit code
    """
    logger.info("Starting models fetch process...")
    
    # Run the scraper to get fresh data
    temp_file = run_scraper(args.output + ".temp")
    if not temp_file or not os.path.exists(temp_file):
        logger.error("Failed to get fresh model data. Update aborted.")
        return 1
    
    # Validate the data
    if not args.skip_validation:
        validation_passed = validate_data(temp_file)
        if not validation_passed and not args.force:
            logger.error("Data validation failed. Use --force to update anyway.")
            logger.info("Update aborted.")
            cleanup(temp_file)
            return 1
        if not validation_passed:
            logger.warning("Validation failed but --force flag is set, continuing anyway.")
    
    # Update the main file
    if not update_main_file(temp_file, args.output):
        logger.error("Failed to update main file. Update aborted.")
        cleanup(temp_file)
        return 1
    
    # Clean up
    cleanup(temp_file)
    
    logger.info("Update process completed successfully")
    return 0

def run_scraper(temp_output):
    """Run the Ollama scraper to get fresh model data"""
    logger.info("Running Ollama scraper to get fresh model data...")
    
    try:
        # Use the integrated scraper module
        from ollama_models.core.scraper import scrape_and_save
        
        # Run the scraper
        scrape_and_save(temp_output)
        
        return temp_output
    except Exception as e:
        logger.error(f"Failed to run scraper: {e}", exc_info=True)
        return None

def validate_data(file_path):
    """
    Validate the scraped data to ensure it has models and tags.
    
    Current validation criteria has been adjusted for modern Ollama website structure.
    """
    logger.info(f"Validating data in {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        model_count = len(data)
        tag_count = sum(len(model.get("tags", [])) for model in data)
        
        logger.info(f"Found {model_count} models with {tag_count} tags")
        
        # Check if we have a reasonable number of models
        if model_count < 10:
            logger.error(f"Too few models found ({model_count}). Expected at least 10.")
            return False
            
        # Less strict tag count requirement - the Ollama website structure has changed
        if tag_count < 10:
            logger.warning(f"Very few tags found ({tag_count}). The Ollama website structure may have changed.")
            return False
            
        # Check if at least some models have tags - we've relaxed this requirement
        models_with_tags = sum(1 for model in data if len(model.get("tags", [])) > 0)
        if models_with_tags == 0:
            logger.error(f"No models have tags. This suggests a scraping issue.")
            return False
        
        logger.info(f"Models with tags: {models_with_tags}/{model_count} ({models_with_tags/model_count:.1%})")
        return True
    except Exception as e:
        logger.error(f"Error validating data: {e}", exc_info=True)
        return False

def update_main_file(temp_file, main_file):
    """Update the main JSON file with validated new data"""
    logger.info(f"Updating {main_file} with data from {temp_file}...")
    
    try:
        # Create a backup of the existing file
        if os.path.exists(main_file):
            backup_file = f"{main_file}.bak"
            with open(main_file, 'r') as src, open(backup_file, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Created backup at {backup_file}")
          # Load the temp file data
        with open(temp_file, 'r') as f:
            models_data = json.load(f)
        
        # Write the updated data to the main file
        with open(main_file, 'w') as f:
            json.dump(models_data, f, indent=2)
        
        logger.info(f"Successfully updated {main_file}")
        return True
    except Exception as e:
        logger.error(f"Error updating main file: {e}", exc_info=True)
        return False

def cleanup(temp_file):
    """Clean up temporary files"""
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logger.info(f"Removed temporary file {temp_file}")
    except Exception as e:
        logger.warning(f"Error removing temporary file: {e}")

def cmd_edit(args):
    """
    Implement the model edit command (former tag_selector.py).
    
    Args:
        args: Command arguments
        
    Returns:
        int: Exit code
    """
    try:
        from ollama_models.core.tag_selector import run_selector
        
        # Run the tag selector with the provided arguments
        return run_selector(args.models_file, args.config_file)
    except Exception as e:
        logger.error(f"Failed to run tag selector: {e}", exc_info=True)
        return 1

def cmd_apply(args):
    """
    Implement the model apply command (former sync_models.py).
    
    Args:
        args: Command arguments
        
    Returns:
        int: Exit code
    """
    from ollama_models.utils import API_BASE
    from ollama_models.core.syncer import sync_ollama
    
    logger.info("Syncing models with Ollama...")
    
    # Use the integrated syncer module
    success, new_models, removed_models = sync_ollama(args.config_file, API_BASE)
    
    if not success:
        logger.warning("Some operations failed during sync")
        
    return 0 if success else 1

def load_config(config_file):
    """Load selected models from config file"""
    selected = set()
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    selected.add(line)
    return selected

def cmd_init(args):
    """
    Implement the model init command (former ollama_api_updater.py).
    
    Args:
        args: Command arguments
        
    Returns:
        int: Exit code
    """
    from ollama_models.utils import API_BASE
    from ollama_models.core.initializer import init_from_api
    
    # Use the integrated initializer module
    success, models = init_from_api(args.config_file, API_BASE)
    
    return 0 if success else 1
