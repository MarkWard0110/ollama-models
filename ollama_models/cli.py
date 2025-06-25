#!/usr/bin/env python3
"""
Main CLI entry point for the Ollama Models application.
"""
import argparse
import sys
import os
import logging
from ollama_models import __version__
from ollama_models.commands import model, context
from ollama_models.config import DEFAULT_API_BASE, DEFAULT_CONFIG_FILE

def setup_logging(verbose=False):
    """Configure logging for the application"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ollama_models")

def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        prog="ollama-models",
        description="Tools for managing and analyzing Ollama models.",
    )
    parser.add_argument(
        "--version", action="version", 
        version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help=f"Path to configuration file for Ollama host (default: ollama_models.conf)"
    )
    parser.add_argument(
        "--host-config", type=str, default=None,
        help=f"Path to Ollama host configuration file (default: ollama_host.conf)"
    )
    parser.add_argument(
        "--api", "-a", type=str, default=None,
        help=f"Ollama API base URL (default: from host config file or {DEFAULT_API_BASE})"
    )
    
    # Create subparsers for our command groups
    subparsers = parser.add_subparsers(dest="command_group", help="Command group")
    
    # Add model commands
    model_parser = subparsers.add_parser("model", help="Model management commands")
    model.setup_parser(model_parser)
    
    # Add context commands
    context_parser = subparsers.add_parser("context", help="Context size analysis commands")
    context.setup_parser(context_parser)
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    # Check for mutually exclusive --api and --host-config
    if args.api and args.host_config:
        logger.error("--api and --host-config are mutually exclusive. Please specify only one.")
        return 1

    # Determine API base URL
    api_base = DEFAULT_API_BASE
    if args.api:
        api_base = args.api
    elif args.host_config:
        from ollama_models.config import load_api_base_from_config
        config_api = load_api_base_from_config(args.host_config)
        if config_api:
            api_base = config_api
    else:
        from ollama_models.config import DEFAULT_HOST_CONFIG_FILE, load_api_base_from_config
        if os.path.isfile(DEFAULT_HOST_CONFIG_FILE):
            config_api = load_api_base_from_config(DEFAULT_HOST_CONFIG_FILE)
            if config_api:
                api_base = config_api
    # If none of the above, api_base remains DEFAULT_API_BASE

    # Import utils and set the API base URL
    from ollama_models.utils import set_api_base
    set_api_base(api_base)
    
    # Dispatch to the appropriate command handler
    try:
        if args.command_group == "model":
            return model.handle_command(args)
        elif args.command_group == "context":
            return context.handle_command(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
