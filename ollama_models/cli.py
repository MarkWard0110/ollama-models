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
from ollama_models.config import DEFAULT_API_BASE

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
        "--api", "-a", type=str, default=DEFAULT_API_BASE,
        help=f"Ollama API base URL (default: {DEFAULT_API_BASE})"
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
    
    # Check if a command was specified
    if not args.command_group:
        parser.print_help()
        return 0
    
    # Import utils and set the API base URL
    from ollama_models.utils import set_api_base
    set_api_base(args.api)
    
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
