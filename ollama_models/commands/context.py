"""
Context analysis commands for the ollama-models CLI.
"""
import os
import logging
from ollama_models.core.context_usage import generate_usage_report
from ollama_models.core.context_probe import probe_max_context, SearchAlgorithm
from ollama_models.config import DEFAULT_CONTEXT_USAGE_CSV, DEFAULT_MAX_CONTEXT_CSV

logger = logging.getLogger("ollama_models.context")

def setup_parser(parser):
    """
    Set up the argument parser for the context command group.
    
    Args:
        parser: The argument parser to set up
    """
    subparsers = parser.add_subparsers(dest="subcommand", help="Context subcommands")
    
    # usage command (was context_usage_report.py)
    usage_parser = subparsers.add_parser("usage", help="Generate context usage report")
    usage_parser.add_argument("--output", "-o", default=DEFAULT_CONTEXT_USAGE_CSV,
                            help=f"Output CSV file (default: {DEFAULT_CONTEXT_USAGE_CSV})")
    usage_parser.add_argument("--model", "-m", 
                            help="Process only this specific model (optional)")
    
    # probe command (was max_context_fit.py)
    probe_parser = subparsers.add_parser("probe", help="Probe for maximum context sizes")
    probe_parser.add_argument("--output", "-o", default=DEFAULT_MAX_CONTEXT_CSV,
                            help=f"Output CSV file (default: {DEFAULT_MAX_CONTEXT_CSV})")
    probe_parser.add_argument("--model", "-m", 
                            help="Process only this specific model (optional)")
    probe_parser.add_argument("--max-vram", "-v", 
                            help="Limit the amount of VRAM by setting a max VRAM amount (optional)")

def handle_command(args):
    """
    Handle context commands.
    
    Args:
        args: Command arguments
        
    Returns:
        int: Exit code
    """
    if args.subcommand == "usage":
        return cmd_usage(args)
    elif args.subcommand == "probe":
        return cmd_probe(args)
    else:
        logger.error("No subcommand specified")
        return 1

def cmd_usage(args):
    """
    Implement the context usage command (former context_usage_report.py).
    
    Args:
        args: Command arguments
        
    Returns:
        int: Exit code
    """
    try:
        output_file = args.output
        model_name = args.model if hasattr(args, 'model') else None
        
        logger.info(f"Generating context usage")
        
        # Use the integrated context usage module
        usage_rows = generate_usage_report(output_file, model_name)
        
        logger.info(f"Successfully generated context usage report with {len(usage_rows)} entries")
        return 0
    except Exception as e:
        logger.error(f"Error generating context usage report: {e}", exc_info=True)
        return 1



def cmd_probe(args):
    """
    Implement the context probe command (former max_context_fit.py).
    
    Args:
        args: Command arguments
        
    Returns:
        int: Exit code
    """
    try:
        output_file = args.output
        model_name = args.model if hasattr(args, 'model') else None
        max_vram_arg = args.max_vram if hasattr(args, 'max_vram') else None

        if max_vram_arg is not None:
            try:
                # parse the max VRAM argument.  It is in GiB, so we convert it to a number of bytes
                max_vram = int(float(max_vram_arg) * 1024 * 1024 * 1024) # Convert GiB to bytes
            except ValueError:
                logger.error(f"Invalid max VRAM value: {max_vram_arg}. Must be an number.")
                return 1
        else:
            max_vram = 0
            
        logger.info(f"Probing maximum context sizes")
        if max_vram > 0:
            logger.info(f"Using maximum VRAM limit: {max_vram}")

        # Use the integrated context probe module
        fit_rows = probe_max_context(output_file, SearchAlgorithm.PURE_BINARY_MAX_FIRST_G01, model_name, max_vram=max_vram)
        
        logger.info(f"Successfully probed maximum context sizes with {len(fit_rows)} entries")
        return 0
    except Exception as e:
        logger.error(f"Error probing maximum context sizes: {e}", exc_info=True)
        return 1
