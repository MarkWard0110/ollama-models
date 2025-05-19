"""
File utility functions for the Ollama Models CLI.
"""
import os
import json
import logging
import shutil
import sys
import importlib.util

logger = logging.getLogger("ollama_models.file_utils")

class ModelFileManager:
    """
    Manages the operations and resolution of the models JSON file.
    This class handles the logic of determining whether to use a local
    ollama_models.json file or the package's default one.
    """
    
    def __init__(self):
        self._package_name = "ollama_models"
        self._default_models_filename = "ollama_models.json"
    def get_default_models_path(self):
        """
        Get the path to the default models file packaged with the module.
        
        Returns:
            str: The absolute path to the default models file
        """
        try:
            # Try to get the file from the package
            spec = importlib.util.find_spec(self._package_name)
            if spec is not None:
                package_path = os.path.dirname(spec.origin)
                return os.path.normpath(os.path.join(package_path, self._default_models_filename))
        except (ImportError, AttributeError):
            pass
            
        # Fallback to direct file path lookup
        import ollama_models
        return os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(ollama_models.__file__)), 
                        self._default_models_filename))
    
    def get_models_path(self, specified_path=None):
        """
        Determine which models file to use based on context.
        Priority:
        1. Explicitly specified path (if provided)
        2. Local file in current directory
        3. Package default file
        
        Args:
            specified_path (str, optional): An explicitly specified path
            
        Returns:
            str: The path to the models file to use
        """
        # Priority 1: Explicitly specified path
        if specified_path:
            # If the path is absolute, use it directly
            if os.path.isabs(specified_path):
                if os.path.exists(specified_path):
                    return specified_path
                else:
                    logger.warning(f"Specified models file not found: {specified_path}")
                    # Fall through to priority 2
            else:
                # If relative, make it relative to current directory
                full_path = os.path.join(os.getcwd(), specified_path)
                if os.path.exists(full_path):
                    return full_path
                else:
                    logger.warning(f"Specified models file not found: {full_path}")
                    # Fall through to priority 2
        
        # Priority 2: Local file in current directory
        local_file = os.path.join(os.getcwd(), self._default_models_filename)
        if os.path.exists(local_file):
            logger.debug(f"Using local models file: {local_file}")
            return local_file
        
        # Priority 3: Package default file
        default_file = self.get_default_models_path()
        logger.debug(f"Using package default models file: {default_file}")
        return default_file
    
    def create_local_models_file(self, output_path=None):
        """
        Create a local models file in the current directory by copying
        the package's default file.
        
        Args:
            output_path (str, optional): Path where to create the file
                                         (defaults to current directory)
            
        Returns:
            str: Path to the created file or None if the operation failed
        """
        default_file = self.get_default_models_path()
        if not os.path.exists(default_file):
            logger.error(f"Default models file not found: {default_file}")
            return None
        
        if not output_path:
            output_path = os.path.join(os.getcwd(), self._default_models_filename)
        
        try:
            shutil.copy2(default_file, output_path)
            logger.info(f"Created local models file: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to create local models file: {e}")
            return None
    
    def read_models_file(self, file_path=None):
        """
        Read and parse the models file.
        
        Args:
            file_path (str, optional): Path to the models file to read
            
        Returns:
            dict: Parsed models data or None if the operation failed
        """
        models_path = self.get_models_path(file_path)
        
        try:
            with open(models_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in models file {models_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to read models file {models_path}: {e}")
            return None
    
    def write_models_file(self, data, file_path=None):
        """
        Write data to the models file.
        
        Args:
            data: The data to write
            file_path (str, optional): Path to the models file to write
            
        Returns:
            bool: True if the operation succeeded, False otherwise
        """
        output_path = file_path if file_path else os.path.join(os.getcwd(), self._default_models_filename)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to write models file {output_path}: {e}")
            return False
