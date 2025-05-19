"""
Test the CLI model commands.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import json
from ollama_models.commands import model
from ollama_models.config import DEFAULT_MODELS_JSON, DEFAULT_CONFIG_FILE

class TestModelCommands(unittest.TestCase):
    """
    Test the model commands.
    """
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary models JSON file
        self.temp_models_json = tempfile.NamedTemporaryFile(delete=False)
        self.temp_models_json.write(json.dumps([
            {
                "name": "test-model",
                "tags": [
                    {
                        "name": "latest",
                        "parameter_size": "7B",
                        "size": "4.1 GB"
                    },
                    {
                        "name": "v1.0",
                        "parameter_size": "7B",
                        "size": "4.1 GB"
                    }
                ]
            }
        ]).encode("utf-8"))
        self.temp_models_json.close()
        
        # Create a temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(delete=False)
        self.temp_config.write(b"test-model:latest\n")
        self.temp_config.close()
    
    def tearDown(self):
        """Clean up the test environment."""
        os.unlink(self.temp_models_json.name)
        os.unlink(self.temp_config.name)

    @patch("ollama_models.core.scraper.scrape_and_save")
    def test_fetch(self, mock_scrape_and_save):
        """Test the fetch command."""
        # Mock the scraper to write a valid file
        def side_effect(output_file):
            with open(output_file, 'w') as f:
                json.dump([{
                    "name": "test-model",
                    "description": "Test model",
                    "url": "https://ollama.com/library/test-model",
                    "updated_timestamp": "2025-05-19T12:00:00.000000",
                    "tags": [{"name": "latest", "parameter_size": "7B", "size": "4.1 GB"}]
                }], f)
            return output_file
            
        mock_scrape_and_save.side_effect = side_effect
        
        # Create a mock args object
        args = MagicMock()
        args.output = self.temp_models_json.name
        args.skip_validation = True
        args.force = False
        
        # Run the fetch command
        result = model.cmd_fetch(args)
        
        # Check the result
        self.assertEqual(result, 0)
          # Check that the file was updated
        with open(self.temp_models_json.name, "r") as f:
            data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["name"], "test-model")
    
    @patch("ollama_models.file_utils.ModelFileManager.get_models_path")
    @patch("ollama_models.core.tag_selector.run_selector")
    def test_edit(self, mock_run_selector, mock_get_models_path):
        """Test the edit command."""
        # Mock the tag selector
        mock_run_selector.return_value = 0
        
        # Mock the file manager to return our temp file path
        mock_get_models_path.return_value = self.temp_models_json.name
        
        # Create a mock args object
        args = MagicMock()
        args.models_file = None  # Test the case where models_file is None
        args.config_file = self.temp_config.name
        
        # Run the edit command
        result = model.cmd_edit(args)
        
        # Check the result
        self.assertEqual(result, 0)
        mock_get_models_path.assert_called_once_with(None)
        mock_run_selector.assert_called_once_with(self.temp_models_json.name, self.temp_config.name)
    
    @patch("ollama_models.core.syncer.sync_ollama")
    def test_apply(self, mock_sync_ollama):
        """Test the apply command."""
        # Mock the syncer
        mock_sync_ollama.return_value = (True, ["test-model:latest"], [])
        
        # Create a mock args object
        args = MagicMock()
        args.config_file = self.temp_config.name
        
        # Run the apply command
        result = model.cmd_apply(args)
        
        # Check the result
        self.assertEqual(result, 0)
        mock_sync_ollama.assert_called_once()
    
    @patch("ollama_models.core.initializer.init_from_api")
    def test_init(self, mock_init_from_api):
        """Test the init command."""
        # Mock the initializer
        mock_init_from_api.return_value = (True, ["test-model:latest"])
        
        # Create a mock args object
        args = MagicMock()
        args.config_file = self.temp_config.name
        
        # Run the init command
        result = model.cmd_init(args)
        
        # Check the result
        self.assertEqual(result, 0)
        mock_init_from_api.assert_called_once()

if __name__ == "__main__":
    unittest.main()
