"""
Test the utility functions.
"""
import unittest
from unittest.mock import patch, MagicMock
import requests
from ollama_models.utils import (
    fetch_installed_models, fetch_max_context_size,
    try_model_call, fetch_memory_usage, format_size
)

class TestUtils(unittest.TestCase):
    """
    Test the utility functions.
    """
    
    def test_format_size(self):
        """Test the format_size function."""
        self.assertEqual(format_size(1024), "1.0KiB")
        self.assertEqual(format_size(1024*1024), "1.0MiB")
        self.assertEqual(format_size(1024*1024*1024), "1.0GiB")
        self.assertEqual(format_size(1024*1024*1024*1024), "1.0TiB")
        self.assertEqual(format_size(500), "500.0B")
        self.assertEqual(format_size(1500), "1.5KiB")
    
    @patch("requests.get")
    def test_fetch_installed_models(self, mock_get):
        """Test the fetch_installed_models function."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "test-model:latest"},
                {"name": "another-model:v1.0"}
            ]
        }
        mock_get.return_value = mock_response
        
        # Call the function
        models = fetch_installed_models()
        
        # Check the result
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["name"], "test-model:latest")
        self.assertEqual(models[1]["name"], "another-model:v1.0")
    
    @patch("requests.get")
    def test_fetch_installed_models_error(self, mock_get):
        """Test the fetch_installed_models function with an error."""
        # Mock the response to raise an exception
        mock_get.side_effect = requests.ConnectionError("Connection refused")
        
        # Call the function and check for the expected exception
        with self.assertRaises(ConnectionError):
            fetch_installed_models()
    
    @patch("requests.post")
    def test_fetch_max_context_size(self, mock_post):
        """Test the fetch_max_context_size function."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model_info": {
                "parameter_size": "7B",
                "llama.context_length": 4096
            }
        }
        mock_post.return_value = mock_response
        
        # Call the function
        context_size = fetch_max_context_size("test-model")
        
        # Check the result
        self.assertEqual(context_size, 4096)
    
    @patch("requests.post")
    def test_fetch_max_context_size_default(self, mock_post):
        """Test the fetch_max_context_size function when no context size is found."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model_info": {
                "parameter_size": "7B"
            }
        }
        mock_post.return_value = mock_response
        
        # Call the function
        context_size = fetch_max_context_size("test-model")
        
        # Check the result (should return the default)
        self.assertEqual(context_size, 2048)
    
    @patch("requests.get")
    def test_fetch_memory_usage(self, mock_get):
        """Test the fetch_memory_usage function."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {
                    "model": "other-model",
                    "size": 1000000,
                    "size_vram": 800000
                },
                {
                    "model": "test-model",
                    "size": 2000000,
                    "size_vram": 1500000
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Call the function
        size, vram = fetch_memory_usage("test-model")
        
        # Check the result
        self.assertEqual(size, 2000000)
        self.assertEqual(vram, 1500000)

if __name__ == "__main__":
    unittest.main()
