"""
Test the CLI context commands.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import csv
from ollama_models.commands import context

class TestContextCommands(unittest.TestCase):
    """
    Test the context commands.
    """
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary files for outputs
        self.temp_usage_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        self.temp_usage_csv.close()
        
        self.temp_probe_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        self.temp_probe_csv.close()
    
    def tearDown(self):
        """Clean up the test environment."""
        for temp_file in [self.temp_usage_csv.name, self.temp_probe_csv.name]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    @patch("ollama_models.commands.context.generate_usage_report")
    def test_usage(self, mock_generate_report):
        """Test the context usage command."""
        # Mock the report generator
        mock_generate_report.return_value = [
            ["test-model", 2048, "4.1GiB"],
            ["test-model", 4096, "4.8GiB"]
        ]
        
        # Create a mock args object
        args = MagicMock()
        args.output = self.temp_usage_csv.name
        args.model = "test-model"
        
        # Run the usage command
        result = context.cmd_usage(args)
        
        # Check the result
        self.assertEqual(result, 0)
        mock_generate_report.assert_called_once_with(self.temp_usage_csv.name, "test-model")
    @patch("ollama_models.commands.context.probe_max_context")
    def test_probe(self, mock_probe):
        """Test the context probe command."""
        # Mock the probe function
        mock_probe.return_value = [
            ["test-model", 8192]
        ]
        
        # Create a mock args object
        args = MagicMock()
        args.output = self.temp_probe_csv.name
        args.model = "test-model"
        
        # Run the probe command
        result = context.cmd_probe(args)
        
        # Check the result
        self.assertEqual(result, 0)
        mock_probe.assert_called_once_with(self.temp_probe_csv.name, "test-model")

if __name__ == "__main__":
    unittest.main()
