"""
Test the CLI entry point.
"""
import unittest
from unittest.mock import patch
import sys
import io
from ollama_models.cli import main

class TestCLI(unittest.TestCase):
    """
    Test the CLI entry point.
    """
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_no_args(self, mock_args, mock_stdout):
        """Test the main function with no arguments."""
        # Mock parse_args to return an object with command_group=None
        mock_args.return_value = type('obj', (object,), {
            'command_group': None,
            'verbose': False,
            'api': 'http://localhost:11434'
        })
          # Call the main function
        result = main()
        
        # Check the return code and output
        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn('usage:', output)

if __name__ == '__main__':
    unittest.main()
