"""
Test the file_utils module.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import json
import shutil
from ollama_models.file_utils import ModelFileManager

class TestFileUtils(unittest.TestCase):
    """
    Test the file utilities.
    """
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary models JSON file
        self.temp_models_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_models_json.write(json.dumps([
            {
                "name": "test-model",
                "tags": [
                    {
                        "name": "latest",
                        "parameter_size": "7B",
                        "size": "4.1 GB"
                    }
                ]
            }
        ]).encode("utf-8"))
        self.temp_models_json.close()
        
        # Create a temporary directory to act as the current directory
        self.temp_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Manager instance for tests
        self.manager = ModelFileManager()
    
    def tearDown(self):
        """Clean up the test environment."""
        os.chdir(self.old_cwd)
        os.unlink(self.temp_models_json.name)
        shutil.rmtree(self.temp_dir)
    @patch('importlib.util.find_spec')
    def test_get_default_models_path(self, mock_find_spec):
        """Test the get_default_models_path method."""
        # Test when package is installed
        mock_spec = MagicMock()
        mock_spec.origin = "/package/path/__init__.py"
        mock_find_spec.return_value = mock_spec
        
        path = self.manager.get_default_models_path()
        expected_path = os.path.normpath("/package/path/ollama_models.json")
        self.assertEqual(path, expected_path)
        mock_find_spec.assert_called_once_with("ollama_models")

    def test_get_models_path_explicit(self):
        """Test get_models_path with an explicitly specified path."""
        # Case 1: Absolute path that exists
        path = self.manager.get_models_path(self.temp_models_json.name)
        self.assertEqual(path, self.temp_models_json.name)
        
        # Case 2: Absolute path that doesn't exist
        non_existent = os.path.join(self.temp_dir, "non_existent.json")
        with patch.object(self.manager, 'get_default_models_path') as mock_default:
            mock_default.return_value = self.temp_models_json.name
            path = self.manager.get_models_path(non_existent)
            # Should fall back to default path
            self.assertEqual(path, self.temp_models_json.name)
    
    def test_get_models_path_relative(self):
        """Test get_models_path with a relative path."""
        # Create a file in the current directory
        local_file = "local_test.json"
        with open(local_file, 'w') as f:
            f.write("{}")
        
        try:
            # Test with relative path to existing file
            path = self.manager.get_models_path(local_file)
            self.assertEqual(path, os.path.join(self.temp_dir, local_file))
            
            # Test with relative path to non-existent file
            with patch.object(self.manager, 'get_default_models_path') as mock_default:
                mock_default.return_value = self.temp_models_json.name
                path = self.manager.get_models_path("non_existent.json")
                # Should fall back to default path
                self.assertEqual(path, self.temp_models_json.name)
        finally:
            # Cleanup
            if os.path.exists(local_file):
                os.unlink(local_file)
    
    def test_get_models_path_local_file(self):
        """Test get_models_path with a local file in current directory."""
        # Create the default file in the current directory
        local_file = "ollama_models.json"
        with open(local_file, 'w') as f:
            f.write("{}")
        
        try:
            # When no path is specified, should find the local file
            path = self.manager.get_models_path()
            self.assertEqual(path, os.path.join(self.temp_dir, local_file))
        finally:
            # Cleanup
            if os.path.exists(local_file):
                os.unlink(local_file)
    
    def test_get_models_path_default(self):
        """Test get_models_path fallback to default."""
        # When no local file exists, should return the default path
        with patch.object(self.manager, 'get_default_models_path') as mock_default:
            mock_default.return_value = self.temp_models_json.name
            path = self.manager.get_models_path()
            self.assertEqual(path, self.temp_models_json.name)
    
    def test_create_local_models_file(self):
        """Test create_local_models_file method."""
        with patch.object(self.manager, 'get_default_models_path') as mock_default:
            mock_default.return_value = self.temp_models_json.name
            
            # Test creating a local file
            local_path = os.path.join(self.temp_dir, "new_local.json")
            result = self.manager.create_local_models_file(local_path)
            
            self.assertEqual(result, local_path)
            self.assertTrue(os.path.exists(local_path))
            
            # Cleanup
            if os.path.exists(local_path):
                os.unlink(local_path)
    
    def test_read_models_file(self):
        """Test read_models_file method."""
        # Test reading from a specific path
        data = self.manager.read_models_file(self.temp_models_json.name)
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["name"], "test-model")
        
        # Test reading with invalid JSON
        invalid_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("{ this is not valid JSON }")
        
        try:
            data = self.manager.read_models_file(invalid_file)
            self.assertIsNone(data)
        finally:
            if os.path.exists(invalid_file):
                os.unlink(invalid_file)
    
    def test_write_models_file(self):
        """Test write_models_file method."""
        # Test writing to a specific path
        out_path = os.path.join(self.temp_dir, "output.json")
        test_data = [{"name": "new-model", "tags": [{"name": "latest"}]}]
        
        result = self.manager.write_models_file(test_data, out_path)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(out_path))
        
        # Verify the content
        with open(out_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["name"], "new-model")
        
        # Test writing to a path with invalid directory
        invalid_path = "/invalid/dir/file.json"
        result = self.manager.write_models_file(test_data, invalid_path)
        self.assertFalse(result)
        
        # Cleanup
        if os.path.exists(out_path):
            os.unlink(out_path)

if __name__ == "__main__":
    unittest.main()
