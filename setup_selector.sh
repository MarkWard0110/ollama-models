#!/bin/bash

echo "Installing dependencies for Ollama Model Selector..."

# First try to install python-inquirer
pip install python-inquirer || {
    echo "Failed to install python-inquirer, trying alternative package inquirer..."
    pip install inquirer
}

# Install PyInquirer as a fallback (has Separator class)
echo "Installing PyInquirer as a fallback..."
pip install PyInquirer || echo "PyInquirer installation failed, will use custom separator"

echo "Dependencies installed successfully!"
echo "Run the model selector with: python model_selector.py"
