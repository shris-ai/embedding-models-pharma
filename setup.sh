#!/bin/bash

# Medical Embedding Models Comparison - Setup Script
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on error

echo "=========================================="
echo "Medical Embedding Models - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip3 install --upgrade pip -q

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take several minutes..."
pip3 install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Create embeddings:"
echo "   python scripts/create_embeddings.py"
echo ""
echo "3. Run the demo:"
echo "   streamlit run src/app.py"
echo ""
