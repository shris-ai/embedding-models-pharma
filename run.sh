#!/bin/bash

# Medical Embedding Models Comparison - Run Script
# This script activates the virtual environment and runs the Streamlit app

set -e  # Exit on error

echo "=========================================="
echo "Medical Embedding Models - Run Demo"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup.sh first:"
    echo "  bash setup.sh"
    exit 1
fi

# Check if general model embeddings exist
if [ ! -f "vector_dbs/general/meddra_general.index" ]; then
    echo "⚠️  General model embeddings not found!"
    echo ""
    echo "Please create embeddings first. You have two options:"
    echo ""
    echo "Option 1: Generate embeddings for a specific medical model"
    echo "  source venv/bin/activate"
    echo "  python scripts/create_embeddings.py [model_key]"
    echo "  Available models: pubmedbert, clinicalbert, biobert, sapbert, bluebert"
    echo ""
    echo "Option 2: Generate embeddings for all medical models (recommended)"
    echo "  source venv/bin/activate"
    echo "  python scripts/generate_all_models.py"
    echo ""
    read -p "Do you want to create embeddings now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        source venv/bin/activate
        echo ""
        echo "Which option would you like?"
        echo "  1. Generate for default model (pubmedbert) - faster"
        echo "  2. Generate for all models - comprehensive"
        read -p "Enter choice (1 or 2): " -n 1 -r choice
        echo ""
        if [[ $choice == "1" ]]; then
            python scripts/create_embeddings.py pubmedbert
        elif [[ $choice == "2" ]]; then
            python scripts/generate_all_models.py
        else
            echo "Invalid choice. Exiting."
            exit 1
        fi
    else
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Run Streamlit app
echo ""
echo "Starting Streamlit app..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "💡 Tip: Use the sidebar in the app to switch between different medical models!"
echo ""
streamlit run src/app.py
