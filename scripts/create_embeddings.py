"""
Script to create embeddings and vector databases from MedDRA data.
Run this first before launching the Streamlit app.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.embedding_comparison import main

if __name__ == '__main__':
    print("="*60)
    print("Creating Embeddings for MedDRA Terms")
    print("="*60)
    print("\nThis script will:")
    print("1. Load MedDRA Preferred Terms from the data file")
    print("2. Generate embeddings using both general and medical models")
    print("3. Create vector databases for fast similarity search")
    print("\nThis may take several minutes depending on your hardware...")
    print("="*60)
    print()
    
    main()
