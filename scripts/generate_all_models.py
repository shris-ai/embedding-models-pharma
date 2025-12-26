"""
Script to generate embeddings for all available medical models.
This allows you to pre-generate embeddings for all models and then
switch between them in the Streamlit app.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.models.embedding_comparison import main as create_embeddings


def generate_all_models():
    """Generate embeddings for all medical models defined in config."""
    print("="*70)
    print("Generating Embeddings for All Medical Models")
    print("="*70)
    print()
    
    # List available models
    print("Available medical models:")
    for i, (key, model_info) in enumerate(config.MEDICAL_MODELS.items(), 1):
        print(f"  {i}. {key}: {model_info['description']}")
    print()
    
    # Ask user which models to generate
    print("Options:")
    print("  1. Generate embeddings for ALL models (recommended)")
    print("  2. Select specific models")
    print("  3. Generate only the default model")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    
    models_to_generate = []
    
    if choice == "1":
        models_to_generate = list(config.MEDICAL_MODELS.keys())
    elif choice == "2":
        print("\nEnter model keys separated by commas (e.g., pubmedbert,clinicalbert):")
        selected = input("> ").strip().split(',')
        models_to_generate = [k.strip() for k in selected if k.strip() in config.MEDICAL_MODELS]
    elif choice == "3":
        models_to_generate = [config.DEFAULT_MEDICAL_MODEL]
    else:
        print("Invalid choice. Exiting.")
        return
    
    if not models_to_generate:
        print("No valid models selected. Exiting.")
        return
    
    print(f"\nWill generate embeddings for: {', '.join(models_to_generate)}")
    print(f"This will take approximately {len(models_to_generate) * 15}-{len(models_to_generate) * 30} minutes.")
    print()
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Generate embeddings for each model
    for i, model_key in enumerate(models_to_generate, 1):
        print("\n" + "="*70)
        print(f"Processing Model {i}/{len(models_to_generate)}: {model_key}")
        print("="*70)
        print()
        
        try:
            create_embeddings(medical_model_key=model_key)
            print(f"\n✓ Successfully generated embeddings for {model_key}")
        except Exception as e:
            print(f"\n✗ Error generating embeddings for {model_key}: {e}")
            continue
    
    print("\n" + "="*70)
    print("✓ Embedding generation complete!")
    print("="*70)
    print("\nYou can now run the Streamlit app and switch between models:")
    print("  streamlit run src/app.py")
    print()


if __name__ == '__main__':
    generate_all_models()
