# Medical Embedding Models Comparison

This project demonstrates how specialized vs non-specialized embedding models perform on medical terminology (MedDRA Preferred Terms). It showcases the importance of using domain-specific models for medical use cases.

## Overview

The project compares a general-purpose model against multiple medical-specialized models:

**General-Purpose Model:**
- `all-MiniLM-L6-v2` - Fast, general-purpose sentence embeddings

**Medical-Specialized Models:**
- `PubMedBERT` - Fine-tuned on MS-MARCO with medical literature
- `ClinicalBERT` - Trained on clinical notes
- `BioBERT` - Pre-trained on biomedical literature
- `SapBERT` - Self-alignment pre-training for medical entities
- `BlueBERT` - Trained on PubMed and MIMIC-III

## Features

- **Multiple Model Support**: Compare different medical embedding models
- **Interactive Model Selection**: Switch between models via sidebar dropdown
- **MedDRA Code Display**: Shows PT codes alongside preferred terms
- **Vector Search**: FAISS-based similarity search (Python 3.14 compatible)
- **Query Mapping**: Find similar MedDRA terms for natural language queries
- **Interactive Visualizations**: Compare embedding spaces with UMAP/t-SNE/PCA
- **Side-by-Side Comparison**: View results from general vs medical models
- **Python 3.14 Compatible**: Uses FAISS instead of ChromaDB

## Installation

### Quick Setup (Recommended)
```bash
bash setup.sh
```

This will:
- Create a virtual environment (`venv/`)
- Install all dependencies
- Set up the project structure

### Manual Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
bash run.sh
```

The script will:
1. Check for virtual environment and embeddings
2. Offer to generate embeddings if missing
3. Launch the Streamlit web app

### Manual Steps

#### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

#### 2. Generate Embeddings

**Option A: Generate for a specific model**
```bash
python scripts/create_embeddings.py pubmedbert
```

Available models: `pubmedbert`, `clinicalbert`, `biobert`, `sapbert`, `bluebert`

**Option B: Generate for all models (recommended)**
```bash
python scripts/generate_all_models.py
```

This takes 1-2 hours but allows you to compare all models in the app.

#### 3. Run the Web App
```bash
streamlit run src/app.py
```

### Switching Between Models

Once embeddings are generated, use the **sidebar dropdown** in the web app to switch between different medical models and compare their performance on the same queries.

## Project Structure

```
EmbeddingModels/
├── data/                           # MedDRA data files
│   └── Meddra 20.0_AE_PT.xls
├── src/                            # Source code
│   ├── models/                     # Embedding models
│   │   └── embedding_comparison.py
│   ├── utils/                      # Utilities
│   │   └── visualize_embeddings.py
│   ├── database/                   # Database utilities
│   └── app.py                      # Streamlit web app
├── scripts/                        # Executable scripts
│   ├── create_embeddings.py        # Generate embeddings for one model
│   └── generate_all_models.py      # Generate embeddings for all models
├── notebooks/                      # Jupyter notebooks (optional)
├── logs/                          # Log files
├── vector_dbs/                    # Generated vector databases
│   ├── general/                    # General model embeddings
│   ├── pubmedbert/                 # PubMedBERT embeddings
│   ├── clinicalbert/               # ClinicalBERT embeddings
│   ├── biobert/                    # BioBERT embeddings
│   ├── sapbert/                    # SapBERT embeddings
│   └── bluebert/                   # BlueBERT embeddings
├── venv/                          # Virtual environment (generated)
├── config.py                      # Model configuration
├── setup.sh                       # Setup script
├── run.sh                         # Run script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── QUICKSTART.md                  # Quick start guide
```

## Example Queries

Try queries like:
- "feeling dizzy and nauseous"
- "heart racing and chest pain"
- "skin rash with itching"
- "trouble breathing"
- "severe headache with vision problems"
- "abdominal pain lower right side"

You'll see how different medical-specialized models understand medical terminology compared to the general-purpose model. Each model may excel at different types of medical queries based on its training data.

## Key Results

Each search result includes:
- **Rank**: Position in similarity ranking
- **MedDRA Code**: PT (Preferred Term) code
- **Term**: MedDRA preferred term
- **Similarity Score**: Cosine similarity (0-1, higher is better)
- **Distance**: 1 - similarity

## Configuration

Edit `config.py` to:
- Add new medical models
- Change the default model
- Adjust vector database settings
- Modify the maximum number of terms to process

## Technical Details

- **Vector Database**: FAISS with cosine similarity
- **Embedding Dimensions**: 384 (general), 768 (medical models)
- **Dataset**: MedDRA 20.0 Preferred Terms
- **Default Sample Size**: 5000 terms (configurable)
- **Python Version**: 3.14+ compatible

## Troubleshooting

**Error: Index files not found**
```bash
# Generate embeddings for the model
source venv/bin/activate
python scripts/create_embeddings.py pubmedbert
```

**No models in dropdown**
- Generate embeddings for at least one medical model
- Check that `vector_dbs/{model_key}/` directories exist

**App won't start**
```bash
# Reinstall dependencies
bash setup.sh
```

## License

This project is for educational and research purposes.
