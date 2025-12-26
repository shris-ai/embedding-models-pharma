# Quick Start Guide

## Overview
This project demonstrates how specialized embedding models outperform general-purpose models on medical terminology (MedDRA Preferred Terms).

## Setup Instructions

### 1. Run Setup Script (Recommended)
```bash
bash setup.sh
```

This automated script will:
- Create a Python virtual environment (`venv/`)
- Upgrade pip to the latest version
- Install all required dependencies:
  - `pandas`, `openpyxl`, `xlrd` - For reading MedDRA Excel data
  - `sentence-transformers` - For embedding models
  - `faiss-cpu` - Vector database for similarity search (Python 3.14 compatible)
  - `streamlit` - Web interface
  - `plotly`, `umap-learn` - Visualization tools
  - `torch`, `transformers` - Deep learning frameworks

### 2. Activate Virtual Environment
```bash
source venv/bin/activate
```

**Note**: You need to activate the virtual environment every time you open a new terminal session.

### 3. Create Vector Databases
```bash
python scripts/create_embeddings.py
```

This script will:
- Load MedDRA Preferred Terms from `data/Meddra 20.0_AE_PT.xls`
- Generate embeddings using:
  - **General Model**: `all-MiniLM-L6-v2` (384 dimensions)
  - **Medical Model**: `pritamdeka/S-PubMedBert-MS-MARCO` (768 dimensions)
- Create FAISS vector databases in `vector_dbs/` folder
- Process up to 5000 terms (configurable in code)

**Note**: This may take 10-30 minutes depending on your hardware and dataset size.

### 4. Launch the Demo
```bash
streamlit run src/app.py
```

Or use the convenience script:
```bash
bash run.sh
```

The web interface will open in your browser at `http://localhost:8501`

## Using the Demo

### Search Functionality
1. Enter a natural language medical query (e.g., "feeling dizzy and nauseous")
2. View side-by-side results from both models
3. Compare similarity scores and term rankings

### Example Queries to Try
- "feeling dizzy and nauseous"
- "heart racing and chest pain"
- "skin rash with itching"
- "trouble breathing"
- "severe headache with vision problems"
- "stomach pain after eating"
- "muscle weakness in legs"

### Features
- **Side-by-Side Comparison**: See how each model ranks MedDRA terms
- **Similarity Analysis**: Heatmap visualization of similarity scores
- **Detailed Results**: Full result lists with scores and distances
- **Embedding Visualization**: 2D projections of embedding spaces using UMAP/t-SNE/PCA

## Expected Results

### General Model (all-MiniLM-L6-v2)
- Fast and efficient
- Good for general text similarity
- May miss medical nuances
- Might return less clinically relevant terms

### Medical Model (S-PubMedBert-MS-MARCO)
- Trained on PubMed medical literature
- Better understanding of medical terminology
- More clinically relevant results
- Better semantic understanding of symptoms and conditions

## Project Structure

```
EmbeddingModels/
├── data/                               # MedDRA data files
│   └── Meddra 20.0_AE_PT.xls
├── src/                                # Source code
│   ├── models/                         # Embedding models
│   │   └── embedding_comparison.py
│   ├── utils/                          # Utilities
│   │   └── visualize_embeddings.py
│   ├── database/                       # Database utilities
│   └── app.py                          # Streamlit web app
├── scripts/                            # Executable scripts
│   └── create_embeddings.py
├── notebooks/                          # Jupyter notebooks (optional)
├── logs/                               # Log files
├── vector_dbs/                         # Generated vector databases
│   ├── general/                        # General model embeddings
│   └── medical/                        # Medical model embeddings
├── venv/                               # Virtual environment (generated)
├── setup.sh                            # Setup script
├── run.sh                              # Run script
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── QUICKSTART.md                       # This file
```

## Customization

### Change Number of Terms
Edit `src/models/embedding_comparison.py`, line ~130:
```python
# Change 5000 to desired number or remove limit entirely
if len(terms) > 5000:
    terms = terms[:5000]
```

### Use Different Models
Edit `src/models/embedding_comparison.py`, `__init__` method:
```python
# Replace with any SentenceTransformer-compatible model
self.general_model = SentenceTransformer('your-model-name')
self.medical_model = SentenceTransformer('your-medical-model')
```

### Adjust Search Parameters
In `src/app.py`, modify the `top_k` slider range or default value.

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: "Vector databases not found"
**Solution**: Run the embedding creation script first:
```bash
python scripts/create_embeddings.py
```

### Issue: Out of memory
**Solution**: Reduce the number of terms processed or batch size in `src/models/embedding_comparison.py`

### Issue: Slow performance
**Solution**: 
- Use a smaller sample of terms
- Use PCA instead of UMAP/t-SNE for visualization
- Reduce `top_k` results

## Technical Details

### Embedding Models
- **General**: 384-dimensional embeddings, ~22M parameters
- **Medical**: 768-dimensional embeddings, ~110M parameters

### Vector Database
- FAISS with cosine similarity (Python 3.14 compatible)
- Persistent storage for fast loading
- Supports efficient similarity search

### Visualization
- UMAP: Best for preserving global structure
- t-SNE: Good for local clustering
- PCA: Fastest but less accurate

## Next Steps

1. Try different medical queries and observe the differences
2. Experiment with other embedding models
3. Add more MedDRA data (SOC, HLT, HLGT levels)
4. Implement evaluation metrics (precision, recall)
5. Fine-tune models on your specific medical domain

## Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [MedDRA Information](https://www.meddra.org/)
- [PubMedBERT Paper](https://arxiv.org/abs/2007.15779)
