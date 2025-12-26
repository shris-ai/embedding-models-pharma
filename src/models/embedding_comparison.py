"""
Core module for embedding comparison between general and medical-specialized models.
Uses FAISS for vector similarity search (Python 3.14 compatible).
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Dict, Tuple, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class EmbeddingComparison:
    """Compare different embedding models on medical terminology."""
    
    def __init__(self, medical_model_key: Optional[str] = None):
        """Initialize embedding models.
        
        Args:
            medical_model_key: Key from config.MEDICAL_MODELS to use (e.g., 'pubmedbert', 'clinicalbert')
                              If None, uses DEFAULT_MEDICAL_MODEL from config
        """
        # Import config
        try:
            import config
        except ImportError:
            # Fallback if config not found
            config = type('obj', (object,), {
                'GENERAL_MODEL': {'name': 'all-MiniLM-L6-v2'},
                'MEDICAL_MODELS': {'pubmedbert': {'name': 'pritamdeka/S-PubMedBert-MS-MARCO'}},
                'DEFAULT_MEDICAL_MODEL': 'pubmedbert'
            })()
        
        print("Loading embedding models...")
        
        # General-purpose model (non-specialized)
        self.general_model_name = config.GENERAL_MODEL['name']
        self.general_model = SentenceTransformer(self.general_model_name)
        print(f"✓ Loaded general model: {self.general_model_name}")
        
        # Medical-specialized model
        if medical_model_key is None:
            medical_model_key = config.DEFAULT_MEDICAL_MODEL
        
        self.medical_model_key = medical_model_key
        self.medical_model_config = config.MEDICAL_MODELS.get(medical_model_key)
        
        if self.medical_model_config is None:
            raise ValueError(f"Unknown medical model key: {medical_model_key}")
        
        self.medical_model_name = self.medical_model_config['name']
        self.medical_model = SentenceTransformer(self.medical_model_name)
        print(f"✓ Loaded medical model: {self.medical_model_name}")
        print(f"  Description: {self.medical_model_config['description']}")
        
        self.meddra_terms = []
        self.meddra_data = []  # Store dicts with code and term
        self.general_index = None
        self.medical_index = None
        self.general_data = []  # Store dicts with code and term
        self.medical_data = []
        
    def load_meddra_data(self, file_path: str) -> pd.DataFrame:
        """Load MedDRA Preferred Terms from Excel file."""
        print(f"Loading MedDRA data from {file_path}...")
        
        # Try different engines for Excel files
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
        except:
            try:
                df = pd.read_excel(file_path, engine='xlrd')
            except Exception as e:
                print(f"Error loading Excel file: {e}")
                raise
        
        print(f"✓ Loaded {len(df)} MedDRA terms")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def prepare_terms(self, df: pd.DataFrame) -> List[Dict]:
        """Extract and prepare MedDRA preferred terms with codes."""
        # Find PT (Preferred Term) column
        possible_pt_columns = ['PT', 'Preferred Term', 'pt', 'preferred_term', 'PREFERRED_TERM', 'Term English']
        pt_column = None
        for col in possible_pt_columns:
            if col in df.columns:
                pt_column = col
                break
        
        if pt_column is None:
            pt_column = df.columns[0]
            print(f"Using column '{pt_column}' for preferred terms")
        
        # Find Code column
        possible_code_columns = ['PT Code', 'Code', 'PT_CODE', 'code', 'LLT Code']
        code_column = None
        for col in possible_code_columns:
            if col in df.columns:
                code_column = col
                break
        
        if code_column:
            print(f"Using column '{code_column}' for MedDRA codes")
        else:
            print("Warning: No code column found, codes will be empty")
        
        # Extract unique terms with codes
        data_list = []
        seen_terms = set()
        
        for _, row in df.iterrows():
            term = str(row[pt_column]).strip() if pd.notna(row[pt_column]) else None
            code = str(int(row[code_column])) if code_column and pd.notna(row[code_column]) else ""
            
            if term and term not in seen_terms:
                data_list.append({
                    'code': code,
                    'term': term
                })
                seen_terms.add(term)
        
        print(f"✓ Prepared {len(data_list)} unique MedDRA terms with codes")
        self.meddra_data = data_list
        self.meddra_terms = [item['term'] for item in data_list]
        
        return data_list
    
    def create_vector_db(self, data_list: List[Dict], model_type: str, 
                         collection_name: str, persist_dir: str):
        """Create a FAISS vector database with embeddings."""
        print(f"\nCreating vector DB for {model_type}...")
        
        # Select the appropriate model
        model = self.general_model if 'general' in model_type.lower() else self.medical_model
        
        # Extract terms for embedding
        terms = [item['term'] for item in data_list]
        
        # Generate embeddings in batches
        print("Generating embeddings...")
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(terms), batch_size):
            batch = terms[i:i+batch_size]
            embeddings = model.encode(batch, show_progress_bar=True, convert_to_numpy=True)
            all_embeddings.append(embeddings)
            
            if (i + batch_size) % 500 == 0:
                print(f"  Processed {min(i+batch_size, len(terms))}/{len(terms)} terms")
        
        # Concatenate all embeddings
        embeddings_matrix = np.vstack(all_embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Create FAISS index
        dimension = embeddings_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings_matrix)
        
        # Save index and data
        os.makedirs(persist_dir, exist_ok=True)
        index_path = os.path.join(persist_dir, f'{collection_name}.index')
        data_path = os.path.join(persist_dir, f'{collection_name}_data.pkl')
        
        faiss.write_index(index, index_path)
        with open(data_path, 'wb') as f:
            pickle.dump(data_list, f)
        
        print(f"✓ Created vector DB with {len(terms)} embeddings")
        print(f"  Saved to: {persist_dir}")
        
        # Store in memory
        if 'general' in model_type.lower():
            self.general_index = index
            self.general_data = data_list
        else:
            self.medical_index = index
            self.medical_data = data_list
        
        return index, data_list
    
    def load_index(self, model_type: str, persist_dir: str):
        """Load FAISS index from disk.
        
        Args:
            model_type: 'general' or the medical model key (e.g., 'pubmedbert')
            persist_dir: Directory containing the index files
        """
        # For general model, use 'meddra_general'
        # For medical models, the collection name is 'meddra_{model_key}' where model_key is in the persist_dir
        if model_type == 'general':
            collection_name = 'meddra_general'
        else:
            # Extract model key from persist_dir (e.g., 'vector_dbs/pubmedbert' -> 'pubmedbert')
            model_key = os.path.basename(persist_dir)
            collection_name = f'meddra_{model_key}'
        
        index_path = os.path.join(persist_dir, f'{collection_name}.index')
        data_path = os.path.join(persist_dir, f'{collection_name}_data.pkl')
        
        if not os.path.exists(index_path) or not os.path.exists(data_path):
            raise FileNotFoundError(f"Index files not found in {persist_dir}. Looking for {collection_name}.index and {collection_name}_data.pkl")
        
        index = faiss.read_index(index_path)
        with open(data_path, 'rb') as f:
            data_list = pickle.load(f)
        
        if model_type == 'general':
            self.general_index = index
            self.general_data = data_list
        else:
            self.medical_index = index
            self.medical_data = data_list
        
        return index, data_list
    
    def query_similar_terms(self, query: str, model_type: str, 
                           top_k: int = 10) -> List[Dict]:
        """Query for similar MedDRA terms."""
        # Select model and index
        if model_type == 'general':
            model = self.general_model
            index = self.general_index
            data_list = self.general_data
        else:
            model = self.medical_model
            index = self.medical_index
            data_list = self.medical_data
        
        if index is None:
            raise ValueError(f"Index for {model_type} model not loaded")
        
        # Generate query embedding
        query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = index.search(query_embedding, top_k)
        
        # Format results
        formatted_results = []
        for i, (idx, similarity) in enumerate(zip(indices[0], similarities[0])):
            formatted_results.append({
                'rank': i + 1,
                'code': data_list[idx]['code'],
                'term': data_list[idx]['term'],
                'similarity': float(similarity),
                'distance': float(1 - similarity)  # Convert similarity to distance
            })
        
        return formatted_results
    
    def get_all_embeddings(self, model_type: str) -> Tuple[np.ndarray, List[str]]:
        """Get all embeddings from a vector database."""
        if model_type == 'general':
            index = self.general_index
            data_list = self.general_data
        else:
            index = self.medical_index
            data_list = self.medical_data
        
        if index is None:
            raise ValueError(f"Index for {model_type} model not loaded")
        
        # Reconstruct embeddings from FAISS index
        n = index.ntotal
        embeddings = np.zeros((n, index.d), dtype='float32')
        index.reconstruct_n(0, n, embeddings)
        
        # Extract just terms for visualization
        terms = [item['term'] for item in data_list]
        
        return embeddings, terms


def main(medical_model_key: Optional[str] = None):
    """Main function to create embeddings and vector databases.
    
    Args:
        medical_model_key: Which medical model to use (e.g., 'pubmedbert', 'clinicalbert')
    """
    # Initialize
    comparator = EmbeddingComparison(medical_model_key=medical_model_key)
    
    # Load data - try new file first, fall back to old file
    data_path = 'data/Meddra_AE_PT.xlsx'
    if not os.path.exists(data_path):
        data_path = 'data/Meddra 20.0_AE_PT.xls'
        print(f"Using fallback file: {data_path}")
    
    df = comparator.load_meddra_data(data_path)
    
    # Prepare terms with codes
    data_list = comparator.prepare_terms(df)
    
    # Limit to first 5000 terms for faster processing (remove this for full dataset)
    if len(data_list) > 5000:
        print(f"\nLimiting to first 5000 terms for demo purposes...")
        data_list = data_list[:5000]
    
    # Create vector databases
    model_key = comparator.medical_model_key
    os.makedirs('vector_dbs/general', exist_ok=True)
    os.makedirs(f'vector_dbs/{model_key}', exist_ok=True)
    
    # General model
    comparator.create_vector_db(
        data_list, 
        'general', 
        'meddra_general',
        'vector_dbs/general'
    )
    
    # Medical model
    comparator.create_vector_db(
        data_list,
        'medical',
        f'meddra_{model_key}', 
        f'vector_dbs/{model_key}'
    )
    
    print("\n" + "="*60)
    print("✓ Successfully created vector databases!")
    print("="*60)
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run src/app.py")


if __name__ == '__main__':
    import sys
    # Allow passing model key as command line argument
    model_key = sys.argv[1] if len(sys.argv) > 1 else None
    main(medical_model_key=model_key)
