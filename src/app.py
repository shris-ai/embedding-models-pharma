"""Streamlit web application for comparing embedding models on MedDRA terms."""
import streamlit as st
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.embedding_comparison import EmbeddingComparison
from src.utils.visualize_embeddings import (
    create_comparison_plot, 
    create_results_comparison_table,
    create_similarity_heatmap
)
import plotly.graph_objects as go
import config


# Page configuration
st.set_page_config(
    page_title="Medical Embedding Models Comparison",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: purple;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_comparator(medical_model_key: str):
    """Load the embedding comparison system.
    
    Args:
        medical_model_key: Key for the medical model to load
    """
    comparator = EmbeddingComparison(medical_model_key=medical_model_key)
    
    # Load FAISS indices
    try:
        comparator.load_index('general', 'vector_dbs/general')
        comparator.load_index('medical', f'vector_dbs/{medical_model_key}')
    except Exception as e:
        st.error(f"Error loading indices: {e}")
        st.info(f"Make sure you've generated embeddings for '{medical_model_key}' model. Run: python scripts/create_embeddings.py {medical_model_key}")
        return None
    
    return comparator


def main():
    # Header
    st.markdown('<div class="main-header">🏥 Medical Embedding Models Comparison</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparing General vs Medical-Specialized Embeddings on MedDRA Terms</div>', 
                unsafe_allow_html=True)
    
    # Sidebar for model selection
    st.sidebar.title("⚙️ Model Configuration")
    st.sidebar.markdown("### Select Medical Model")
    
    # Get available models
    available_models = {}
    for key, model_info in config.MEDICAL_MODELS.items():
        # Check if embeddings exist for this model
        if os.path.exists(f'vector_dbs/{key}'):
            available_models[key] = model_info
    
    if not available_models:
        st.sidebar.warning("No medical model embeddings found. Please generate embeddings first.")
        selected_model_key = config.DEFAULT_MEDICAL_MODEL
    else:
        # Create dropdown with model descriptions
        model_options = {f"{key}: {info['description']}": key for key, info in available_models.items()}
        
        default_option = None
        for option, key in model_options.items():
            if key == config.DEFAULT_MEDICAL_MODEL:
                default_option = option
                break
        
        selected_option = st.sidebar.selectbox(
            "Choose a medical embedding model:",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(default_option) if default_option else 0,
            help="Different models are trained on different medical corpora"
        )
        selected_model_key = model_options[selected_option]
    
    # Display model info
    if selected_model_key in config.MEDICAL_MODELS:
        model_info = config.MEDICAL_MODELS[selected_model_key]
        st.sidebar.markdown(f"**Model:** `{model_info['name']}`")
        st.sidebar.markdown(f"**Dimensions:** {model_info['dimensions']}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### General Model")
    st.sidebar.markdown(f"**Model:** `{config.GENERAL_MODEL['name']}`")
    st.sidebar.markdown(f"**Dimensions:** {config.GENERAL_MODEL['dimensions']}")
    
    # Check if vector databases exist
    if not os.path.exists('vector_dbs/general') or not os.path.exists(f'vector_dbs/{selected_model_key}'):
        st.markdown(f"""
        <div class="warning-box">
            <h3>⚠️ Vector Databases Not Found</h3>
            <p>Please run the following command first to create the vector databases:</p>
            <code>python scripts/create_embeddings.py {selected_model_key}</code>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Load comparator
    try:
        comparator = load_comparator(selected_model_key)
    except Exception as e:
        st.error(f"Error loading comparator: {e}")
        return
    
    # Information section
    with st.expander("ℹ️ About This Demo", expanded=False):
        st.markdown("""
        ### Purpose
        This demo showcases how **specialized embedding models** perform better on domain-specific tasks 
        compared to general-purpose models.
        
        ### Models Being Compared
        - **General Model**: `{config.GENERAL_MODEL['name']}` - {config.GENERAL_MODEL['description']}
        - **Medical Model**: `{comparator.medical_model_name}` - {comparator.medical_model_config['description']}
        
        ### What to Expect
        - The medical model should better understand medical terminology and context
        - Search results from the medical model should be more clinically relevant
        - Embedding visualizations will show different clustering patterns
        
        ### Try These Example Queries
        - "feeling dizzy and nauseous"
        - "heart racing and chest pain"
        - "skin rash with itching"
        - "trouble breathing"
        - "severe headache with vision problems"
        """)
    
    # Main content
    st.markdown("---")
    
    # Initialize session state for query
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    # Example queries
    st.markdown("**Quick examples:**")
    example_cols = st.columns(5)
    examples = [
        "feeling dizzy",
        "heart racing",
        "skin rash",
        "trouble breathing",
        "severe headache"
    ]
    for i, (col, example) in enumerate(zip(example_cols, examples)):
        if col.button(example, key=f"example_{i}"):
            st.session_state.query = example
    
    # Query input
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "🔍 Enter your medical query:",
            value=st.session_state.query,
            placeholder="e.g., feeling dizzy and nauseous",
            help="Enter a natural language description of a medical symptom or condition"
        )
        # Update session state when user types
        if query != st.session_state.query:
            st.session_state.query = query
    with col2:
        top_k = st.slider("Results to show:", 5, 20, 10)
    
    if query:
        st.markdown("---")
        st.markdown(f"### Results for: *\"{query}\"*")
        
        # Search with both models
        with st.spinner("Searching with both models..."):
            try:
                general_results = comparator.query_similar_terms(query, 'general', top_k)
                medical_results = comparator.query_similar_terms(query, 'medical', top_k)
            except Exception as e:
                st.error(f"Error during search: {e}")
                return
        print(general_results)
        print(medical_results)
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["📊 Side-by-Side Comparison", "📈 Similarity Analysis", "🔬 Detailed Results"])
        
        with tab1:
            st.markdown("### Side-by-Side Results Comparison")
            st.markdown("""
            <div class="info-box">
                Compare how each model ranks MedDRA terms for your query. 
                Notice the differences in term selection and similarity scores.
            </div>
            """, unsafe_allow_html=True)
            
            # Display comparison table
            html_table = create_results_comparison_table(general_results, medical_results)
            st.markdown(html_table, unsafe_allow_html=True)
            
            # Key insights
            st.markdown("#### 🔑 Key Observations")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "General Model Avg Similarity",
                    f"{sum(r['similarity'] for r in general_results[:5])/5:.4f}",
                    help="Average similarity of top 5 results"
                )
            
            with col2:
                st.metric(
                    "Medical Model Avg Similarity", 
                    f"{sum(r['similarity'] for r in medical_results[:5])/5:.4f}",
                    help="Average similarity of top 5 results"
                )
        
        with tab2:
            st.markdown("### Similarity Score Analysis")
            
            # Create similarity heatmap
            fig = create_similarity_heatmap(query, general_results, medical_results)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                Higher similarity scores (darker blue) indicate better matches. 
                Compare which model provides more confident and relevant results.
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Detailed Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔵 General Model Results")
                for result in general_results:
                    
                    with st.container():
                        code_display = f"**Code:** `{result['code']}` | " if result.get('code') else ""
                        st.markdown(f"""
                        **{result['rank']}. {result['term']}**  
                        {code_display}Similarity: `{result['similarity']:.4f}` | Distance: `{result['distance']:.4f}`
                        """)
                        st.markdown("---")
            
            with col2:
                st.markdown("#### 🟢 Medical Model Results")
                for result in medical_results:
                    with st.container():
                        code_display = f"**Code:** `{result['code']}` | " if result.get('code') else ""
                        st.markdown(f"""
                        **{result['rank']}. {result['term']}**  
                        {code_display}Similarity: `{result['similarity']:.4f}` | Distance: `{result['distance']:.4f}`
                        """)
                        st.markdown("---")
    
    # Visualization section
    st.markdown("---")
    st.markdown("## 🎨 Embedding Space Visualization")
    
    with st.expander("📊 View Embedding Space Visualizations", expanded=False):
        st.markdown("""
        <div class="info-box">
            Visualize how different models organize MedDRA terms in embedding space. 
            Similar terms should cluster together. Medical models typically show better 
            clinical clustering.
        </div>
        """, unsafe_allow_html=True)
        
        viz_method = st.selectbox(
            "Dimensionality Reduction Method:",
            ["umap", "tsne", "pca"],
            help="Method to reduce high-dimensional embeddings to 2D for visualization"
        )
        
        sample_size = st.slider(
            "Number of terms to visualize:",
            100, 2000, 500,
            step=100,
            help="Larger samples take longer to process"
        )
        
        if st.button("Generate Visualization"):
            with st.spinner("Generating visualizations... This may take a minute."):
                try:
                    # Get embeddings
                    general_emb, docs = comparator.get_all_embeddings('general')
                    medical_emb, _ = comparator.get_all_embeddings('medical')
                    
                    # Sample if needed
                    if len(docs) > sample_size:
                        import numpy as np
                        indices = np.random.choice(len(docs), sample_size, replace=False)
                        general_emb = general_emb[indices]
                        medical_emb = medical_emb[indices]
                        docs = [docs[i] for i in indices]
                    
                    # Create plots
                    general_fig, medical_fig = create_comparison_plot(
                        general_emb, medical_emb, docs, method=viz_method
                    )
                    
                    # Display plots
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(general_fig, use_container_width=True)
                    with col2:
                        st.plotly_chart(medical_fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="success-box">
                        <b>Interpretation:</b> Each point represents a MedDRA term. 
                        Terms closer together are semantically similar according to the model. 
                        Notice how the medical model may show different clustering patterns 
                        that better reflect clinical relationships.
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating visualization: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built with Streamlit • Powered by Sentence Transformers & FAISS</p>
        <p>This demo illustrates the importance of using domain-specific embedding models for specialized tasks.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
