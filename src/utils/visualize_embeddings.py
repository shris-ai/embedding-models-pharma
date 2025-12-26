"""
Visualization utilities for comparing embedding spaces.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple


def reduce_dimensions(embeddings: np.ndarray, method: str = 'umap', 
                     n_components: int = 2) -> np.ndarray:
    """
    Reduce embedding dimensions for visualization.
    
    Args:
        embeddings: High-dimensional embeddings
        method: Reduction method ('pca', 'tsne', or 'umap')
        n_components: Number of dimensions to reduce to
    
    Returns:
        Reduced embeddings
    """
    print(f"Reducing dimensions using {method.upper()}...")
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, 
                      perplexity=30, max_iter=1000)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42,
                           n_neighbors=15, min_dist=0.1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced = reducer.fit_transform(embeddings)
    print(f"✓ Reduced from {embeddings.shape[1]}D to {n_components}D")
    
    return reduced


def create_embedding_plot(embeddings: np.ndarray, documents: List[str],
                         title: str, highlight_indices: List[int] = None,
                         method: str = 'umap') -> go.Figure:
    """
    Create an interactive 2D scatter plot of embeddings.
    
    Args:
        embeddings: High-dimensional embeddings
        documents: List of document texts
        title: Plot title
        highlight_indices: Indices of points to highlight
        method: Dimensionality reduction method
    
    Returns:
        Plotly figure
    """
    # Reduce dimensions
    reduced = reduce_dimensions(embeddings, method=method, n_components=2)
    
    # Create color array
    colors = ['lightblue'] * len(documents)
    sizes = [5] * len(documents)
    
    if highlight_indices:
        for idx in highlight_indices:
            colors[idx] = 'red'
            sizes[idx] = 12
    
    # Create hover text (show first 100 chars of each term)
    hover_texts = [doc[:100] + '...' if len(doc) > 100 else doc 
                   for doc in documents]
    
    # Create scatter plot
    fig = go.Figure(data=go.Scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.6,
            line=dict(width=0.5, color='white')
        ),
        text=hover_texts,
        hovertemplate='<b>%{text}</b><br>' +
                     f'{method.upper()} 1: %{{x:.2f}}<br>' +
                     f'{method.upper()} 2: %{{y:.2f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=f'{method.upper()} Dimension 1',
        yaxis_title=f'{method.upper()} Dimension 2',
        hovermode='closest',
        width=800,
        height=600,
        template='plotly_white'
    )
    
    return fig


def create_comparison_plot(general_embeddings: np.ndarray, 
                          medical_embeddings: np.ndarray,
                          documents: List[str],
                          query_idx: int = None,
                          method: str = 'umap') -> Tuple[go.Figure, go.Figure]:
    """
    Create side-by-side comparison plots.
    
    Args:
        general_embeddings: Embeddings from general model
        medical_embeddings: Embeddings from medical model
        documents: List of document texts
        query_idx: Index of query point to highlight
        method: Dimensionality reduction method
    
    Returns:
        Tuple of (general_fig, medical_fig)
    """
    highlight = [query_idx] if query_idx is not None else None
    
    general_fig = create_embedding_plot(
        general_embeddings,
        documents,
        'General Model (all-MiniLM-L6-v2)',
        highlight,
        method
    )
    
    medical_fig = create_embedding_plot(
        medical_embeddings,
        documents,
        'Medical Model (S-PubMedBert)',
        highlight,
        method
    )
    
    return general_fig, medical_fig


def create_similarity_heatmap(query: str, general_results: List[dict],
                              medical_results: List[dict]) -> go.Figure:
    """
    Create a heatmap comparing similarity scores.
    
    Args:
        query: Query text
        general_results: Results from general model
        medical_results: Results from medical model
    
    Returns:
        Plotly figure
    """
    # Get top 10 results
    n = min(10, len(general_results), len(medical_results))
    
    # Extract terms and scores
    general_terms = [r['term'][:50] for r in general_results[:n]]
    medical_terms = [r['term'][:50] for r in medical_results[:n]]
    
    general_scores = [r['similarity'] for r in general_results[:n]]
    medical_scores = [r['similarity'] for r in medical_results[:n]]
    
    # Create data matrix
    data = []
    labels = []
    
    for i, (term, score) in enumerate(zip(general_terms, general_scores)):
        data.append([score, 0])
        labels.append(f"G{i+1}: {term}")
    
    for i, (term, score) in enumerate(zip(medical_terms, medical_scores)):
        data.append([0, score])
        labels.append(f"M{i+1}: {term}")
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=['General Model', 'Medical Model'],
        y=labels,
        colorscale='Blues',
        text=[[f'{val:.3f}' if val > 0 else '' for val in row] for row in data],
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Model: %{x}<br>Term: %{y}<br>Similarity: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Similarity Scores for Query: "{query[:50]}..."',
        height=600,
        width=600,
        template='plotly_white'
    )
    
    return fig


def create_results_comparison_table(general_results: List[dict],
                                    medical_results: List[dict]) -> str:
    """
    Create an HTML table comparing results side by side.
    
    Args:
        general_results: Results from general model
        medical_results: Results from medical model
    
    Returns:
        HTML string
    """
    n = max(len(general_results), len(medical_results))
    
    html = """
    <style>
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }
        .comparison-table th {
            background-color: cadetblue;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        .comparison-table td {
            border: 1px solid #ddd;
            padding: 10px;
        }
        .comparison-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .comparison-table tr:hover {
            background-color: #ddd;
        }
        .rank {
            font-weight: bold;
            color: #4CAF50;
        }
        .similarity {
            color: #2196F3;
            font-weight: bold;
        }
    </style>
    <table class="comparison-table">
        <tr>
            <th colspan="4" style="background-color: rebeccapurple;">General Model Results</th>
            <th colspan="4" style="background-color: purple;">Medical Model Results</th>
        </tr>
        <tr>
            <th>Rank</th>
            <th>Code</th>
            <th>Term</th>
            <th>Similarity</th>
            <th>Rank</th>
            <th>Code</th>
            <th>Term</th>
            <th>Similarity</th>
        </tr>
    """
    
    for i in range(n):
        html += "<tr>"
        
        # General model result
        if i < len(general_results):
            g = general_results[i]
            html += f'<td class="rank">{g["rank"]}</td>'
            html += f'<td style="font-family: monospace; color: #666;">{g.get("code", "")}</td>'
            html += f'<td>{g["term"]}</td>'
            html += f'<td class="similarity">{g["similarity"]:.4f}</td>'
        else:
            html += '<td colspan="4"></td>'
        
        # Medical model result
        if i < len(medical_results):
            m = medical_results[i]
            html += f'<td class="rank">{m["rank"]}</td>'
            html += f'<td style="font-family: monospace; color: #666;">{m.get("code", "")}</td>'
            html += f'<td>{m["term"]}</td>'
            html += f'<td class="similarity">{m["similarity"]:.4f}</td>'
        else:
            html += '<td colspan="4"></td>'
        
        html += "</tr>"
    
    html += "</table>"
    
    return html
