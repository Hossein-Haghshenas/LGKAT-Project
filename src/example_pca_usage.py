"""
Example usage of PCA preprocessing module.

This script demonstrates how to use PCA for preprocessing different types of data
in the knowledge graph-based recommendation system.
"""

import numpy as np
from pca_preprocessing import (
    PCA, 
    apply_pca_to_ratings, 
    apply_pca_to_kg_features, 
    apply_pca_to_embeddings
)


def example_custom_pca():
    """Example of using the custom PCA implementation."""
    print("=" * 60)
    print("Example 1: Custom PCA Implementation")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 100, 50
    # Create data with some correlation
    X = np.random.randn(n_samples, n_features)
    X = X + 0.5 * X[:, :1]  # Add correlation
    
    print(f"Original data shape: {X.shape}")
    
    # Initialize and fit PCA
    pca = PCA(n_components=10)
    X_transformed = pca.fit_transform(X)
    
    print(f"Transformed data shape: {X_transformed.shape}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    print(f"Number of components: {len(pca.explained_variance_ratio_)}")
    
    # Inverse transform
    X_reconstructed = pca.inverse_transform(X_transformed)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"Reconstruction error (MSE): {reconstruction_error:.6f}")
    print()


def example_ratings_pca():
    """Example of applying PCA to ratings."""
    print("=" * 60)
    print("Example 2: PCA on Ratings Data")
    print("=" * 60)
    
    # This would be used with actual data files
    # Uncomment and modify paths as needed:
    """
    rating_file = '../data/movie/ratings_final'
    output_file = '../data/movie/ratings_final_pca'
    
    reduced_ratings, pca_model = apply_pca_to_ratings(
        rating_file=rating_file,
        output_file=output_file,
        n_components=50,  # Keep 50 components
        variance_threshold=0.95,
        use_sklearn=True,
        standardize=True
    )
    
    print(f"Reduced ratings shape: {reduced_ratings.shape}")
    """
    print("To use this function, provide paths to your rating files.")
    print("Example command:")
    print("  python pca_preprocessing.py -t ratings -d movie -n 50")
    print()


def example_kg_features_pca():
    """Example of applying PCA to KG features."""
    print("=" * 60)
    print("Example 3: PCA on KG Features")
    print("=" * 60)
    
    # This would be used with actual data files
    # Uncomment and modify paths as needed:
    """
    kg_feat_file = '../data/book/kg_feat.npz'
    output_file = '../data/book/kg_feat_pca'
    
    reduced_features, pca_model = apply_pca_to_kg_features(
        kg_feat_file=kg_feat_file,
        output_file=output_file,
        n_components=None,  # Auto-determine based on variance
        variance_threshold=0.95,
        use_sklearn=True,
        standardize=True
    )
    
    print(f"Reduced KG features shape: {reduced_features.shape}")
    """
    print("To use this function, provide paths to your KG feature files.")
    print("Example command:")
    print("  python pca_preprocessing.py -t kg_features -d book -v 0.95")
    print()


def example_embeddings_pca():
    """Example of applying PCA to embeddings."""
    print("=" * 60)
    print("Example 4: PCA on Embeddings")
    print("=" * 60)
    
    # Generate sample embedding data
    np.random.seed(42)
    n_entities, embedding_dim = 1000, 128
    embeddings = np.random.randn(n_entities, embedding_dim)
    
    print(f"Original embeddings shape: {embeddings.shape}")
    
    # Apply PCA to reduce dimensionality
    reduced_embeddings, pca_model = apply_pca_to_embeddings(
        embedding_file=None,  # Using in-memory data, so we'll do it manually
        output_file=None,
        n_components=64,  # Reduce from 128 to 64 dimensions
        variance_threshold=0.95,
        use_sklearn=True,
        standardize=True
    )
    
    # Manual application for demonstration
    from sklearn.decomposition import PCA as SklearnPCA
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    pca = SklearnPCA(n_components=64)
    reduced_embeddings = pca.fit_transform(embeddings_scaled)
    
    print(f"Reduced embeddings shape: {reduced_embeddings.shape}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    print()


def example_variance_threshold():
    """Example of using variance threshold to determine components."""
    print("=" * 60)
    print("Example 5: Auto-determining Components via Variance Threshold")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 200, 100
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure (first 10 features have most variance)
    X[:, :10] *= 5
    
    print(f"Original data shape: {X.shape}")
    
    # Use variance threshold to determine number of components
    from sklearn.decomposition import PCA as SklearnPCA
    
    # First, fit with all components to see variance
    pca_full = SklearnPCA()
    pca_full.fit(X)
    
    # Find number of components for 95% variance
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
    
    print(f"Components needed for 95% variance: {n_components_95}")
    print(f"Variance explained by first {n_components_95} components: {cumsum_variance[n_components_95-1]:.4f}")
    
    # Now fit with determined number of components
    pca = SklearnPCA(n_components=n_components_95)
    X_reduced = pca.fit_transform(X)
    
    print(f"Reduced data shape: {X_reduced.shape}")
    print()


if __name__ == '__main__':
    print("\nPCA Preprocessing Examples")
    print("=" * 60)
    print()
    
    # Run examples
    example_custom_pca()
    example_ratings_pca()
    example_kg_features_pca()
    example_embeddings_pca()
    example_variance_threshold()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nFor command-line usage:")
    print("  python pca_preprocessing.py -t ratings -d movie -n 50")
    print("  python pca_preprocessing.py -t kg_features -d book -v 0.95")
    print("\nFor integration with preprocessing pipeline:")
    print("  python preprocess.py -d movie --pca_ratings --pca_components 50")

