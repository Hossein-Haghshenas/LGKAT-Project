"""
PCA (Principal Component Analysis) Preprocessing Module

This module provides PCA implementations for dimensionality reduction
in the knowledge graph-based recommendation system.

Features:
- From-scratch PCA implementation
- Integration with sklearn PCA
- Support for sparse and dense matrices
- Feature reduction for ratings, KG features, and embeddings
"""

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import os
import argparse


class PCA:
    """
    Principal Component Analysis implementation from scratch.
    
    PCA finds the principal components (directions of maximum variance)
    in the data and projects the data onto these components.
    """
    
    def __init__(self, n_components=None, whiten=False):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of components to keep. If None, keeps all components.
            whiten: Whether to whiten the components (scale to unit variance).
        """
        self.n_components = n_components
        self.whiten = whiten
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_features_ = None
        self.n_samples_ = None
        
    def fit(self, X):
        """
        Fit PCA on the data.
        
        Args:
            X: Input data, shape (n_samples, n_features)
        """
        # Convert sparse matrix to dense if needed
        if sp.issparse(X):
            X = X.toarray()
        
        X = np.asarray(X, dtype=np.float64)
        self.n_samples_, self.n_features_ = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        # For large datasets, use X^T * X / (n-1) instead of full covariance
        if self.n_samples_ > self.n_features_:
            # More samples than features: use feature covariance
            cov_matrix = np.dot(X_centered.T, X_centered) / (self.n_samples_ - 1)
        else:
            # More features than samples: use sample covariance
            cov_matrix = np.dot(X_centered, X_centered.T) / (self.n_samples_ - 1)
            # We'll compute eigenvectors differently in this case
        
        # Compute eigenvalues and eigenvectors
        if self.n_samples_ > self.n_features_:
            # Standard case: eigen decomposition of covariance matrix
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        else:
            # Few samples: compute eigenvectors from sample covariance
            # Use SVD for numerical stability
            U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
            # Eigenvalues are squares of singular values divided by (n-1)
            eigenvalues = (s ** 2) / (self.n_samples_ - 1)
            # Principal components are the right singular vectors
            eigenvectors = Vt.T
            # Pad with zeros if needed (when n_samples < n_features)
            if self.n_features_ > self.n_samples_:
                n_zeros = self.n_features_ - self.n_samples_
                eigenvalues = np.concatenate([eigenvalues, np.zeros(n_zeros)])
                zero_pad = np.zeros((self.n_features_, n_zeros))
                eigenvectors = np.hstack([eigenvectors, zero_pad])
        
        # Sort by eigenvalue (descending order)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select number of components
        if self.n_components is None:
            n_components = min(self.n_samples_, self.n_features_)
        else:
            n_components = min(self.n_components, self.n_samples_, self.n_features_)
        
        self.components_ = eigenvectors[:, :n_components].T
        self.explained_variance_ = eigenvalues[:n_components]
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        if total_variance > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros_like(self.explained_variance_)
        
        # Apply whitening if requested
        if self.whiten:
            self.components_ = self.components_ / np.sqrt(self.explained_variance_[:, np.newaxis])
        
        return self
    
    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        
        Args:
            X: Input data, shape (n_samples, n_features)
            
        Returns:
            Transformed data, shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before transform")
        
        # Convert sparse matrix to dense if needed
        if sp.issparse(X):
            X = X.toarray()
        
        X = np.asarray(X, dtype=np.float64)
        
        # Center the data
        X_centered = X - self.mean_
        
        # Project onto principal components
        X_transformed = np.dot(X_centered, self.components_.T)
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit PCA and transform the data.
        
        Args:
            X: Input data, shape (n_samples, n_features)
            
        Returns:
            Transformed data, shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """
        Transform data back to original space.
        
        Args:
            X: Transformed data, shape (n_samples, n_components)
            
        Returns:
            Data in original space, shape (n_samples, n_features)
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before inverse_transform")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Project back to original space
        X_original = np.dot(X, self.components_)
        
        # Add back the mean
        X_original = X_original + self.mean_
        
        return X_original


def apply_pca_to_ratings(rating_file, output_file, n_components=None, variance_threshold=0.95, 
                         use_sklearn=False, standardize=True):
    """
    Apply PCA to rating matrix for dimensionality reduction.
    
    Args:
        rating_file: Path to rating file (.npy or .txt)
        output_file: Path to save reduced rating matrix
        n_components: Number of components to keep (if None, use variance_threshold)
        variance_threshold: Minimum variance to retain (used if n_components is None)
        use_sklearn: Whether to use sklearn's PCA (faster) or custom implementation
        standardize: Whether to standardize features before PCA
        
    Returns:
        Reduced rating matrix and PCA model
    """
    print(f'Loading rating file: {rating_file} ...')
    
    # Load rating data
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    elif os.path.exists(rating_file + '.txt'):
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.float64)
    else:
        raise FileNotFoundError(f"Rating file not found: {rating_file}")
    
    print(f'Original rating matrix shape: {rating_np.shape}')
    
    # Create user-item matrix
    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    
    # Build user-item matrix
    user_item_matrix = np.zeros((n_user, n_item), dtype=np.float64)
    for row in rating_np:
        user_idx = int(row[0])
        item_idx = int(row[1])
        rating = row[2]
        user_item_matrix[user_idx, item_idx] = rating
    
    print(f'User-item matrix shape: {user_item_matrix.shape}')
    
    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        user_item_matrix = scaler.fit_transform(user_item_matrix)
    
    # Apply PCA
    if use_sklearn:
        pca = SklearnPCA(n_components=n_components)
        if n_components is None:
            # Determine n_components based on variance threshold
            pca.fit(user_item_matrix)
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
            pca = SklearnPCA(n_components=n_components)
    else:
        pca = PCA(n_components=n_components)
    
    # Fit and transform
    user_item_reduced = pca.fit_transform(user_item_matrix)
    
    print(f'Reduced matrix shape: {user_item_reduced.shape}')
    print(f'Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}')
    
    # Save reduced matrix
    np.save(output_file + '.npy', user_item_reduced)
    np.savetxt(output_file + '.txt', user_item_reduced, fmt='%.6f')
    print(f'Saved reduced rating matrix to: {output_file}')
    
    return user_item_reduced, pca


def apply_pca_to_kg_features(kg_feat_file, output_file, n_components=None, variance_threshold=0.95,
                            use_sklearn=False, standardize=True):
    """
    Apply PCA to knowledge graph feature matrix.
    
    Args:
        kg_feat_file: Path to KG feature file (.npz)
        output_file: Path to save reduced features
        n_components: Number of components to keep
        variance_threshold: Minimum variance to retain
        use_sklearn: Whether to use sklearn's PCA
        standardize: Whether to standardize features before PCA
        
    Returns:
        Reduced feature matrix and PCA model
    """
    print(f'Loading KG feature file: {kg_feat_file} ...')
    
    if not os.path.exists(kg_feat_file):
        print(f'KG feature file not found: {kg_feat_file}')
        return None, None
    
    # Load sparse matrix
    kg_feat = sp.load_npz(kg_feat_file)
    print(f'Original KG feature matrix shape: {kg_feat.shape}')
    
    is_sparse = sp.issparse(kg_feat)
    
    # For sklearn: use TruncatedSVD for sparse matrices, PCA for dense
    # For custom PCA: convert to dense (handles sparse internally but converts)
    if use_sklearn and is_sparse:
        # Use TruncatedSVD for sparse matrices (similar to PCA but supports sparse)
        print('Using TruncatedSVD for sparse matrix (sklearn PCA does not support sparse matrices).')
        
        # Standardize sparse matrix (mean centering)
        if standardize:
            # Center the sparse matrix efficiently
            # Compute mean per feature (column)
            mean = np.array(kg_feat.mean(axis=0)).flatten()
            # Create a sparse matrix with means repeated for each row
            # More efficient: subtract mean using sparse matrix operations
            from scipy.sparse import csr_matrix
            mean_matrix = csr_matrix(np.tile(mean, (kg_feat.shape[0], 1)))
            kg_feat_centered = kg_feat - mean_matrix
            print('Centered sparse matrix (mean subtraction).')
        else:
            kg_feat_centered = kg_feat
        
        # Determine number of components
        if n_components is None:
            # Fit with max components to determine variance
            max_components = min(kg_feat.shape[0], kg_feat.shape[1], 1000)  # Limit for efficiency
            svd_temp = TruncatedSVD(n_components=max_components)
            svd_temp.fit(kg_feat_centered)
            # Calculate explained variance ratio
            explained_variance_ratio = svd_temp.explained_variance_ratio_
            cumsum_variance = np.cumsum(explained_variance_ratio)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
            if n_components == 0:
                n_components = min(50, max_components)  # Default to 50 if threshold not met
            print(f'Auto-determined n_components: {n_components} (variance threshold: {variance_threshold})')
        
        # Apply TruncatedSVD
        svd = TruncatedSVD(n_components=n_components)
        kg_feat_reduced = svd.fit_transform(kg_feat_centered)
        
        # Calculate explained variance ratio
        explained_variance_ratio = svd.explained_variance_ratio_
        pca = svd  # Store for return (TruncatedSVD has similar interface)
        
    else:
        # Convert to dense for custom PCA or sklearn PCA
        if is_sparse:
            print('Converting sparse matrix to dense for PCA...')
            # Check if conversion is feasible
            n_elements = kg_feat.shape[0] * kg_feat.shape[1]
            if n_elements > 1e8:  # More than 100M elements
                print(f'Warning: Large matrix ({n_elements} elements). This may use significant memory.')
            kg_feat_dense = kg_feat.toarray()
        else:
            kg_feat_dense = kg_feat
        
        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            kg_feat_dense = scaler.fit_transform(kg_feat_dense)
        
        # Apply PCA
        if use_sklearn:
            pca = SklearnPCA(n_components=n_components)
            if n_components is None:
                pca.fit(kg_feat_dense)
                cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
                pca = SklearnPCA(n_components=n_components)
        else:
            pca = PCA(n_components=n_components)
        
        # Fit and transform
        kg_feat_reduced = pca.fit_transform(kg_feat_dense)
    
    print(f'Reduced feature matrix shape: {kg_feat_reduced.shape}')
    if hasattr(pca, 'explained_variance_ratio_'):
        print(f'Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}')
    else:
        print('Note: Explained variance ratio not available')
    
    # Save reduced features
    np.save(output_file + '.npy', kg_feat_reduced)
    print(f'Saved reduced KG features to: {output_file}')
    
    return kg_feat_reduced, pca


def apply_pca_to_embeddings(embedding_file, output_file, n_components=None, variance_threshold=0.95,
                           use_sklearn=False, standardize=True):
    """
    Apply PCA to embedding vectors (user, entity, or relation embeddings).
    
    Args:
        embedding_file: Path to embedding file (.npy)
        output_file: Path to save reduced embeddings
        n_components: Number of components to keep
        variance_threshold: Minimum variance to retain
        use_sklearn: Whether to use sklearn's PCA
        standardize: Whether to standardize features before PCA
        
    Returns:
        Reduced embeddings and PCA model
    """
    print(f'Loading embedding file: {embedding_file} ...')
    
    if not os.path.exists(embedding_file):
        print(f'Embedding file not found: {embedding_file}')
        return None, None
    
    embeddings = np.load(embedding_file)
    print(f'Original embedding shape: {embeddings.shape}')
    
    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    
    # Apply PCA
    if use_sklearn:
        pca = SklearnPCA(n_components=n_components)
        if n_components is None:
            pca.fit(embeddings)
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
            pca = SklearnPCA(n_components=n_components)
    else:
        pca = PCA(n_components=n_components)
    
    # Fit and transform
    embeddings_reduced = pca.fit_transform(embeddings)
    
    print(f'Reduced embedding shape: {embeddings_reduced.shape}')
    print(f'Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}')
    
    # Save reduced embeddings
    np.save(output_file, embeddings_reduced)
    print(f'Saved reduced embeddings to: {output_file}')
    
    return embeddings_reduced, pca


def main():
    """
    Main function for command-line usage of PCA preprocessing.
    """
    parser = argparse.ArgumentParser(description='PCA Preprocessing for Recommendation System')
    parser.add_argument('-d', '--dataset', type=str, default='movie', 
                       help='Dataset name (movie, book, music, restaurant)')
    parser.add_argument('-t', '--type', type=str, choices=['ratings', 'kg_features', 'embeddings'],
                       required=True, help='Type of data to preprocess')
    parser.add_argument('-i', '--input', type=str, help='Input file path (optional, uses default if not specified)')
    parser.add_argument('-o', '--output', type=str, help='Output file path (optional, uses default if not specified)')
    parser.add_argument('-n', '--n_components', type=int, default=None,
                       help='Number of principal components to keep')
    parser.add_argument('-v', '--variance_threshold', type=float, default=0.95,
                       help='Minimum variance to retain (used if n_components is None)')
    parser.add_argument('--use_sklearn', action='store_true',
                       help='Use sklearn PCA instead of custom implementation')
    parser.add_argument('--no_standardize', action='store_true',
                       help='Do not standardize features before PCA')
    
    args = parser.parse_args()
    
    standardize = not args.no_standardize
    
    if args.type == 'ratings':
        if args.input:
            input_file = args.input
        else:
            input_file = f'../data/{args.dataset}/ratings_final'
        
        if args.output:
            output_file = args.output
        else:
            output_file = f'../data/{args.dataset}/ratings_final_pca'
        
        apply_pca_to_ratings(input_file, output_file, args.n_components, 
                           args.variance_threshold, args.use_sklearn, standardize)
    
    elif args.type == 'kg_features':
        if args.input:
            input_file = args.input
        else:
            input_file = f'../data/{args.dataset}/kg_feat.npz'
        
        if args.output:
            output_file = args.output
        else:
            output_file = f'../data/{args.dataset}/kg_feat_pca'
        
        apply_pca_to_kg_features(input_file, output_file, args.n_components,
                                args.variance_threshold, args.use_sklearn, standardize)
    
    elif args.type == 'embeddings':
        if not args.input:
            raise ValueError("Input file must be specified for embeddings")
        if not args.output:
            raise ValueError("Output file must be specified for embeddings")
        
        apply_pca_to_embeddings(args.input, args.output, args.n_components,
                               args.variance_threshold, args.use_sklearn, standardize)
    
    print('PCA preprocessing completed!')


if __name__ == '__main__':
    main()

