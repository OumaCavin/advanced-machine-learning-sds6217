"""
Clustering and Unsupervised Learning Implementation

This module provides implementations of clustering algorithms including
K-Means and demonstrates concepts from unsupervised learning.

Author: Cavin Otieno Ouma
Registration: SDS6/46982/2024
Course: SDS 6217 Advanced Machine Learning
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler


class KMeans:
    """
    K-Means Clustering Implementation.
    
    As mentioned in the quiz: "Partitional Clustering is an unsupervised
    learning approach that entails division of data objects into
    non-overlapping subsets (clusters)."
    
    K-Means is the most common partitional clustering algorithm.
    """
    
    def __init__(self, n_clusters=3, max_iterations=300, random_seed=42):
        """
        Initialize K-Means clustering.
        
        Args:
            n_clusters: Number of clusters to form
            max_iterations: Maximum iterations for convergence
            random_seed: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.centroids = None
        self.labels = None
        self.inertia = None
        
    def initialize_centroids(self, X):
        """
        Initialize centroids using k-means++ method.
        
        Better initialization than random selection.
        """
        np.random.seed(self.random_seed)
        n_samples = len(X)
        
        # Select first centroid randomly from data
        centroids = [X[np.random.randint(n_samples)]]
        
        # Select remaining centroids using probability proportional to distance
        for _ in range(self.n_clusters - 1):
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) 
                                 for x in X])
            probabilities = distances / np.sum(distances)
            cumulative = np.cumsum(probabilities)
            r = np.random.rand()
            
            for i, p in enumerate(cumulative):
                if r < p:
                    centroids.append(X[i])
                    break
        
        return np.array(centroids)
    
    def assign_clusters(self, X, centroids):
        """
        Assign each point to nearest centroid.
        
        This is the "E" step in EM algorithm.
        """
        distances = np.array([[np.linalg.norm(x - c) for c in centroids] 
                             for x in X])
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        """
        Update centroids as mean of cluster points.
        
        This is the "M" step in EM algorithm.
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)
        
        return new_centroids
    
    def compute_inertia(self, X, centroids, labels):
        """
        Compute inertia (sum of squared distances to nearest centroid).
        """
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia
    
    def fit(self, X):
        """
        Fit K-Means clustering to data.
        
        Iterative process:
        1. Initialize centroids
        2. Assign points to nearest centroid
        3. Update centroids
        4. Repeat until convergence
        """
        np.random.seed(self.random_seed)
        X = np.array(X)
        
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        
        for iteration in range(self.max_iterations):
            # Assign clusters
            old_labels = self.labels.copy() if self.labels is not None else None
            self.labels = self.assign_clusters(X, self.centroids)
            
            # Update centroids
            new_centroids = self.update_centroids(X, self.labels)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged at iteration {iteration + 1}")
                break
            
            self.centroids = new_centroids
        
        # Compute final inertia
        self.inertia = self.compute_inertia(X, self.centroids, self.labels)
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        return self.assign_clusters(X, self.centroids)
    
    def fit_predict(self, X):
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels


def demonstrate_kmeans():
    """
    Demonstrate K-Means clustering algorithm.
    """
    print("\n" + "=" * 60)
    print("K-Means Clustering Demonstration")
    print("=" * 60)
    
    # Generate sample data with 3 clusters
    np.random.seed(42)
    X, true_labels = make_blobs(n_samples=300, centers=3, 
                                 cluster_std=1.0, random_state=42)
    
    print(f"\nGenerated {len(X)} data points with 3 true clusters")
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=3, random_seed=42)
    predicted_labels = kmeans.fit_predict(X)
    
    print(f"\nK-Means Results:")
    print(f"  Number of clusters: {kmeans.n_clusters}")
    print(f"  Inertia (within-cluster sum of squares): {kmeans.inertia:.2f}")
    print(f"  Final centroids:\n{kmeans.centroids}")
    
    # Calculate accuracy (using Hungarian algorithm would be better)
    # Simple comparison for demonstration
    from scipy.optimize import linear_sum_assignment
    
    # Create cost matrix
    cost_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cost_matrix[i, j] = np.sum((true_labels == i) & (predicted_labels == j))
    
    # Optimal assignment
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    
    # Calculate accuracy
    matched = sum(cost_matrix[r, c] for r, c in zip(row_ind, col_ind))
    accuracy = matched / len(true_labels)
    
    print(f"\nClustering accuracy (with optimal assignment): {accuracy:.2%}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # True labels
    axes[0].scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', 
                   marker='o', s=30, alpha=0.7)
    axes[0].set_title('True Clusters', fontsize=12)
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # Predicted labels
    axes[1].scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis',
                   marker='o', s=30, alpha=0.7)
    axes[1].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                   label='Centroids')
    axes[1].set_title('K-Means Clusters', fontsize=12)
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()
    
    # Cluster assignment visualization
    axes[2].scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis',
                   marker='o', s=30, alpha=0.7)
    for k in range(kmeans.n_clusters):
        # Draw cluster boundaries (Voronoi-like visualization)
        center = kmeans.centroids[k]
        axes[2].add_patch(plt.Circle((center[0], center[1]), 1.5,
                                     fill=False, edgecolor='red', 
                                     linestyle='--', linewidth=2))
    axes[2].set_title('Cluster Assignments', fontsize=12)
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('/workspace/advanced-machine-learning-sds6217/code_examples/kmeans_clustering.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to kmeans_clustering.png")


def compare_clustering_algorithms():
    """
    Compare different types of clustering algorithms.
    """
    print("\n" + "=" * 60)
    print("Comparing Clustering Algorithms")
    print("=" * 60)
    
    # Generate different types of cluster structures
    np.random.seed(42)
    
    # Blobs (spherical clusters)
    X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, 
                           random_state=42)
    
    # Moons (non-convex clusters)
    X_moons, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    # Scale data
    X_blobs = StandardScaler().fit_transform(X_blobs)
    X_moons = StandardScaler().fit_transform(X_moons)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # K-Means on blobs
    kmeans_blobs = KMeans(n_clusters=3, random_seed=42)
    labels_blobs = kmeans_blobs.fit_predict(X_blobs)
    
    axes[0, 0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels_blobs, 
                       cmap='viridis', marker='o', s=30, alpha=0.7)
    axes[0, 0].scatter(kmeans_blobs.centroids[:, 0], kmeans_blobs.centroids[:, 1],
                       c='red', marker='X', s=200, edgecolors='black')
    axes[0, 0].set_title('K-Means on Blobs\n(Spherical clusters)', fontsize=11)
    
    # K-Means on moons
    kmeans_moons = KMeans(n_clusters=2, random_seed=42)
    labels_moons = kmeans_moons.fit_predict(X_moons)
    
    axes[0, 1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons,
                       cmap='viridis', marker='o', s=30, alpha=0.7)
    axes[0, 1].scatter(kmeans_moons.centroids[:, 0], kmeans_moons.centroids[:, 1],
                       c='red', marker='X', s=200, edgecolors='black')
    axes[0, 1].set_title('K-Means on Moons\n(Non-convex clusters)', fontsize=11)
    
    # Show K-Means limitation
    axes[0, 2].scatter(X_moons[:, 0], X_moons[:, 1], c='gray',
                       marker='o', s=30, alpha=0.5)
    axes[0, 2].annotate('K-Means fails on\nnon-convex shapes',
                       xy=(0, 0), fontsize=10, ha='center',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    axes[0, 2].set_title('K-Means Limitation', fontsize=11)
    
    # Demonstrate elbow method
    inertias = []
    K_range = range(1, 10)
    
    X_demo, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.0, 
                          random_state=42)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_seed=42)
        kmeans.fit(X_demo)
        inertias.append(kmeans.inertia)
    
    axes[1, 0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[1, 0].axvline(x=4, color='r', linestyle='--', label='Optimal k=4')
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Inertia')
    axes[1, 0].set_title('Elbow Method\n(Determining optimal k)', fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # K-Means on elongated clusters
    np.random.seed(42)
    X_elongated = np.vstack([
        np.random.normal(0, 0.5, (100, 2)) + np.array([0, 0]),
        np.random.normal(0, 0.3, (100, 2)) + np.array([5, 0]),
        np.random.normal(0, 0.8, (100, 2)) + np.array([2.5, 5])
    ])
    
    kmeans_elongated = KMeans(n_clusters=3, random_seed=42)
    labels_elongated = kmeans_elongated.fit_predict(X_elongated)
    
    axes[1, 1].scatter(X_elongated[:, 0], X_elongated[:, 1], c=labels_elongated,
                       cmap='viridis', marker='o', s=30, alpha=0.7)
    axes[1, 1].scatter(kmeans_elongated.centroids[:, 0], kmeans_elongated.centroids[:, 1],
                       c='red', marker='X', s=200, edgecolors='black')
    axes[1, 1].set_title('K-Means on Elongated Clusters', fontsize=11)
    
    # Summary
    axes[1, 2].axis('off')
    summary_text = """
    Clustering Summary
    
    Partitional Clustering (K-Means):
    - Divides data into k non-overlapping clusters
    - Each point belongs to exactly one cluster
    - Works well for spherical/globular clusters
    - Sensitive to initialization and outliers
    
    Hierarchical Clustering:
    - Creates tree-like structure of clusters
    - No need to pre-specify number of clusters
    
    Density-Based Clustering (DBSCAN):
    - Finds clusters of arbitrary shape
    - Can identify outliers as noise
    - No need to specify number of clusters
    """
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/workspace/advanced-machine-learning-sds6217/code_examples/clustering_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to clustering_comparison.png")


def unsupervised_vs_supervised():
    """
    Compare supervised and unsupervised learning paradigms.
    """
    print("\n" + "=" * 60)
    print("Supervised vs Unsupervised Learning")
    print("=" * 60)
    
    print("\nAs mentioned in the quiz:")
    print("'Partitional Clustering is an unsupervised learning approach'")
    
    print("\n" + "-" * 60)
    print("Supervised Learning:")
    print("-" * 60)
    print("Requires: Labeled data (input-output pairs)")
    print("Goal: Learn mapping from input to output")
    print("Examples: Classification, Regression")
    print("SVM, Neural Networks with labels, Decision Trees")
    
    print("\n" + "-" * 60)
    print("Unsupervised Learning:")
    print("-" * 60)
    print("Requires: Unlabeled data (input only)")
    print("Goal: Find structure/patterns in data")
    print("Examples: Clustering, Dimensionality Reduction")
    print("K-Means, PCA, Autoencoders, DBSCAN")
    
    print("\n" + "-" * 60)
    print("Which from the quiz is NOT unsupervised?")
    print("-" * 60)
    print("Answer: Support Vector Machines (SVM)")
    print("  - SVM requires labeled training data")
    print("  - It's a supervised classification algorithm")
    print("  - K-Means, PCA, and EM are unsupervised")
    
    print("\n" + "=" * 60)


def demonstrate_pca():
    """
    Demonstrate Principal Component Analysis (PCA).
    
    PCA is an unsupervised learning method for dimensionality reduction.
    """
    print("\n" + "=" * 60)
    print("Principal Component Analysis (PCA)")
    print("=" * 60)
    
    print("\nPCA is an unsupervised learning technique that:")
    print("- Reduces dimensionality of data")
    print("- Preserves maximum variance")
    print("- Finds orthogonal directions of maximum variation")
    
    # Generate high-dimensional data
    np.random.seed(42)
    n_samples = 200
    
    # Create data with inherent 2D structure in 4D space
    t = np.linspace(0, 4 * np.pi, n_samples)
    # Primary variation (PC1)
    x1 = np.sin(t) + np.random.normal(0, 0.1, n_samples)
    x2 = np.cos(t) + np.random.normal(0, 0.1, n_samples)
    # Secondary variation (PC2)
    x3 = np.sin(2*t) + np.random.normal(0, 0.1, n_samples)
    x4 = np.cos(2*t) + np.random.normal(0, 0.1, n_samples)
    
    X = np.column_stack([x1, x2, x3, x4])
    
    # Apply PCA using numpy
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project data onto first 2 principal components
    X_pca = X_centered @ eigenvectors[:, :2]
    
    # Calculate explained variance
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    
    print(f"\nOriginal data shape: {X.shape} (200 samples, 4 features)")
    print(f"Reduced data shape: {X_pca.shape} (200 samples, 2 features)")
    print(f"\nExplained Variance Ratio:")
    print(f"  PC1: {explained_variance_ratio[0]:.2%}")
    print(f"  PC2: {explained_variance_ratio[1]:.2%}")
    print(f"  PC3: {explained_variance_ratio[2]:.2%}")
    print(f"  PC4: {explained_variance_ratio[3]:.2%}")
    print(f"\nTotal variance explained by first 2 PCs: {sum(explained_variance_ratio[:2]):.2%}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data (2D projection)
    axes[0].scatter(X[:, 0], X[:, 1], c=t, cmap='viridis', marker='o', s=30)
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].set_title('Original Data (Features 1 & 2)')
    axes[0].colorbar = plt.colorbar(axes[0].collections[0], ax=axes[0], label='t')
    
    # PCA projection
    im = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=t, cmap='viridis', marker='o', s=30)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('PCA Projection (PC1 & PC2)')
    plt.colorbar(im, ax=axes[1], label='t')
    
    # Explained variance
    axes[2].bar(range(1, 5), explained_variance_ratio * 100, color='steelblue', edgecolor='black')
    axes[2].set_xlabel('Principal Component')
    axes[2].set_ylabel('Explained Variance (%)')
    axes[2].set_title('Explained Variance by Component')
    axes[2].set_xticks(range(1, 5))
    
    # Add cumulative line
    cumulative = np.cumsum(explained_variance_ratio) * 100
    axes[2].plot(range(1, 5), cumulative, 'ro-', linewidth=2, label='Cumulative')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('/workspace/advanced-machine-learning-sds6217/code_examples/pca_demonstration.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to pca_demonstration.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Clustering and Unsupervised Learning Demonstration")
    print("Course: SDS 6217 Advanced Machine Learning")
    print("Student: Cavin Otieno Ouma")
    print("=" * 60)
    
    # K-Means demonstration
    print("\n1. Demonstrating K-Means Clustering...")
    demonstrate_kmeans()
    
    # Compare algorithms
    print("\n2. Comparing Clustering Algorithms...")
    compare_clustering_algorithms()
    
    # Supervised vs Unsupervised
    print("\n3. Explaining Supervised vs Unsupervised Learning...")
    unsupervised_vs_supervised()
    
    # PCA demonstration
    print("\n4. Demonstrating PCA (Unsupervised Dimensionality Reduction)...")
    demonstrate_pca()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)
