"""
Clustering Engine Module
Implements multiple clustering algorithms with statistical validation
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score
)
from scipy.spatial.distance import cdist
from scipy import stats
from typing import Dict, List, Tuple, Optional
import config


class ClusteringEngine:
    """Enterprise clustering engine with multiple algorithms and validation"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.validation_metrics = {}
        
    def fit_kmeans(self, X: np.ndarray, n_clusters: int, **kwargs) -> Dict:
        """
        K-Means clustering with enhanced validation
        
        Args:
            X: Feature matrix (n_samples, n_features)
            n_clusters: Number of clusters
        """
        model = KMeans(
            n_clusters=n_clusters,
            n_init=20,  # Multiple initializations for stability
            max_iter=500,
            random_state=42,
            **kwargs
        )
        
        labels = model.fit_predict(X)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_clustering_metrics(X, labels, model.cluster_centers_)
        
        # Calculate variance explained
        total_variance = np.var(X, axis=0).sum()
        wcss = model.inertia_
        bcss = total_variance * len(X) - wcss
        variance_explained = bcss / (total_variance * len(X))
        
        metrics['variance_explained'] = variance_explained
        metrics['wcss'] = wcss
        metrics['bcss'] = bcss
        metrics['inertia'] = model.inertia_
        
        result = {
            'model': model,
            'labels': labels,
            'n_clusters': n_clusters,
            'centroids': model.cluster_centers_,
            'metrics': metrics,
            'algorithm': 'K-Means'
        }
        
        self.models['K-Means'] = model
        self.results['K-Means'] = result
        
        return result
    
    def fit_hierarchical(self, X: np.ndarray, n_clusters: int, 
                        linkage: str = 'ward', **kwargs) -> Dict:
        """
        Hierarchical Agglomerative Clustering
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            linkage: Linkage method ('ward', 'complete', 'average', 'single')
        """
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            **kwargs
        )
        
        labels = model.fit_predict(X)
        
        # Calculate centroids manually for hierarchical
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        
        metrics = self._calculate_clustering_metrics(X, labels, centroids)
        
        result = {
            'model': model,
            'labels': labels,
            'n_clusters': n_clusters,
            'centroids': centroids,
            'metrics': metrics,
            'algorithm': f'Hierarchical ({linkage})'
        }
        
        self.models['Hierarchical'] = model
        self.results['Hierarchical'] = result
        
        return result
    
    def fit_dbscan(self, X: np.ndarray, eps: float, min_samples: int, **kwargs) -> Dict:
        """
        DBSCAN clustering for density-based segmentation
        
        Args:
            X: Feature matrix
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
        """
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            **kwargs
        )
        
        labels = model.fit_predict(X)
        
        # Handle noise points (-1 label)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Calculate centroids (excluding noise)
        if n_clusters > 0:
            centroids = np.array([
                X[labels == i].mean(axis=0) 
                for i in range(n_clusters)
            ])
            
            # Calculate metrics only for non-noise points
            non_noise_mask = labels != -1
            if non_noise_mask.sum() > 1 and n_clusters > 1:
                metrics = self._calculate_clustering_metrics(
                    X[non_noise_mask], 
                    labels[non_noise_mask], 
                    centroids
                )
            else:
                metrics = self._get_empty_metrics()
        else:
            centroids = None
            metrics = self._get_empty_metrics()
        
        metrics['n_noise_points'] = n_noise
        metrics['noise_ratio'] = n_noise / len(X)
        
        result = {
            'model': model,
            'labels': labels,
            'n_clusters': n_clusters,
            'centroids': centroids,
            'metrics': metrics,
            'algorithm': 'DBSCAN',
            'n_noise': n_noise
        }
        
        self.models['DBSCAN'] = model
        self.results['DBSCAN'] = result
        
        return result
    
    def fit_gaussian_mixture(self, X: np.ndarray, n_components: int, **kwargs) -> Dict:
        """
        Gaussian Mixture Model for probabilistic clustering
        
        Args:
            X: Feature matrix
            n_components: Number of mixture components
        """
        model = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            n_init=10,
            random_state=42,
            **kwargs
        )
        
        model.fit(X)
        labels = model.predict(X)
        
        metrics = self._calculate_clustering_metrics(X, labels, model.means_)
        
        # Add GMM-specific metrics
        metrics['bic'] = model.bic(X)
        metrics['aic'] = model.aic(X)
        metrics['log_likelihood'] = model.score(X) * len(X)
        
        result = {
            'model': model,
            'labels': labels,
            'n_clusters': n_components,
            'centroids': model.means_,
            'metrics': metrics,
            'algorithm': 'Gaussian Mixture',
            'probabilities': model.predict_proba(X)
        }
        
        self.models['GMM'] = model
        self.results['GMM'] = result
        
        return result
    
    def _calculate_clustering_metrics(self, X: np.ndarray, labels: np.ndarray, 
                                     centroids: np.ndarray) -> Dict:
        """Calculate comprehensive clustering validation metrics"""
        metrics = {}
        
        n_clusters = len(np.unique(labels))
        n_samples = len(X)
        
        # Basic validation - need at least 2 clusters and reasonable number of samples
        if n_clusters <= 1 or n_samples < 2:
            return self._get_empty_metrics()
        
        try:
            # Silhouette Score (-1 to 1, higher is better)
            silhouette_avg = silhouette_score(X, labels)
            metrics['silhouette_score'] = silhouette_avg
            
            # Silhouette samples for stability analysis
            silhouette_vals = silhouette_samples(X, labels)
            metrics['silhouette_std'] = np.std(silhouette_vals)
            
            # Calinski-Harabasz Index (higher is better)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            
            # Davies-Bouldin Index (lower is better)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
            
            # Within-cluster sum of squares
            wcss = 0
            for i in range(n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    wcss += np.sum((cluster_points - centroids[i]) ** 2)
            metrics['wcss'] = wcss
            
            # Between-cluster sum of squares
            overall_mean = np.mean(X, axis=0)
            bcss = 0
            for i in range(n_clusters):
                n_i = np.sum(labels == i)
                bcss += n_i * np.sum((centroids[i] - overall_mean) ** 2)
            metrics['bcss'] = bcss
            
            # Total sum of squares
            tss = np.sum((X - overall_mean) ** 2)
            metrics['tss'] = tss
            
            # R-squared (variance explained)
            metrics['r_squared'] = bcss / tss if tss > 0 else 0
            
            # Separation Index (higher is better)
            if n_clusters > 1:
                min_centroid_dist = np.min(cdist(centroids, centroids) + np.eye(n_clusters) * 1e10)
                avg_cluster_radius = np.sqrt(wcss / n_samples)
                metrics['separation_index'] = min_centroid_dist / (avg_cluster_radius + 1e-10)
            else:
                metrics['separation_index'] = 0
            
            # Cluster balance (entropy-based)
            cluster_sizes = np.array([np.sum(labels == i) for i in range(n_clusters)])
            cluster_props = cluster_sizes / n_samples
            entropy = -np.sum(cluster_props * np.log(cluster_props + 1e-10))
            max_entropy = np.log(n_clusters)
            metrics['balance_score'] = entropy / max_entropy if max_entropy > 0 else 0
            
            # Dunn Index (ratio of minimum inter-cluster to maximum intra-cluster distance)
            metrics['dunn_index'] = self._calculate_dunn_index(X, labels, n_clusters)
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return self._get_empty_metrics()
        
        return metrics
    
    def _calculate_dunn_index(self, X: np.ndarray, labels: np.ndarray, n_clusters: int) -> float:
        """Calculate Dunn Index (higher is better)"""
        try:
            # Minimum inter-cluster distance
            min_inter_dist = float('inf')
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    dist = np.min(cdist(X[labels == i], X[labels == j]))
                    min_inter_dist = min(min_inter_dist, dist)
            
            # Maximum intra-cluster distance
            max_intra_dist = 0
            for i in range(n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 1:
                    intra_dist = np.max(cdist(cluster_points, cluster_points))
                    max_intra_dist = max(max_intra_dist, intra_dist)
            
            dunn = min_inter_dist / (max_intra_dist + 1e-10)
            return dunn
        except:
            return 0.0
    
    def _get_empty_metrics(self) -> Dict:
        """Return empty metrics dict for failed clustering"""
        return {
            'silhouette_score': np.nan,
            'calinski_harabasz_score': np.nan,
            'davies_bouldin_score': np.nan,
            'wcss': np.nan,
            'bcss': np.nan,
            'r_squared': np.nan,
            'separation_index': np.nan,
            'balance_score': np.nan
        }
    
    def find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 11)) -> Dict:
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        
        Returns:
            Dictionary with optimal k and scoring metrics
        """
        wcss_scores = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            wcss_scores.append(kmeans.inertia_)
            
            if k > 1:
                silhouette_scores.append(silhouette_score(X, labels))
            else:
                silhouette_scores.append(0)
        
        # Find elbow point
        elbow_k = self._find_elbow_point(list(k_range), wcss_scores)
        
        # Find best silhouette
        silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        return {
            'elbow_k': elbow_k,
            'silhouette_k': silhouette_k,
            'wcss_scores': wcss_scores,
            'silhouette_scores': silhouette_scores,
            'k_range': list(k_range)
        }
    
    def _find_elbow_point(self, k_values: List[int], scores: List[float]) -> int:
        """Find elbow point using distance method"""
        k_values = np.array(k_values)
        scores = np.array(scores)
        
        # Normalize
        k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        
        # Calculate distances
        distances = []
        for i in range(len(k_values)):
            point = np.array([k_norm[i], scores_norm[i]])
            line_start = np.array([k_norm[0], scores_norm[0]])
            line_end = np.array([k_norm[-1], scores_norm[-1]])
            
            dist = np.abs(np.cross(line_end - line_start, line_start - point)) / \
                   np.linalg.norm(line_end - line_start)
            distances.append(dist)
        
        elbow_idx = np.argmax(distances)
        return k_values[elbow_idx]
    
    def calculate_stability(self, X: np.ndarray, labels: np.ndarray, n_iterations: int = 10) -> float:
        """
        Calculate clustering stability using bootstrap resampling
        
        Returns:
            Stability score (0-1, higher is better)
        """
        n_samples = len(X)
        n_clusters = len(np.unique(labels))
        
        agreement_scores = []
        
        for _ in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            
            # Recluster
            kmeans = KMeans(n_clusters=n_clusters, random_state=None, n_init=10)
            labels_boot = kmeans.fit_predict(X_boot)
            
            # Calculate agreement with original labels
            original_labels_boot = labels[indices]
            
            # Adjusted Rand Index
            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(original_labels_boot, labels_boot)
            agreement_scores.append(ari)
        
        return np.mean(agreement_scores)
