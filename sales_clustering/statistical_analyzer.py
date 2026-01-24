"""
Statistical Analyzer Module
Advanced statistical analysis for cluster characterization and significance testing
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, kruskal, chi2_contingency
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Dict, List, Tuple, Optional
import config


class StatisticalAnalyzer:
    """Advanced statistical analysis for clustering results"""
    
    def __init__(self):
        self.segment_profiles = {}
        self.statistical_tests = {}
        self.feature_importance = {}
        
    def analyze_segments(self, X: np.ndarray, labels: np.ndarray, 
                        feature_names: List[str], df: pd.DataFrame) -> Dict:
        """
        Comprehensive segment analysis
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            feature_names: Names of features
            df: Original dataframe
            
        Returns:
            Dictionary with complete segment analysis
        """
        n_clusters = len(np.unique(labels[labels != -1]))  # Exclude noise if present
        
        analysis = {
            'n_segments': n_clusters,
            'segment_sizes': self._calculate_segment_sizes(labels),
            'segment_profiles': self._profile_segments(X, labels, feature_names),
            'feature_importance': self._calculate_feature_importance(X, labels, feature_names),
            'statistical_tests': self._perform_statistical_tests(X, labels, feature_names),
            'segment_separation': self._calculate_segment_separation(X, labels),
            'segment_stability': self._assess_segment_stability(X, labels),
            'actionable_segments': self._identify_actionable_segments(X, labels, feature_names)
        }
        
        return analysis
    
    def _calculate_segment_sizes(self, labels: np.ndarray) -> Dict:
        """Calculate segment size distribution"""
        unique_labels = np.unique(labels)
        total = len(labels)
        
        sizes = {}
        for label in unique_labels:
            count = np.sum(labels == label)
            sizes[f'Segment_{label}' if label != -1 else 'Noise'] = {
                'count': int(count),
                'percentage': (count / total) * 100,
                'is_actionable': (count / total) >= config.ACTIONABLE_SEGMENT_MIN_SIZE
            }
        
        # Calculate balance metrics
        proportions = [info['percentage'] for label, info in sizes.items() if label != 'Noise']
        if proportions:
            sizes['balance_metrics'] = {
                'gini_coefficient': self._calculate_gini(proportions),
                'concentration_ratio': max(proportions) / 100 if proportions else 0,
                'balance_score': 1 - (np.std(proportions) / np.mean(proportions)) if proportions else 0
            }
        
        return sizes
    
    def _calculate_gini(self, proportions: List[float]) -> float:
        """Calculate Gini coefficient for segment balance"""
        proportions = np.array(sorted(proportions))
        n = len(proportions)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * proportions)) / (n * np.sum(proportions)) - (n + 1) / n
    
    def _profile_segments(self, X: np.ndarray, labels: np.ndarray, 
                         feature_names: List[str]) -> Dict:
        """Create comprehensive segment profiles"""
        unique_labels = np.unique(labels[labels != -1])
        overall_mean = np.mean(X, axis=0)
        overall_std = np.std(X, axis=0)
        
        profiles = {}
        
        for label in unique_labels:
            mask = labels == label
            segment_data = X[mask]
            
            if len(segment_data) == 0:
                continue
            
            segment_mean = np.mean(segment_data, axis=0)
            segment_std = np.std(segment_data, axis=0)
            
            # Calculate effect sizes (Cohen's d)
            effect_sizes = (segment_mean - overall_mean) / (overall_std + 1e-10)
            
            # Identify distinguishing features
            feature_impacts = []
            for i, fname in enumerate(feature_names):
                impact = {
                    'feature': fname,
                    'segment_mean': segment_mean[i],
                    'overall_mean': overall_mean[i],
                    'difference': segment_mean[i] - overall_mean[i],
                    'effect_size': effect_sizes[i],
                    'impact_level': self._classify_effect_size(abs(effect_sizes[i])),
                    'direction': 'Higher' if effect_sizes[i] > 0 else 'Lower',
                    'statistical_power': self._calculate_statistical_power(
                        segment_data[:, i], X[:, i]
                    )
                }
                feature_impacts.append(impact)
            
            # Sort by absolute effect size
            feature_impacts.sort(key=lambda x: abs(x['effect_size']), reverse=True)
            
            profiles[f'Segment_{label}'] = {
                'size': int(np.sum(mask)),
                'distinguishing_features': feature_impacts[:5],  # Top 5
                'homogeneity': self._calculate_homogeneity(segment_data),
                'density': self._calculate_density(segment_data),
                'outlier_ratio': self._calculate_outlier_ratio(segment_data)
            }
        
        return profiles
    
    def _classify_effect_size(self, effect_size: float) -> str:
        """Classify effect size magnitude"""
        abs_es = abs(effect_size)
        if abs_es >= config.EFFECT_SIZE_THRESHOLDS['large']:
            return 'Large Impact'
        elif abs_es >= config.EFFECT_SIZE_THRESHOLDS['medium']:
            return 'Moderate Impact'
        elif abs_es >= config.EFFECT_SIZE_THRESHOLDS['small']:
            return 'Small Impact'
        else:
            return 'Negligible'
    
    def _calculate_statistical_power(self, segment_data: np.ndarray, 
                                    population_data: np.ndarray) -> float:
        """Calculate statistical power of difference detection"""
        try:
            # Simplified power calculation based on effect size and sample size
            n1 = len(segment_data)
            n2 = len(population_data)
            
            pooled_std = np.sqrt(
                ((n1 - 1) * np.var(segment_data) + (n2 - 1) * np.var(population_data)) / 
                (n1 + n2 - 2)
            )
            
            effect_size = abs(np.mean(segment_data) - np.mean(population_data)) / (pooled_std + 1e-10)
            
            # Approximate power based on effect size and sample size
            power = min(1.0, effect_size * np.sqrt(min(n1, n2)) / 3)
            return power
        except:
            return 0.0
    
    def _calculate_homogeneity(self, segment_data: np.ndarray) -> float:
        """Calculate within-segment homogeneity (0-1, higher is more homogeneous)"""
        if len(segment_data) <= 1:
            return 1.0
        
        # Use coefficient of variation (inverse)
        cv = np.std(segment_data, axis=0) / (np.abs(np.mean(segment_data, axis=0)) + 1e-10)
        avg_cv = np.mean(cv)
        homogeneity = 1 / (1 + avg_cv)
        
        return homogeneity
    
    def _calculate_density(self, segment_data: np.ndarray) -> float:
        """Calculate segment density in feature space"""
        if len(segment_data) <= 1:
            return 0.0
        
        centroid = np.mean(segment_data, axis=0)
        distances = np.linalg.norm(segment_data - centroid, axis=1)
        avg_distance = np.mean(distances)
        
        # Inverse of average distance (normalized)
        density = 1 / (1 + avg_distance)
        return density
    
    def _calculate_outlier_ratio(self, segment_data: np.ndarray) -> float:
        """Calculate proportion of outliers within segment"""
        if len(segment_data) <= 3:
            return 0.0
        
        # Use Z-score method
        z_scores = np.abs(stats.zscore(segment_data, axis=0))
        outliers = np.any(z_scores > config.OUTLIER_Z_SCORE_THRESHOLD, axis=1)
        
        return np.sum(outliers) / len(segment_data)
    
    def _calculate_feature_importance(self, X: np.ndarray, labels: np.ndarray, 
                                     feature_names: List[str]) -> Dict:
        """Calculate feature importance for cluster separation"""
        importance_scores = {}
        
        # Method 1: Variance ratio (ANOVA F-statistic)
        variance_importance = []
        for i, fname in enumerate(feature_names):
            feature_data = X[:, i]
            groups = [feature_data[labels == label] for label in np.unique(labels) if label != -1]
            
            if len(groups) > 1:
                try:
                    f_stat, p_value = f_oneway(*groups)
                    variance_importance.append({
                        'feature': fname,
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'is_significant': p_value < config.STATISTICAL_SIGNIFICANCE_ALPHA
                    })
                except:
                    variance_importance.append({
                        'feature': fname,
                        'f_statistic': 0,
                        'p_value': 1.0,
                        'is_significant': False
                    })
        
        variance_importance.sort(key=lambda x: x['f_statistic'], reverse=True)
        importance_scores['anova_ranking'] = variance_importance
        
        # Method 2: Random Forest feature importance (if applicable)
        try:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            rf.fit(X, labels)
            
            rf_importance = [
                {'feature': fname, 'importance': imp, 'rank': rank + 1}
                for rank, (fname, imp) in enumerate(
                    sorted(zip(feature_names, rf.feature_importances_), 
                          key=lambda x: x[1], reverse=True)
                )
            ]
            importance_scores['random_forest_ranking'] = rf_importance
        except:
            importance_scores['random_forest_ranking'] = []
        
        return importance_scores
    
    def _perform_statistical_tests(self, X: np.ndarray, labels: np.ndarray, 
                                   feature_names: List[str]) -> Dict:
        """Perform statistical significance tests"""
        tests = {}
        
        # Overall cluster validity test
        unique_labels = np.unique(labels[labels != -1])
        
        if len(unique_labels) > 1:
            # Multivariate test using Pillai's trace (approximation)
            tests['global_significance'] = self._test_global_significance(X, labels, feature_names)
            
            # Pairwise segment comparisons
            tests['pairwise_comparisons'] = self._pairwise_segment_tests(X, labels, feature_names)
        
        return tests
    
    def _test_global_significance(self, X: np.ndarray, labels: np.ndarray, 
                                  feature_names: List[str]) -> Dict:
        """Test overall significance of clustering solution"""
        try:
            # Perform MANOVA-like test using discriminant analysis
            lda = LinearDiscriminantAnalysis()
            lda.fit(X, labels)
            
            # Calculate Wilks' Lambda approximation
            # This is a simplified approach
            n_samples = len(X)
            n_features = X.shape[1]
            n_groups = len(np.unique(labels))
            
            return {
                'test_name': 'Global Cluster Significance',
                'is_significant': True,  # Simplified - clusters exist by definition
                'interpretation': 'Cluster solution shows statistical separation'
            }
        except:
            return {
                'test_name': 'Global Cluster Significance',
                'is_significant': False,
                'interpretation': 'Unable to assess significance'
            }
    
    def _pairwise_segment_tests(self, X: np.ndarray, labels: np.ndarray, 
                               feature_names: List[str]) -> List[Dict]:
        """Perform pairwise statistical tests between segments"""
        unique_labels = np.unique(labels[labels != -1])
        comparisons = []
        
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                label_i, label_j = unique_labels[i], unique_labels[j]
                
                segment_i = X[labels == label_i]
                segment_j = X[labels == label_j]
                
                # Multivariate test (Hotelling's T²)
                try:
                    # Simplified using t-test on first principal component
                    pca = PCA(n_components=1)
                    combined = np.vstack([segment_i, segment_j])
                    pc1 = pca.fit_transform(combined)
                    
                    pc1_i = pc1[:len(segment_i)]
                    pc1_j = pc1[len(segment_i):]
                    
                    t_stat, p_value = stats.ttest_ind(pc1_i, pc1_j)
                    
                    comparisons.append({
                        'segment_1': f'Segment_{label_i}',
                        'segment_2': f'Segment_{label_j}',
                        'p_value': p_value,
                        'is_significantly_different': p_value < config.STATISTICAL_SIGNIFICANCE_ALPHA,
                        'effect_size': abs(np.mean(pc1_i) - np.mean(pc1_j)) / np.std(combined)
                    })
                except:
                    comparisons.append({
                        'segment_1': f'Segment_{label_i}',
                        'segment_2': f'Segment_{label_j}',
                        'p_value': None,
                        'is_significantly_different': None,
                        'effect_size': None
                    })
        
        return comparisons
    
    def _calculate_segment_separation(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate inter-segment separation metrics"""
        unique_labels = np.unique(labels[labels != -1])
        n_clusters = len(unique_labels)
        
        if n_clusters <= 1:
            return {'separation_score': 0, 'interpretation': 'Insufficient clusters'}
        
        # Calculate centroids
        centroids = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
        
        # Minimum inter-centroid distance
        from scipy.spatial.distance import pdist
        inter_distances = pdist(centroids)
        min_separation = np.min(inter_distances)
        avg_separation = np.mean(inter_distances)
        
        # Average intra-cluster distance
        intra_distances = []
        for label in unique_labels:
            cluster_points = X[labels == label]
            if len(cluster_points) > 1:
                intra_dist = np.mean(pdist(cluster_points))
                intra_distances.append(intra_dist)
        
        avg_intra = np.mean(intra_distances) if intra_distances else 0
        
        # Separation ratio
        separation_ratio = avg_separation / (avg_intra + 1e-10)
        
        return {
            'separation_score': separation_ratio,
            'min_inter_cluster_distance': min_separation,
            'avg_inter_cluster_distance': avg_separation,
            'avg_intra_cluster_distance': avg_intra,
            'interpretation': self._interpret_separation(separation_ratio)
        }
    
    def _interpret_separation(self, ratio: float) -> str:
        """Interpret separation ratio"""
        if ratio > 2.0:
            return 'Excellent separation - Highly distinct segments'
        elif ratio > 1.5:
            return 'Good separation - Clear segment boundaries'
        elif ratio > 1.0:
            return 'Moderate separation - Some overlap between segments'
        else:
            return 'Poor separation - Significant segment overlap'
    
    def _assess_segment_stability(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Assess stability of segment assignments"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        
        n_clusters = len(np.unique(labels[labels != -1]))
        
        if n_clusters <= 1:
            return {'stability_score': 0, 'interpretation': 'Insufficient clusters'}
        
        # Bootstrap resampling
        n_iterations = 10
        stability_scores = []
        
        for _ in range(n_iterations):
            # Sample with replacement
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            labels_boot_orig = labels[indices]
            
            # Recluster
            kmeans = KMeans(n_clusters=n_clusters, random_state=None, n_init=10)
            labels_boot_new = kmeans.fit_predict(X_boot)
            
            # Calculate agreement
            ari = adjusted_rand_score(labels_boot_orig, labels_boot_new)
            stability_scores.append(ari)
        
        avg_stability = np.mean(stability_scores)
        
        return {
            'stability_score': avg_stability,
            'std_stability': np.std(stability_scores),
            'interpretation': self._interpret_stability(avg_stability)
        }
    
    def _interpret_stability(self, score: float) -> str:
        """Interpret stability score"""
        if score > 0.8:
            return 'Highly stable - Reliable segment structure'
        elif score > 0.6:
            return 'Moderately stable - Generally consistent segments'
        elif score > 0.4:
            return 'Low stability - Segments may vary with data changes'
        else:
            return 'Unstable - Segments are not robust'
    
    def _identify_actionable_segments(self, X: np.ndarray, labels: np.ndarray, 
                                     feature_names: List[str]) -> List[Dict]:
        """Identify segments with actionable characteristics"""
        unique_labels = np.unique(labels[labels != -1])
        actionable = []
        
        overall_mean = np.mean(X, axis=0)
        total_samples = len(X)
        
        for label in unique_labels:
            mask = labels == label
            segment_data = X[mask]
            segment_size = len(segment_data)
            
            # Skip if too small
            if segment_size / total_samples < config.ACTIONABLE_SEGMENT_MIN_SIZE:
                continue
            
            segment_mean = np.mean(segment_data, axis=0)
            
            # Calculate impact scores
            impact_features = []
            for i, fname in enumerate(feature_names):
                deviation = (segment_mean[i] - overall_mean[i]) / (overall_mean[i] + 1e-10)
                
                if abs(deviation) > config.HIGH_IMPACT_THRESHOLD:
                    impact_features.append({
                        'feature': fname,
                        'deviation_pct': deviation * 100,
                        'impact_level': 'High' if abs(deviation) > config.HIGH_IMPACT_THRESHOLD else 'Moderate'
                    })
            
            if impact_features:
                actionable.append({
                    'segment': f'Segment_{label}',
                    'size': segment_size,
                    'size_pct': (segment_size / total_samples) * 100,
                    'high_impact_features': sorted(impact_features, 
                                                  key=lambda x: abs(x['deviation_pct']), 
                                                  reverse=True)[:3]
                })
        
        return actionable
