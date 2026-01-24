"""
Visualization Module
Professional visualizations for executive presentations
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import config


class VisualizationEngine:
    """Professional visualization engine for clustering results"""
    
    def __init__(self):
        self.colors = config.COLOR_SCHEMES['executive']
        
    def create_segment_overview(self, labels: np.ndarray, segment_term: str = 'Segment') -> go.Figure:
        """Create segment size distribution visualization"""
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        
        # Calculate percentages
        total = len(labels[labels != -1])
        percentages = (counts / total) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f'{segment_term} {i}' for i in unique_labels],
            y=counts,
            text=[f'{p:.1f}%' for p in percentages],
            textposition='auto',
            marker=dict(
                color=self.colors[:len(unique_labels)],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{x}</b><br>' +
                         'Count: %{y}<br>' +
                         'Percentage: %{text}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'{segment_term} Distribution Analysis',
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title=segment_term,
            yaxis_title='Population Count',
            template='plotly_white',
            height=400,
            showlegend=False,
            font=dict(size=12)
        )
        
        return fig
    
    def create_pca_visualization(self, X: np.ndarray, labels: np.ndarray, 
                                 segment_term: str = 'Segment') -> go.Figure:
        """Create 2D PCA visualization with professional styling"""
        from sklearn.decomposition import PCA
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        variance_explained = pca.explained_variance_ratio_
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Segment': [f'{segment_term} {l}' if l != -1 else 'Noise' for l in labels]
        })
        
        fig = px.scatter(
            df,
            x='PC1',
            y='PC2',
            color='Segment',
            color_discrete_sequence=self.colors,
            title=f'{segment_term} Visualization (PCA Projection)',
            template='plotly_white',
            height=500
        )
        
        fig.update_traces(
            marker=dict(size=8, line=dict(width=1, color='white')),
            opacity=0.7
        )
        
        fig.update_layout(
            xaxis_title=f'First Principal Component ({variance_explained[0]*100:.1f}% variance)',
            yaxis_title=f'Second Principal Component ({variance_explained[1]*100:.1f}% variance)',
            title=dict(x=0.5, xanchor='center', font=dict(size=18)),
            legend=dict(
                title=dict(text=segment_term, font=dict(size=14)),
                font=dict(size=12)
            ),
            font=dict(size=12)
        )
        
        return fig
    
    def create_performance_dashboard(self, metrics: Dict, algorithm_name: str) -> go.Figure:
        """Create comprehensive performance metrics dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Quality Metrics',
                'Variance Analysis',
                'Separation & Balance',
                'Statistical Validation'
            ),
            specs=[
                [{'type': 'indicator'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'indicator'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Silhouette Score Gauge
        silhouette = metrics.get('silhouette_score', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=silhouette,
                title={'text': "Silhouette Score", 'font': {'size': 14}},
                delta={'reference': 0.5},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': self._get_quality_color(silhouette)},
                    'steps': [
                        {'range': [-1, 0], 'color': "#ffcccc"},
                        {'range': [0, 0.25], 'color': "#ffffcc"},
                        {'range': [0.25, 0.5], 'color': "#ccffcc"},
                        {'range': [0.5, 1], 'color': "#99ff99"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ),
            row=1, col=1
        )
        
        # Variance Explained
        r_squared = metrics.get('r_squared', 0) * 100
        wcss = metrics.get('wcss', 0)
        bcss = metrics.get('bcss', 0)
        
        fig.add_trace(
            go.Bar(
                x=['Between-Cluster', 'Within-Cluster'],
                y=[bcss, wcss],
                text=[f'{bcss:.0f}', f'{wcss:.0f}'],
                textposition='auto',
                marker=dict(color=['#2ca02c', '#d62728']),
                name='Variance'
            ),
            row=1, col=2
        )
        
        # Separation and Balance
        sep_index = metrics.get('separation_index', 0)
        balance_score = metrics.get('balance_score', 0)
        
        fig.add_trace(
            go.Bar(
                x=['Separation Index', 'Balance Score'],
                y=[min(sep_index, 5), balance_score],  # Cap separation at 5 for visualization
                text=[f'{sep_index:.2f}', f'{balance_score:.2f}'],
                textposition='auto',
                marker=dict(color=['#ff7f0e', '#9467bd']),
                name='Structure'
            ),
            row=2, col=1
        )
        
        # R-squared Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=r_squared,
                title={'text': "Variance Explained (%)", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': self._get_quality_color(r_squared / 100)},
                    'steps': [
                        {'range': [0, 40], 'color': "#ffcccc"},
                        {'range': [40, 60], 'color': "#ffffcc"},
                        {'range': [60, 80], 'color': "#ccffcc"},
                        {'range': [80, 100], 'color': "#99ff99"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=dict(
                text=f'{algorithm_name} - Performance Metrics Dashboard',
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Arial Black')
            ),
            showlegend=False,
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance: Dict) -> go.Figure:
        """Create feature importance visualization"""
        anova_ranking = feature_importance.get('anova_ranking', [])
        
        if not anova_ranking:
            return None
        
        # Take top 10 features
        top_features = anova_ranking[:10]
        
        features = [f['feature'] for f in top_features]
        f_stats = [f['f_statistic'] for f in top_features]
        is_sig = [f['is_significant'] for f in top_features]
        
        colors_list = ['#2ca02c' if sig else '#d62728' for sig in is_sig]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features,
            x=f_stats,
            orientation='h',
            marker=dict(
                color=colors_list,
                line=dict(color='white', width=1)
            ),
            text=[f'{stat:.1f}' for stat in f_stats],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>' +
                         'F-Statistic: %{x:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='Feature Importance for Segment Separation',
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            xaxis_title='F-Statistic (ANOVA)',
            yaxis_title='',
            template='plotly_white',
            height=400,
            showlegend=False,
            font=dict(size=11)
        )
        
        # Reverse y-axis for better reading
        fig.update_yaxes(autorange="reversed")
        
        return fig
    
    def create_segment_comparison_matrix(self, segment_profiles: Dict, 
                                        top_n_features: int = 5) -> go.Figure:
        """Create heatmap comparing segments across features"""
        if not segment_profiles:
            return None
        
        # Extract data
        segments = list(segment_profiles.keys())
        
        # Get top features across all segments
        all_features = {}
        for segment, profile in segment_profiles.items():
            for feat in profile.get('distinguishing_features', []):
                fname = feat['feature']
                if fname not in all_features:
                    all_features[fname] = []
                all_features[fname].append(abs(feat['effect_size']))
        
        # Get top N features by average effect size
        top_features = sorted(
            all_features.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True
        )[:top_n_features]
        
        feature_names = [f[0] for f in top_features]
        
        # Build matrix
        matrix = []
        for segment in segments:
            row = []
            profile = segment_profiles[segment]
            features_dict = {
                f['feature']: f['effect_size'] 
                for f in profile.get('distinguishing_features', [])
            }
            
            for fname in feature_names:
                row.append(features_dict.get(fname, 0))
            
            matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=feature_names,
            y=segments,
            colorscale='RdBu',
            zmid=0,
            text=[[f'{val:.2f}' for val in row] for row in matrix],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Effect Size")
        ))
        
        fig.update_layout(
            title=dict(
                text='Segment Feature Profile Matrix',
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            xaxis_title='Key Differentiating Features',
            yaxis_title='Segments',
            template='plotly_white',
            height=400,
            font=dict(size=11)
        )
        
        return fig
    
    def _get_quality_color(self, score: float) -> str:
        """Get color based on quality score"""
        if score >= 0.7:
            return "#2ca02c"  # Green
        elif score >= 0.5:
            return "#ff7f0e"  # Orange
        else:
            return "#d62728"  # Red
    
    def create_elbow_plot(self, k_range: List[int], wcss_scores: List[float], 
                         optimal_k: int) -> go.Figure:
        """Create elbow plot for optimal K determination"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=k_range,
            y=wcss_scores,
            mode='lines+markers',
            name='WCSS',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10)
        ))
        
        # Mark optimal K
        optimal_idx = k_range.index(optimal_k)
        fig.add_trace(go.Scatter(
            x=[optimal_k],
            y=[wcss_scores[optimal_idx]],
            mode='markers',
            name='Optimal K',
            marker=dict(size=20, color='red', symbol='star')
        ))
        
        fig.update_layout(
            title=dict(
                text='Elbow Method for Optimal Cluster Count',
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Within-Cluster Sum of Squares (WCSS)',
            template='plotly_white',
            height=400,
            showlegend=True,
            font=dict(size=12)
        )
        
        return fig
    
    def create_silhouette_plot(self, k_range: List[int], silhouette_scores: List[float],
                              optimal_k: int) -> go.Figure:
        """Create silhouette score plot"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=k_range,
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=10)
        ))
        
        # Mark optimal K
        optimal_idx = k_range.index(optimal_k)
        fig.add_trace(go.Scatter(
            x=[optimal_k],
            y=[silhouette_scores[optimal_idx]],
            mode='markers',
            name='Optimal K',
            marker=dict(size=20, color='red', symbol='star')
        ))
        
        # Add reference lines
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                     annotation_text="Good Quality")
        fig.add_hline(y=0.25, line_dash="dash", line_color="orange",
                     annotation_text="Fair Quality")
        
        fig.update_layout(
            title=dict(
                text='Silhouette Analysis for Optimal Cluster Count',
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Silhouette Score',
            template='plotly_white',
            height=400,
            showlegend=True,
            font=dict(size=12)
        )
        
        return fig
