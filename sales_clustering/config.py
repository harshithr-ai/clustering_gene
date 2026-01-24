"""
Configuration Module for Enterprise Clustering Platform
Defines statistical parameters, thresholds, and business metrics
"""

# Statistical Significance Thresholds
STATISTICAL_SIGNIFICANCE_ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95
EFFECT_SIZE_THRESHOLDS = {
    'small': 0.2,
    'medium': 0.5,
    'large': 0.8
}

# Clustering Quality Metrics
SILHOUETTE_EXCELLENT = 0.7
SILHOUETTE_GOOD = 0.5
SILHOUETTE_FAIR = 0.25

# Business Impact Thresholds
HIGH_IMPACT_THRESHOLD = 0.25  # 25% deviation from baseline
MODERATE_IMPACT_THRESHOLD = 0.10  # 10% deviation
ACTIONABLE_SEGMENT_MIN_SIZE = 0.05  # 5% of population

# Outlier Detection
OUTLIER_Z_SCORE_THRESHOLD = 3
OUTLIER_IQR_MULTIPLIER = 1.5

# Model Performance
MIN_CLUSTER_SIZE = 10
MAX_CLUSTER_RATIO = 0.8  # Max 80% in single cluster

# Visualization Settings
COLOR_SCHEMES = {
    'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'executive': ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'],
    'diverging': ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
}

# Statistical Tests
NORMALITY_TEST_THRESHOLD = 0.05
HOMOGENEITY_TEST_THRESHOLD = 0.05

# Industry-Specific Configurations
INDUSTRY_CONFIGS = {
    'default': {
        'segment_terminology': 'Segment',
        'priority_metrics': ['variance_explained', 'separation_index'],
        'insight_focus': 'general'
    },
    'retail': {
        'segment_terminology': 'Customer Cohort',
        'priority_metrics': ['within_cluster_variance', 'segment_density'],
        'insight_focus': 'customer_value'
    },
    'finance': {
        'segment_terminology': 'Risk Profile',
        'priority_metrics': ['statistical_power', 'effect_size'],
        'insight_focus': 'risk_assessment'
    },
    'healthcare': {
        'segment_terminology': 'Patient Cluster',
        'priority_metrics': ['homogeneity_index', 'separation_ratio'],
        'insight_focus': 'outcome_prediction'
    },
    'manufacturing': {
        'segment_terminology': 'Process Group',
        'priority_metrics': ['variance_reduction', 'optimization_potential'],
        'insight_focus': 'efficiency'
    }
}

# Executive Reporting Templates
EXECUTIVE_METRICS = [
    'total_variance_explained',
    'between_cluster_variance_ratio',
    'statistical_significance',
    'segment_stability_index',
    'actionable_insights_count',
    'business_impact_score'
]

# Feature Importance Calculation
FEATURE_IMPORTANCE_METHODS = ['variance', 'anova', 'mutual_information']

# Data Quality Thresholds
MIN_DATA_QUALITY_SCORE = 0.7
MAX_MISSING_RATE = 0.3
MIN_FEATURE_VARIANCE = 0.01
