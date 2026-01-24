"""
Enterprise Clustering Platform - Main Application
Production-grade clustering analysis for executive decision-making
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import DataProcessor
from clustering_engine import ClusteringEngine
from statistical_analyzer import StatisticalAnalyzer
from insight_generator import InsightGenerator
from visualizations import VisualizationEngine
import config

# Page configuration
st.set_page_config(
    page_title="Enterprise Clustering Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .executive-summary {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        margin: 1.5rem 0;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'clustering_engine' not in st.session_state:
    st.session_state.clustering_engine = ClusteringEngine()
if 'statistical_analyzer' not in st.session_state:
    st.session_state.statistical_analyzer = StatisticalAnalyzer()
if 'viz_engine' not in st.session_state:
    st.session_state.viz_engine = VisualizationEngine()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_preprocessed' not in st.session_state:
    st.session_state.data_preprocessed = False
if 'clustering_complete' not in st.session_state:
    st.session_state.clustering_complete = False
if 'selected_industry' not in st.session_state:
    st.session_state.selected_industry = 'default'

# Main header
st.markdown('<div class="main-header">🏢 Enterprise Clustering Intelligence Platform</div>', 
           unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; margin-bottom: 2rem;">
    <b>Advanced Segmentation & Strategic Intelligence for Executive Decision-Making</b>
</div>
""", unsafe_allow_html=True)

# Sidebar - Configuration and Data Upload
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Industry selection
    industry = st.selectbox(
        "Industry Context",
        options=list(config.INDUSTRY_CONFIGS.keys()),
        format_func=lambda x: x.title(),
        help="Select industry for contextualized terminology and insights"
    )
    st.session_state.selected_industry = industry
    
    st.markdown("---")
    st.markdown("### 📤 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV)",
        type=["csv"],
        help="Upload your dataset for clustering analysis"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            quality_metrics = st.session_state.data_processor.load_data(df)
            st.session_state.data_loaded = True
            
            st.success("✅ Data loaded successfully")
            
            # Show data quality score
            quality_score = quality_metrics['data_quality_score']
            st.metric(
                "Data Quality Score",
                f"{quality_score:.2f}",
                delta="Good" if quality_score >= config.MIN_DATA_QUALITY_SCORE else "Needs Attention",
                delta_color="normal" if quality_score >= config.MIN_DATA_QUALITY_SCORE else "inverse"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Records", f"{quality_metrics['total_records']:,}")
            with col2:
                st.metric("Features", quality_metrics['total_features'])
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.session_state.data_loaded = False
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        st.markdown("**Preprocessing Options**")
        imputation = st.selectbox(
            "Missing Value Strategy",
            ['advanced', 'simple', 'statistical'],
            help="Advanced uses KNN imputation for better accuracy"
        )
        
        scaling = st.selectbox(
            "Scaling Method",
            ['robust', 'standard'],
            help="Robust scaling recommended for data with outliers"
        )
        
        handle_outliers = st.checkbox(
            "Handle Outliers",
            value=True,
            help="Cap outliers using IQR method"
        )

# Main content area
if not st.session_state.data_loaded:
    # Welcome screen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Data Upload</h3>
            <p>Upload your dataset to begin comprehensive clustering analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Smart Analysis</h3>
            <p>Automated preprocessing, multi-model clustering, and statistical validation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💡 Executive Insights</h3>
            <p>Strategic recommendations and actionable business intelligence</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("👆 Upload a CSV file using the sidebar to get started")

else:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Quality",
        "⚙️ Preprocessing",
        "🎯 Clustering Analysis",
        "📊 Statistical Insights",
        "💼 Executive Summary"
    ])
    
    # TAB 1: Data Quality Assessment
    with tab1:
        st.markdown("## Data Quality Assessment")
        
        quality = st.session_state.data_processor.quality_report
        
        # Quality metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Quality Score", f"{quality['data_quality_score']:.2%}")
        with col2:
            st.metric("Completeness", f"{(1-quality['missing_rate']):.1%}")
        with col3:
            st.metric("Uniqueness", f"{(1-quality['duplicate_records']/quality['total_records']):.1%}")
        with col4:
            st.metric("Outliers", f"{quality['outlier_percentage']:.1%}")
        
        # Data overview
        st.markdown("### Dataset Overview")
        df_display = st.session_state.data_processor.original_data
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df_display.head(50), width='stretch')
        
        with col2:
            st.markdown("**Feature Summary**")
            st.write(f"Total Records: {quality['total_records']:,}")
            st.write(f"Total Features: {quality['total_features']}")
            st.write(f"Numeric Features: {quality['numeric_features']}")
            st.write(f"Categorical Features: {quality['categorical_features']}")
            
            if quality['feature_variance_issues']:
                st.warning(f"⚠️ {len(quality['feature_variance_issues'])} low-variance features detected")
    
    # TAB 2: Preprocessing
    with tab2:
        st.markdown("## Data Preprocessing & Feature Selection")
        
        if st.button("🚀 Execute Preprocessing Pipeline", type="primary"):
            with st.spinner("Processing data..."):
                processed_df, numeric_features = st.session_state.data_processor.preprocess(
                    imputation_strategy=imputation,
                    scaling_method=scaling,
                    handle_outliers=handle_outliers
                )
                st.session_state.data_preprocessed = True
                st.session_state.numeric_features = numeric_features
                
                st.success("✅ Preprocessing completed successfully")
        
        if st.session_state.data_preprocessed:
            # Show preprocessing log
            with st.expander("📝 Preprocessing Log", expanded=True):
                for log_entry in st.session_state.data_processor.preprocessing_log:
                    st.info(log_entry)
            
            # Feature selection
            st.markdown("### Feature Selection for Clustering")
            
            all_features = st.session_state.numeric_features
            
            selected_features = st.multiselect(
                "Select features for clustering analysis",
                options=all_features,
                default=all_features[:min(8, len(all_features))],
                help="Choose features that best represent your segmentation objectives"
            )
            
            if not selected_features:
                st.warning("⚠️ Please select at least 2 features for clustering")
            else:
                st.session_state.selected_features = selected_features
                
                # Feature statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Selected Features Statistics**")
                    stats_df = pd.DataFrame({
                        feat: st.session_state.data_processor.feature_statistics[feat]
                        for feat in selected_features
                    }).T
                    st.dataframe(stats_df[['mean', 'std', 'cv', 'skewness']], 
                               width='stretch')
                
                with col2:
                    # Check for multicollinearity
                    st.markdown("**Multicollinearity Check**")
                    high_corr = st.session_state.data_processor.detect_multicollinearity(
                        selected_features, threshold=0.9
                    )
                    
                    if high_corr:
                        st.warning(f"⚠️ {len(high_corr)} highly correlated feature pairs detected")
                        for feat1, feat2, corr in high_corr:
                            st.write(f"• {feat1} ↔ {feat2}: {corr:.3f}")
                    else:
                        st.success("✅ No multicollinearity issues detected")
    
    # TAB 3: Clustering Analysis
    with tab3:
        st.markdown("## Clustering Model Configuration & Execution")
        
        if not st.session_state.data_preprocessed:
            st.warning("⚠️ Please complete preprocessing first")
        elif not st.session_state.get('selected_features'):
            st.warning("⚠️ Please select features for clustering")
        else:
            # Model selection
            st.markdown("### Model Selection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**K-Means Clustering**")
                use_kmeans = st.checkbox("Enable K-Means", value=True)
                
                if use_kmeans:
                    auto_k = st.checkbox("Auto-detect optimal K", value=True)
                    
                    if auto_k:
                        k_range = st.slider("K Search Range", 2, 12, (2, 8))
                    else:
                        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            
            with col2:
                st.markdown("**Hierarchical Clustering**")
                use_hierarchical = st.checkbox("Enable Hierarchical", value=True)
                
                if use_hierarchical:
                    n_clusters_hier = st.slider("Hierarchical Clusters", 2, 10, 3, key="hier_k")
                    linkage_method = st.selectbox("Linkage", ['ward', 'complete', 'average'])
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("**DBSCAN (Density-Based)**")
                use_dbscan = st.checkbox("Enable DBSCAN")
                
                if use_dbscan:
                    eps = st.slider("Epsilon (ε)", 0.1, 3.0, 0.5, 0.1)
                    min_samples = st.slider("Min Samples", 3, 20, 5)
            
            with col4:
                st.markdown("**Gaussian Mixture Model**")
                use_gmm = st.checkbox("Enable GMM")
                
                if use_gmm:
                    n_components = st.slider("Components", 2, 10, 3, key="gmm_k")
            
            # Run clustering
            if st.button("🎯 Execute Clustering Analysis", type="primary"):
                if not any([use_kmeans, use_hierarchical, use_dbscan, use_gmm]):
                    st.error("❌ Please select at least one clustering algorithm")
                else:
                    # Get scaled features
                    X_scaled = st.session_state.data_processor.scale_features(
                        st.session_state.selected_features,
                        method=scaling
                    )
                    
                    results = {}
                    
                    with st.spinner("Running clustering algorithms..."):
                        # K-Means
                        if use_kmeans:
                            if auto_k:
                                st.info("🔍 Finding optimal K...")
                                k_analysis = st.session_state.clustering_engine.find_optimal_k(
                                    X_scaled, range(k_range[0], k_range[1]+1)
                                )
                                optimal_k = k_analysis['elbow_k']
                                st.success(f"✅ Optimal K detected: {optimal_k}")
                                
                                # Store for visualization
                                st.session_state.k_analysis = k_analysis
                                
                                result = st.session_state.clustering_engine.fit_kmeans(
                                    X_scaled, optimal_k
                                )
                            else:
                                result = st.session_state.clustering_engine.fit_kmeans(
                                    X_scaled, n_clusters
                                )
                            
                            results['K-Means'] = result
                            st.success(f"✅ K-Means completed with {result['n_clusters']} clusters")
                        
                        # Hierarchical
                        if use_hierarchical:
                            result = st.session_state.clustering_engine.fit_hierarchical(
                                X_scaled, n_clusters_hier, linkage_method
                            )
                            results['Hierarchical'] = result
                            st.success(f"✅ Hierarchical completed with {result['n_clusters']} clusters")
                        
                        # DBSCAN
                        if use_dbscan:
                            result = st.session_state.clustering_engine.fit_dbscan(
                                X_scaled, eps, min_samples
                            )
                            results['DBSCAN'] = result
                            n_noise = result.get('n_noise', 0)
                            st.success(f"✅ DBSCAN completed: {result['n_clusters']} clusters, {n_noise} noise points")
                        
                        # GMM
                        if use_gmm:
                            result = st.session_state.clustering_engine.fit_gaussian_mixture(
                                X_scaled, n_components
                            )
                            results['GMM'] = result
                            st.success(f"✅ GMM completed with {result['n_clusters']} components")
                    
                    st.session_state.clustering_results = results
                    st.session_state.X_scaled = X_scaled
                    st.session_state.clustering_complete = True
                    
                    st.success("🎉 Clustering analysis completed successfully!")
            
            # Show results if available
            if st.session_state.clustering_complete:
                st.markdown("### Model Performance Comparison")
                
                # Performance metrics table
                comparison_data = []
                
                for model_name, result in st.session_state.clustering_results.items():
                    metrics = result['metrics']
                    
                    comparison_data.append({
                        'Algorithm': model_name,
                        'Clusters': result['n_clusters'],
                        'Silhouette': f"{metrics.get('silhouette_score', 0):.3f}",
                        'R² (Var. Explained)': f"{metrics.get('r_squared', 0)*100:.1f}%",
                        'Separation Index': f"{metrics.get('separation_index', 0):.2f}",
                        'Balance Score': f"{metrics.get('balance_score', 0):.3f}",
                        'Quality Rating': '⭐⭐⭐⭐⭐' if metrics.get('silhouette_score', 0) > 0.7
                                        else '⭐⭐⭐⭐' if metrics.get('silhouette_score', 0) > 0.5
                                        else '⭐⭐⭐' if metrics.get('silhouette_score', 0) > 0.25
                                        else '⭐⭐'
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, width='stretch', hide_index=True)
                
                # Show optimal K analysis if available
                if hasattr(st.session_state, 'k_analysis'):
                    with st.expander("📈 Optimal K Analysis"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            k_analysis = st.session_state.k_analysis
                            elbow_fig = st.session_state.viz_engine.create_elbow_plot(
                                k_analysis['k_range'],
                                k_analysis['wcss_scores'],
                                k_analysis['elbow_k']
                            )
                            st.plotly_chart(elbow_fig, width='stretch')
                        
                        with col2:
                            silhouette_fig = st.session_state.viz_engine.create_silhouette_plot(
                                k_analysis['k_range'],
                                k_analysis['silhouette_scores'],
                                k_analysis['silhouette_k']
                            )
                            st.plotly_chart(silhouette_fig, width='stretch')
    
    # TAB 4: Statistical Insights
    with tab4:
        st.markdown("## Statistical Analysis & Segment Characterization")
        
        if not st.session_state.clustering_complete:
            st.warning("⚠️ Please complete clustering analysis first")
        else:
            # Model selection for detailed analysis
            model_name = st.selectbox(
                "Select model for detailed analysis",
                options=list(st.session_state.clustering_results.keys())
            )
            
            result = st.session_state.clustering_results[model_name]
            X_scaled = st.session_state.X_scaled
            labels = result['labels']
            
            # Perform statistical analysis
            with st.spinner("Performing statistical analysis..."):
                stat_analysis = st.session_state.statistical_analyzer.analyze_segments(
                    X_scaled,
                    labels,
                    st.session_state.selected_features,
                    st.session_state.data_processor.processed_data
                )
            
            # Store for executive summary
            st.session_state.statistical_analysis = stat_analysis
            st.session_state.selected_model = model_name
            st.session_state.selected_model_result = result
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Segment distribution
                segment_term = config.INDUSTRY_CONFIGS[st.session_state.selected_industry]['segment_terminology']
                dist_fig = st.session_state.viz_engine.create_segment_overview(labels, segment_term)
                st.plotly_chart(dist_fig, width='stretch')
            
            with col2:
                # PCA visualization
                pca_fig = st.session_state.viz_engine.create_pca_visualization(
                    X_scaled, labels, segment_term
                )
                st.plotly_chart(pca_fig, width='stretch')
            
            # Performance dashboard
            dashboard_fig = st.session_state.viz_engine.create_performance_dashboard(
                result['metrics'], model_name
            )
            st.plotly_chart(dashboard_fig, width='stretch')
            
            # Feature importance
            st.markdown("### Feature Importance Analysis")
            feat_imp_fig = st.session_state.viz_engine.create_feature_importance_chart(
                stat_analysis['feature_importance']
            )
            if feat_imp_fig:
                st.plotly_chart(feat_imp_fig, width='stretch')
            
            # Segment comparison matrix
            comparison_fig = st.session_state.viz_engine.create_segment_comparison_matrix(
                stat_analysis['segment_profiles'], top_n_features=5
            )
            if comparison_fig:
                st.plotly_chart(comparison_fig, width='stretch')
            
            # Detailed segment profiles
            st.markdown("### Detailed Segment Profiles")
            
            for segment_name, profile in stat_analysis['segment_profiles'].items():
                with st.expander(f"📊 {segment_name} - Detailed Analysis"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Size", f"{profile['size']:,} records")
                    with col2:
                        st.metric("Homogeneity", f"{profile['homogeneity']:.3f}")
                    with col3:
                        st.metric("Density", f"{profile['density']:.3f}")
                    
                    st.markdown("**Top Distinguishing Features:**")
                    for feat in profile['distinguishing_features'][:5]:
                        st.write(f"• **{feat['feature']}**: {feat['direction']} "
                               f"(Effect Size: {feat['effect_size']:.3f}, {feat['impact_level']})")
    
    # TAB 5: Executive Summary
    with tab5:
        st.markdown("## Executive Summary & Strategic Recommendations")
        
        if not st.session_state.get('statistical_analysis'):
            st.warning("⚠️ Please complete statistical analysis first")
        else:
            # Generate insights
            insight_gen = InsightGenerator(industry=st.session_state.selected_industry)
            
            with st.spinner("Generating executive insights..."):
                executive_summary = insight_gen.generate_executive_summary(
                    st.session_state.selected_model_result,
                    st.session_state.statistical_analysis,
                    st.session_state.selected_features
                )
            
            # Overview
            st.markdown("### Executive Overview")
            
            overview = executive_summary['overview']
            
            st.markdown(f"""
            <div class="executive-summary">
                <h4>{st.session_state.selected_model} - Strategic Segmentation Analysis</h4>
                <p style="font-size: 1.1rem; line-height: 1.8;">{overview['executive_summary']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Segmentation Quality",
                    f"{overview['segmentation_quality']:.2f}",
                    delta="Production Ready" if overview['business_readiness'] else "Refinement Needed"
                )
            
            with col2:
                st.metric(
                    "Variance Explained",
                    f"{overview['variance_explained_pct']:.1f}%"
                )
            
            with col3:
                perf_metrics = executive_summary['performance_metrics']
                st.metric(
                    "Stability Index",
                    f"{perf_metrics['stability_index']:.3f}"
                )
            
            with col4:
                st.metric(
                    "Actionable Segments",
                    perf_metrics['actionable_segments_count']
                )
            
            st.info(f"**Confidence Assessment:** {overview['confidence_level']}")
            
            # Strategic Insights
            st.markdown("### Strategic Insights")
            
            for i, insight in enumerate(executive_summary['strategic_insights'], 1):
                st.markdown(f"""
                <div class="insight-box">
                    <b>Insight #{i}:</b> {insight}
                </div>
                """, unsafe_allow_html=True)
            
            # Segment Intelligence
            st.markdown("### Segment Intelligence")
            
            for seg_intel in executive_summary['segment_intelligence']:
                with st.expander(f"🎯 {seg_intel['segment']} - {seg_intel['market_share_pct']:.1f}% Market Share"):
                    st.markdown(f"**Characterization:** {seg_intel['characterization']}")
                    
                    st.markdown("**Key Differentiators:**")
                    for diff in seg_intel['key_differentiators']:
                        st.write(f"• {diff['feature']}: {diff['direction']} "
                               f"({diff['impact_level']})")
                    
                    st.markdown("**Business Implications:**")
                    for imp in seg_intel['business_implications']:
                        st.write(f"• {imp}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Homogeneity Index", f"{seg_intel['homogeneity_index']:.3f}")
                    with col2:
                        priority_stars = '⭐' * seg_intel['intervention_priority']
                        st.metric("Intervention Priority", priority_stars)
            
            # Actionable Recommendations
            st.markdown("### Actionable Recommendations")
            
            for i, rec in enumerate(executive_summary['actionable_recommendations'], 1):
                st.markdown(f"""
                <div style="background-color: #fff3cd; border-left: 5px solid #ffc107; 
                            padding: 1rem; margin: 1rem 0; border-radius: 5px;">
                    <h4 style="color: #856404;">Recommendation #{i} [{rec['priority']} Priority]</h4>
                    <p><b>Target:</b> {rec['target']}</p>
                    <p><b>Action:</b> {rec['action']}</p>
                    <p><b>Expected Impact:</b> {rec['expected_impact']:.1%}</p>
                    <p><b>Implementation Complexity:</b> {rec['implementation_complexity']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risks and Opportunities
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ⚠️ Risk Assessment")
                
                risks = executive_summary['risk_opportunities']['risks']
                if risks:
                    for risk in risks:
                        st.markdown(f"""
                        <div style="background-color: #f8d7da; border-left: 5px solid #dc3545; 
                                    padding: 1rem; margin: 0.5rem 0; border-radius: 5px;">
                            <b>{risk['type']}</b> [{risk['severity']} Severity]<br>
                            {risk['description']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No significant risks identified")
            
            with col2:
                st.markdown("### 💡 Opportunities")
                
                opportunities = executive_summary['risk_opportunities']['opportunities']
                if opportunities:
                    for opp in opportunities:
                        st.markdown(f"""
                        <div style="background-color: #d4edda; border-left: 5px solid #28a745; 
                                    padding: 1rem; margin: 0.5rem 0; border-radius: 5px;">
                            <b>{opp['type']}</b> [{opp['potential']} Potential]<br>
                            {opp['description']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No immediate opportunities identified")
            
            # Download results
            st.markdown("### 📥 Export Results")
            
            # Prepare results dataframe
            results_df = st.session_state.data_processor.processed_data.copy()
            results_df['Cluster_Assignment'] = st.session_state.selected_model_result['labels']
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                st.download_button(
                    label="📊 Download Clustering Results (CSV)",
                    data=csv,
                    file_name=f"clustering_results_{st.session_state.selected_model.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    width='stretch'
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <b>Enterprise Clustering Intelligence Platform v2.0</b><br>
    Powered by Advanced Machine Learning & Statistical Analysis
</div>
""", unsafe_allow_html=True)
