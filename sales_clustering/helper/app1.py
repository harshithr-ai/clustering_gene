"""
Clustering POC Application - Bug-Free Version
Upload data → Automatic EDA → Preprocessing → Multi-model Clustering → Visualizations → Statistical Insights
Kimi
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="ML Clustering POC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; font-weight: bold; }
    .sub-header { font-size: 1.5rem; color: #2ca02c; margin-top: 20px; }
    .insight-box { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4; }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🤖 Automated Clustering POC</p>', unsafe_allow_html=True)
st.markdown("Upload your data and let the system handle preprocessing, EDA, clustering, and insights automatically!")

# Session state to store data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = {}
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []

# Sidebar - Data Upload
with st.sidebar:
    st.markdown("### 📤 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("✅ Data uploaded successfully!")
            st.info(f"📊 Shape: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

# Main app logic
if st.session_state.data is not None:
    data = st.session_state.data
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Overview", 
        "🔍 EDA & Visualization", 
        "⚙️ Preprocessing", 
        "🎯 Clustering", 
        "📈 Results & Insights"
    ])
    
    # ==================== TAB 1: DATA OVERVIEW ====================
    with tab1:
        st.markdown('<p class="sub-header">Raw Data Overview</p>', unsafe_allow_html=True)
        
        # Show first few rows
        with st.expander("👀 View Raw Data (first 100 rows)", expanded=True):
            st.dataframe(data.head(100), use_container_width=True)
        
        # Data structure info - FIXED BUG HERE
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Data Structure:**")
            buffer = io.StringIO()
            data.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)
        
        with col2:
            st.markdown("**🔍 Missing Values:**")
            missing = data.isnull().sum()
            missing_pct = (missing / len(data)) * 100
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Missing %': missing_pct.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            if len(missing_df) > 0:
                st.dataframe(missing_df.style.format({'Missing %': '{:.2f}%'}), use_container_width=True)
            else:
                st.success("✅ No missing values!")
        
        # Column types summary
        st.markdown("**📝 Column Summary:**")
        col_types = pd.DataFrame({
            'Column': data.columns,
            'Data Type': [str(dtype) for dtype in data.dtypes.values],
            'Unique Values': [data[col].nunique() for col in data.columns],
            'Missing (%)': [f"{(data[col].isnull().sum() / len(data) * 100):.2f}%" for col in data.columns]
        })
        st.dataframe(col_types, use_container_width=True)
    
    # ==================== TAB 2: EDA ====================
    with tab2:
        st.markdown('<p class="sub-header">Exploratory Data Analysis</p>', unsafe_allow_html=True)
        
        # Statistical summary
        with st.expander("📊 Statistical Summary", expanded=True):
            st.dataframe(data.describe().T, use_container_width=True)
        
        # Correlation heatmap
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            st.markdown("**🔗 Correlation Matrix:**")
            fig = plt.figure(figsize=(12, 10))
            sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
            plt.title('Correlation Heatmap', fontsize=16, pad=20)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        # Distributions
        st.markdown("**📈 Feature Distributions:**")
        if len(numeric_cols) > 0:
            dist_cols = st.multiselect(
                "Select features to visualize distribution",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            for col in dist_cols:
                fig = px.histogram(
                    data, 
                    x=col, 
                    marginal="box", 
                    title=f"Distribution of {col}",
                    nbins=30,
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Categorical analysis
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_cols) > 0:
            st.markdown("**🎯 Categorical Analysis:**")
            cat_col = st.selectbox("Select categorical column", categorical_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                value_counts = data[cat_col].value_counts()
                fig = px.bar(
                    x=value_counts.index, 
                    y=value_counts.values,
                    title=f"Value Counts - {cat_col}",
                    labels={'x': cat_col, 'y': 'Count'},
                    color=value_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(value_counts) <= 10:
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Pie Chart - {cat_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 3: PREPROCESSING ====================
    with tab3:
        st.markdown('<p class="sub-header">Data Preprocessing</p>', unsafe_allow_html=True)
        
        if st.session_state.preprocessed_data is None:
            preprocessed_data = data.copy()
            preprocessing_log = []
            
            # Handle missing values
            for col in preprocessed_data.columns:
                missing_count = preprocessed_data[col].isnull().sum()
                if missing_count > 0:
                    if pd.api.types.is_numeric_dtype(preprocessed_data[col]):
                        preprocessed_data[col].fillna(preprocessed_data[col].median(), inplace=True)
                        preprocessing_log.append(f"✅ Filled {missing_count} missing values in '{col}' with median")
                    else:
                        preprocessed_data[col].fillna(preprocessed_data[col].mode()[0], inplace=True)
                        preprocessing_log.append(f"✅ Filled {missing_count} missing values in '{col}' with mode")
            
            # Encode categorical variables
            le_dict = {}
            categorical_cols = preprocessed_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                # Handle any remaining NaN values
                preprocessed_data[col] = preprocessed_data[col].astype(str)
                preprocessed_data[col] = le.fit_transform(preprocessed_data[col])
                le_dict[col] = le
                preprocessing_log.append(f"✅ Encoded categorical column '{col}' ({len(le.classes_)} unique values)")
            
            # Store preprocessed data
            st.session_state.preprocessed_data = preprocessed_data
            
            # Display log
            st.markdown("**🔧 Preprocessing Log:**")
            for log in preprocessing_log:
                st.info(log)
            
            st.success("✅ Data preprocessed successfully!")
        
        # Show preprocessed data
        with st.expander("👀 View Preprocessed Data", expanded=False):
            st.dataframe(st.session_state.preprocessed_data.head(), use_container_width=True)
        
        # Feature selection for clustering
        st.markdown("**🎯 Select Features for Clustering:**")
        all_features = st.session_state.preprocessed_data.columns.tolist()
        
        # Filter out non-numeric columns if any remain
        numeric_features = st.session_state.preprocessed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        selected_features = st.multiselect(
            "Choose features for clustering (leave empty to use all numeric features)",
            options=all_features,
            default=numeric_features[:min(5, len(numeric_features))]
        )
        
        if not selected_features:
            selected_features = numeric_features
        
        st.session_state.selected_features = selected_features
        st.info(f"📊 Using {len(selected_features)} features for clustering: {', '.join(selected_features)}")
        
        # Scale features
        if len(selected_features) > 0:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(st.session_state.preprocessed_data[selected_features])
            
            st.markdown("**📐 Feature Scaling:**")
            st.success("✅ Applied StandardScaler to normalize features")
            
            # Show scaled data stats
            scaled_df = pd.DataFrame(features_scaled, columns=selected_features)
            with st.expander("📊 Scaled Features Statistics"):
                st.dataframe(scaled_df.describe().T, use_container_width=True)
        else:
            st.error("❌ No numeric features available for clustering!")
    
    # ==================== TAB 4: CLUSTERING ====================
    with tab4:
        st.markdown('<p class="sub-header">Clustering Models</p>', unsafe_allow_html=True)
        
        if st.session_state.preprocessed_data is not None and len(st.session_state.selected_features) > 0:
            
            # Model selection
            st.markdown("**🤖 Select Clustering Models to Run:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                use_kmeans = st.checkbox("K-Means Clustering", value=True)
                if use_kmeans:
                    n_clusters_kmeans = st.slider("K-Means: Number of clusters (k)", 2, 10, 3, key="kmeans_slider")
            
            with col2:
                use_dbscan = st.checkbox("DBSCAN Clustering", value=False)
                if use_dbscan:
                    eps_dbscan = st.slider("DBSCAN: Epsilon (ε)", 0.1, 5.0, 0.5, key="dbscan_eps")
                    min_samples_dbscan = st.slider("DBSCAN: Min Samples", 2, 20, 5, key="dbscan_min")
            
            use_hierarchical = st.checkbox("Hierarchical Clustering (Agglomerative)", value=True)
            if use_hierarchical:
                n_clusters_hier = st.slider("Hierarchical: Number of clusters", 2, 10, 3, key="hier_slider")
                linkage = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
            
            # Run clustering button
            if st.button("🚀 Run Clustering Analysis", type="primary"):
                if not any([use_kmeans, use_dbscan, use_hierarchical]):
                    st.error("❌ Please select at least one clustering model!")
                else:
                    with st.spinner("Running clustering models... Please wait."):
                        features_scaled = StandardScaler().fit_transform(
                            st.session_state.preprocessed_data[st.session_state.selected_features]
                        )
                        
                        # Clear previous results
                        st.session_state.clustering_results = {}
                        
                        # K-Means
                        if use_kmeans:
                            try:
                                kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
                                kmeans_labels = kmeans.fit_predict(features_scaled)
                                st.session_state.clustering_results['K-Means'] = {
                                    'labels': kmeans_labels,
                                    'model': kmeans,
                                    'n_clusters': n_clusters_kmeans
                                }
                                unique_clusters = len(set(kmeans_labels))
                                st.success(f"✅ K-Means completed with {unique_clusters} clusters")
                            except Exception as e:
                                st.error(f"❌ K-Means error: {str(e)}")
                        
                        # DBSCAN
                        if use_dbscan:
                            try:
                                dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
                                dbscan_labels = dbscan.fit_predict(features_scaled)
                                n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                                st.session_state.clustering_results['DBSCAN'] = {
                                    'labels': dbscan_labels,
                                    'model': dbscan,
                                    'n_clusters': n_clusters_dbscan
                                }
                                noise_points = np.sum(dbscan_labels == -1)
                                st.success(f"✅ DBSCAN completed with {n_clusters_dbscan} clusters ({noise_points} noise points)")
                            except Exception as e:
                                st.error(f"❌ DBSCAN error: {str(e)}")
                        
                        # Hierarchical
                        if use_hierarchical:
                            try:
                                hierarchical = AgglomerativeClustering(n_clusters=n_clusters_hier, linkage=linkage)
                                hierarchical_labels = hierarchical.fit_predict(features_scaled)
                                st.session_state.clustering_results['Hierarchical'] = {
                                    'labels': hierarchical_labels,
                                    'model': hierarchical,
                                    'n_clusters': n_clusters_hier
                                }
                                st.success(f"✅ Hierarchical clustering completed with {n_clusters_hier} clusters")
                            except Exception as e:
                                st.error(f"❌ Hierarchical clustering error: {str(e)}")
        
        else:
            st.warning("⚠️ Please complete preprocessing and select features first!")
    
    # ==================== TAB 5: RESULTS & INSIGHTS ====================
    with tab5:
        st.markdown('<p class="sub-header">Clustering Results & Statistical Insights</p>', unsafe_allow_html=True)
        
        if st.session_state.clustering_results and len(st.session_state.selected_features) > 0:
            results = st.session_state.clustering_results
            preprocessed_data = st.session_state.preprocessed_data
            selected_features = st.session_state.selected_features
            
            # Prepare data for visualization
            features_scaled = StandardScaler().fit_transform(preprocessed_data[selected_features])
            
            # PCA for 2D visualization
            if len(selected_features) >= 2:
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(features_scaled)
            else:
                # If only 1 feature, create dummy second dimension
                features_2d = np.column_stack([features_scaled.flatten(), np.zeros_like(features_scaled.flatten())])
            
            # Create results DataFrame
            results_df = preprocessed_data.copy()
            results_df['PCA1'] = features_2d[:, 0]
            results_df['PCA2'] = features_2d[:, 1]
            
            # Model comparison metrics
            st.markdown("**📊 Model Performance Comparison:**")
            
            metrics_data = []
            for model_name, result in results.items():
                labels = result['labels']
                n_clusters = result['n_clusters']
                
                # Skip models that failed or produced single cluster
                if n_clusters <= 1 or len(set(labels)) <= 1:
                    metrics_data.append({
                        'Model': model_name,
                        'Clusters': n_clusters,
                        'Silhouette Score': 'N/A',
                        'Calinski-Harabasz': 'N/A',
                        'Davies-Bouldin': 'N/A',
                        'Status': '⚠️ Single cluster or failed'
                    })
                    continue
                
                try:
                    silhouette = silhouette_score(features_scaled, labels)
                    calinski = calinski_harabasz_score(features_scaled, labels)
                    davies = davies_bouldin_score(features_scaled, labels)
                    
                    metrics_data.append({
                        'Model': model_name,
                        'Clusters': n_clusters,
                        'Silhouette Score': f"{silhouette:.3f}",
                        'Calinski-Harabasz': f"{calinski:.0f}",
                        'Davies-Bouldin': f"{davies:.3f}",
                        'Status': '✅ Success'
                    })
                except Exception as e:
                    metrics_data.append({
                        'Model': model_name,
                        'Clusters': n_clusters,
                        'Silhouette Score': 'Error',
                        'Calinski-Harabasz': 'Error',
                        'Davies-Bouldin': 'Error',
                        'Status': f'❌ {str(e)[:30]}'
                    })
                
                # Add cluster labels to results dataframe
                results_df[f'{model_name}_Cluster'] = labels
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Visualizations for each model
            for model_name, result in results.items():
                if result['n_clusters'] <= 1:
                    continue
                
                st.markdown(f"**🎯 {model_name} Clustering Results:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 2D PCA Plot
                    fig = px.scatter(
                        results_df, 
                        x='PCA1', 
                        y='PCA2', 
                        color=f'{model_name}_Cluster',
                        title=f'{model_name} - PCA Visualization (2D)',
                        hover_data=selected_features[:3] if len(selected_features) >= 3 else selected_features,
                        color_continuous_scale='Viridis',
                        width=600,
                        height=500
                    )
                    fig.update_layout(
                        legend_title_text='Cluster',
                        title_x=0.5
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Cluster distribution
                    cluster_counts = pd.Series(result['labels']).value_counts().sort_index()
                    fig = px.bar(
                        x=cluster_counts.index, 
                        y=cluster_counts.values,
                        title=f'{model_name} - Cluster Distribution',
                        labels={'x': 'Cluster', 'y': 'Count'},
                        color=cluster_counts.values,
                        color_continuous_scale='Blues',
                        width=600,
                        height=500
                    )
                    fig.update_layout(
                        title_x=0.5,
                        xaxis=dict(tickmode='linear')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistical insights
                st.markdown(f"**📈 {model_name} - Statistical Insights:**")
                
                cluster_col = f'{model_name}_Cluster'
                insights = []
                
                # Calculate overall statistics
                overall_means = results_df[selected_features].mean()
                
                for cluster_id in sorted(results_df[cluster_col].unique()):
                    cluster_data = results_df[results_df[cluster_col] == cluster_id]
                    cluster_size = len(cluster_data)
                    size_pct = (cluster_size / len(results_df)) * 100
                    
                    if cluster_id == -1:
                        insights.append(f"**🔵 Noise Points (DBSCAN)** ({cluster_size} records, {size_pct:.1f}%)")
                    else:
                        insights.append(f"**🟢 Cluster {cluster_id}** ({cluster_size} records, {size_pct:.1f}%)")
                    
                    # Feature analysis for this cluster
                    cluster_means = cluster_data[selected_features].mean()
                    
                    insights.append("- Key characteristics:")
                    # Show top 3 features with highest/lowest deviation
                    deviations = []
                    for feature in selected_features:
                        cluster_mean = cluster_means[feature]
                        overall_mean = overall_means[feature]
                        if overall_mean != 0:
                            deviation = ((cluster_mean - overall_mean) / overall_mean) * 100
                            deviations.append((feature, deviation, cluster_mean))
                    
                    # Sort by absolute deviation and show top 3
                    deviations.sort(key=lambda x: abs(x[1]), reverse=True)
                    for feature, deviation, cluster_mean in deviations[:3]:
                        direction = "↑" if deviation > 0 else "↓"
                        color = "green" if deviation > 0 else "red"
                        insights.append(f"  • **{feature}**: {cluster_mean:.2f} <span style='color:{color}'>{direction} {abs(deviation):.1f}%</span>")
                    
                    insights.append("")
                
                # Display insights in a formatted box
                insight_text = "<br>".join(insights)
                st.markdown(f'<div class="insight-box">{insight_text}</div>', unsafe_allow_html=True)
                
                # Detailed cluster statistics
                with st.expander(f"📊 View detailed statistics for {model_name}"):
                    cluster_stats = results_df.groupby(cluster_col)[selected_features].agg([
                        'count', 'mean', 'median', 'std', 'min', 'max'
                    ]).round(2)
                    st.dataframe(cluster_stats, use_container_width=True)
            
            # Download results
            st.markdown("**💾 Download Clustering Results:**")
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name="clustering_results.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
        
        else:
            st.warning("⚠️ No clustering results available. Please run clustering models in the 'Clustering' tab.")

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center;">Built with 🤖 | Streamlit, Scikit-Learn & Plotly</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    st.info("👋 Welcome! Upload a CSV file to get started with automated clustering analysis.")