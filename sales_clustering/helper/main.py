"""
Enhanced Clustering POC Application
Upload data → Advanced EDA → Preprocessing → Multi-model Clustering → Visualizations → Statistical Insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="ML Clustering POC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Enhanced with dark mode support)
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; font-weight: bold; }
    .sub-header { font-size: 1.5rem; color: #2ca02c; margin-top: 20px; }
    .insight-box { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white; }
    .warning-box { background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107; }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🤖 Enhanced Clustering POC</p>', unsafe_allow_html=True)
st.markdown("Upload your data and let the system handle advanced EDA, preprocessing, clustering, and insights!")

# Session state to store data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = {}
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'eda_report' not in st.session_state:
    st.session_state.eda_report = {}
if 'original_data' not in st.session_state:
    st.session_state.original_data = None  # Keep original for comparison

# Sidebar - Data Upload & Advanced Options
with st.sidebar:
    st.markdown("### 📤 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.session_state.original_data = st.session_state.data.copy()  # Keep original
            st.success("✅ Data uploaded successfully!")
            st.info(f"📊 Shape: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
            
            # Data Quality Quick Check
            duplicates = st.session_state.data.duplicated().sum()
            if duplicates > 0:
                st.warning(f"⚠️ Found {duplicates} duplicate rows")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
    
    # Advanced Options
    st.markdown("---")
    st.markdown("### ⚙️ Advanced Options")
    
    # Scaling method
    scaling_method = st.selectbox(
        "Feature Scaling Method",
        ["Standard Scaler (Z-score)", "Robust Scaler", "Min-Max Scaler"]
    )
    
    # Export options
    st.markdown("### 💾 Export Settings")
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
    
    # Model persistence
    if st.session_state.clustering_results:
        if st.button("💾 Save Models to Session"):
            st.session_state.saved_models = st.session_state.clustering_results
            st.success("✅ Models saved to session!")

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
        
        # Data structure info
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
                # Missing values heatmap
                st.markdown("**📊 Missing Values Pattern:**")
                fig = plt.figure(figsize=(10, 4))
                sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap='viridis')
                plt.title('Missing Values Pattern (Yellow = Missing)')
                st.pyplot(fig, use_container_width=True)
            else:
                st.success("✅ No missing values!")
        
        # Enhanced Column Summary with warnings
        st.markdown("**📝 Column Summary & Quality Report:**")
        col_types = pd.DataFrame({
            'Column': data.columns,
            'Data Type': [str(dtype) for dtype in data.dtypes.values],
            'Unique Values': [data[col].nunique() for col in data.columns],
            'Missing (%)': [f"{(data[col].isnull().sum() / len(data) * 100):.2f}%" for col in data.columns],
            'Constant': [data[col].nunique() == 1 for col in data.columns],
            'Highly Cardinal (>50%)': [data[col].nunique() / len(data) > 0.5 for col in data.columns]
        })
        
        # Highlight issues
        col_types_styled = col_types.style.apply(lambda x: ['background-color: #ffcccc' if val else '' for val in x['Constant']], subset=['Constant'])
        st.dataframe(col_types, use_container_width=True)
        
        # Data quality warnings
        constant_cols = col_types[col_types['Constant']]['Column'].tolist()
        high_cardinal_cols = col_types[col_types['Highly Cardinal (>50%)']]['Column'].tolist()
        
        if constant_cols:
            st.markdown('<div class="warning-box">⚠️ **Constant columns detected:** These have no variance and should be removed: ' + ', '.join(constant_cols) + '</div>', unsafe_allow_html=True)
        
        if high_cardinal_cols:
            st.markdown('<div class="warning-box">⚠️ **High cardinality columns:** These may be IDs or unique identifiers: ' + ', '.join(high_cardinal_cols) + '</div>', unsafe_allow_html=True)
        
        # Duplicates check
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            st.markdown(f'<div class="warning-box">⚠️ **Duplicate rows found:** {duplicates} rows are exact duplicates</div>', unsafe_allow_html=True)
            if st.button("Show duplicate rows"):
                st.dataframe(data[data.duplicated(keep=False)], use_container_width=True)
    
    # ==================== TAB 2: ENHANCED EDA ====================
    with tab2:
        st.markdown('<p class="sub-header">Exploratory Data Analysis (Advanced)</p>', unsafe_allow_html=True)
        
        # Statistical summary
        with st.expander("📊 Statistical Summary", expanded=True):
            st.dataframe(data.describe().T, use_container_width=True)
        
        # NEW: Skewness & Kurtosis Analysis
        st.markdown("**📐 Skewness & Kurtosis Analysis:**")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            skewness = data[numeric_cols].skew().sort_values(ascending=False)
            kurtosis = data[numeric_cols].kurtosis().sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Skewness (Symmetry):**")
                skew_df = pd.DataFrame({
                    'Feature': skewness.index,
                    'Skewness': skewness.values,
                    'Interpretation': ['Right-skewed' if x > 1 else 'Left-skewed' if x < -1 else 'Approximately symmetric' for x in skewness.values]
                })
                st.dataframe(skew_df, use_container_width=True)
            
            with col2:
                st.markdown("**Kurtosis (Tailedness):**")
                kurt_df = pd.DataFrame({
                    'Feature': kurtosis.index,
                    'Kurtosis': kurtosis.values,
                    'Interpretation': ['Heavy-tailed' if x > 3 else 'Light-tailed' if x < 3 else 'Normal-tailed' for x in kurtosis.values]
                })
                st.dataframe(kurt_df, use_container_width=True)
            
            # Auto-suggest transformations
            highly_skewed = skew_df[abs(skew_df['Skewness']) > 1]['Feature'].tolist()
            if highly_skewed:
                st.info(f"💡 Consider log/sqrt transformation for highly skewed features: {', '.join(highly_skewed)}")
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            st.markdown("**🔗 Correlation Matrix:**")
            
            # Top correlations
            corr_matrix = data[numeric_cols].corr()
            top_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    top_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            top_corr_df = pd.DataFrame(top_corr)
            top_corr_df['Abs Correlation'] = abs(top_corr_df['Correlation'])
            top_corr_df = top_corr_df.sort_values('Abs Correlation', ascending=False).head(10)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = plt.figure(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
                plt.title('Correlation Heatmap', fontsize=16, pad=20)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            
            with col2:
                st.markdown("**🔝 Top Correlations:**")
                st.dataframe(top_corr_df[['Feature 1', 'Feature 2', 'Correlation']].style.format({'Correlation': '{:.3f}'}), use_container_width=True)
        
        # NEW: Outlier Detection
        st.markdown("**🎯 Outlier Detection:**")
        outlier_method = st.selectbox("Select outlier detection method", ["Box Plot (IQR)", "Z-Score", "Isolation Forest"])
        
        if outlier_method == "Box Plot (IQR)":
            outlier_feature = st.selectbox("Select feature for outlier visualization", numeric_cols)
            fig = px.box(data, y=outlier_feature, title=f"Box Plot - {outlier_feature}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate outliers
            Q1 = data[outlier_feature].quantile(0.25)
            Q3 = data[outlier_feature].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data[outlier_feature] < Q1 - 1.5*IQR) | (data[outlier_feature] > Q3 + 1.5*IQR)]
            st.info(f"🔍 Found {len(outliers)} outliers in {outlier_feature} (using IQR method)")
            
        elif outlier_method == "Z-Score":
            z_threshold = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0)
            outliers_dict = {}
            for col in numeric_cols[:5]:  # Check first 5 numeric columns
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = data[z_scores > z_threshold]
                outliers_dict[col] = len(outliers)
            
            outlier_df = pd.DataFrame(list(outliers_dict.items()), columns=['Feature', 'Outlier Count'])
            st.dataframe(outlier_df, use_container_width=True)
        
        # Distributions
        st.markdown("**📈 Feature Distributions:**")
        if len(numeric_cols) > 0:
            dist_cols = st.multiselect(
                "Select features to visualize distribution",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            # NEW: Distribution type selector
            dist_type = st.radio("Distribution plot type", ["Histogram", "Violin Plot", "ECDF"], horizontal=True)
            
            for col in dist_cols:
                if dist_type == "Histogram":
                    fig = px.histogram(
                        data, 
                        x=col, 
                        marginal="box", 
                        title=f"Distribution of {col}",
                        nbins=30,
                        color_discrete_sequence=['#1f77b4']
                    )
                elif dist_type == "Violin Plot":
                    fig = px.violin(data, y=col, title=f"Violin Plot - {col}", box=True, points="outliers")
                else:  # ECDF
                    fig = px.ecdf(data, x=col, title=f"ECDF - {col}")
                
                st.plotly_chart(fig, use_container_width=True)
        
        # NEW: Pairwise Scatter Plot Matrix
        if len(numeric_cols) >= 2:
            st.markdown("**🔗 Pairwise Relationships:**")
            pairplot_cols = st.multiselect(
                "Select features for pairplot (max 5)", 
                numeric_cols, 
                default=numeric_cols[:min(3, len(numeric_cols))],
                max_selections=5
            )
            
            if len(pairplot_cols) >= 2:
                fig = px.scatter_matrix(
                    data, 
                    dimensions=pairplot_cols,
                    color=pairplot_cols[0],  # Color by first feature
                    title="Scatter Matrix - Pairwise Relationships"
                )
                fig.update_layout(height=800)
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
            
            # NEW: Categorical vs Numeric analysis
            st.markdown("**📊 Categorical vs Numeric Analysis:**")
            if len(numeric_cols) > 0:
                num_for_cat = st.selectbox("Select numeric feature to compare", numeric_cols)
                fig = px.box(
                    data, 
                    x=cat_col, 
                    y=num_for_cat, 
                    title=f"{cat_col} vs {num_for_cat}"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # NEW: Feature Importance based on Variance
        st.markdown("**🏆 Feature Importance (Variance-Based):**")
        if len(numeric_cols) > 0:
            variances = data[numeric_cols].var().sort_values(ascending=False)
            fig = px.bar(
                x=variances.index,
                y=variances.values,
                title="Feature Importance by Variance",
                labels={'x': 'Feature', 'y': 'Variance'},
                color=variances.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Identify low variance features
            low_variance = variances[variances < variances.median() / 2].index.tolist()
            if low_variance:
                st.warning(f"⚠️ Low variance features (consider removal): {', '.join(low_variance)}")
        
        # NEW: Data Quality Score
        st.markdown("**📈 Data Quality Score:**")
        quality_score = 100
        issues = []
        
        if missing_df.sum()['Missing Count'] > 0:
            quality_score -= 10
            issues.append("Missing values")
        
        if duplicates > 0:
            quality_score -= 5
            issues.append(f"{duplicates} duplicates")
        
        if constant_cols:
            quality_score -= 15
            issues.append(f"{len(constant_cols)} constant columns")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Quality Score", f"{quality_score:.0f}/100")
        with col2:
            st.metric("Total Issues", len(issues))
        with col3:
            if issues:
                st.markdown("**Issues:** " + ", ".join(issues))
            else:
                st.success("✅ Clean dataset!")
    
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
                preprocessed_data[col] = preprocessed_data[col].astype(str)
                preprocessed_data[col] = le.fit_transform(preprocessed_data[col])
                le_dict[col] = le
                preprocessing_log.append(f"✅ Encoded categorical column '{col}' ({len(le.classes_)} unique values)")
            
            # NEW: Remove constant columns
            constant_cols = preprocessed_data.columns[preprocessed_data.nunique() == 1].tolist()
            if constant_cols:
                preprocessed_data.drop(columns=constant_cols, inplace=True)
                preprocessing_log.append(f"🗑️ Removed constant columns: {', '.join(constant_cols)}")
            
            # NEW: Remove duplicates
            duplicates_before = len(preprocessed_data)
            preprocessed_data.drop_duplicates(inplace=True)
            duplicates_removed = duplicates_before - len(preprocessed_data)
            if duplicates_removed > 0:
                preprocessing_log.append(f"🗑️ Removed {duplicates_removed} duplicate rows")
            
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
        
        # Scaling based on user selection
        if len(selected_features) > 0:
            if scaling_method == "Robust Scaler":
                scaler = RobustScaler()
            elif scaling_method == "Min-Max Scaler":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
                
            features_scaled = scaler.fit_transform(st.session_state.preprocessed_data[selected_features])
            
            st.markdown(f"**📐 Feature Scaling ({scaling_method}):**")
            st.success(f"✅ Applied {scaling_method} to normalize features")
            
            # Show scaled data stats
            scaled_df = pd.DataFrame(features_scaled, columns=selected_features)
            with st.expander("📊 Scaled Features Statistics"):
                st.dataframe(scaled_df.describe().T, use_container_width=True)
        else:
            st.error("❌ No numeric features available for clustering!")
    
    # ==================== TAB 4: ENHANCED CLUSTERING ====================
    with tab4:
        st.markdown('<p class="sub-header">Clustering Models</p>', unsafe_allow_html=True)
        
        if st.session_state.preprocessed_data is not None and len(st.session_state.selected_features) > 0:
            
            # Model selection
            st.markdown("**🤖 Select Clustering Models to Run:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                use_kmeans = st.checkbox("K-Means Clustering", value=True)
                if use_kmeans:
                    n_clusters_kmeans = st.slider("K-Means: Number of clusters (k)", 2, 15, 3, key="kmeans_slider")
            
            with col2:
                use_dbscan = st.checkbox("DBSCAN Clustering", value=False)
                if use_dbscan:
                    eps_dbscan = st.slider("DBSCAN: Epsilon (ε)", 0.1, 5.0, 0.5, key="dbscan_eps")
                    min_samples_dbscan = st.slider("DBSCAN: Min Samples", 2, 20, 5, key="dbscan_min")
            
            # NEW: OPTICS clustering
            use_optics = st.checkbox("OPTICS Clustering", value=False)
            if use_optics:
                min_samples_optics = st.slider("OPTICS: Min Samples", 2, 20, 5, key="optics_min")
                xi_optics = st.slider("OPTICS: Xi (clustering criterion)", 0.01, 0.5, 0.05, key="optics_xi")
            
            use_hierarchical = st.checkbox("Hierarchical Clustering (Agglomerative)", value=True)
            if use_hierarchical:
                n_clusters_hier = st.slider("Hierarchical: Number of clusters", 2, 15, 3, key="hier_slider")
                linkage = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
            
            # NEW: Advanced options
            with st.expander("🔧 Advanced Clustering Options"):
                random_state = st.number_input("Random State (for reproducibility)", 0, 1000, 42)
                n_init = st.slider("K-Means: Number of initializations", 1, 20, 10)
            
            # Run clustering button
            if st.button("🚀 Run Clustering Analysis", type="primary"):
                if not any([use_kmeans, use_dbscan, use_hierarchical, use_optics]):
                    st.error("❌ Please select at least one clustering model!")
                else:
                    with st.spinner("Running clustering models... Please wait."):
                        features_scaled = scaler.fit_transform(
                            st.session_state.preprocessed_data[st.session_state.selected_features]
                        )
                        
                        # Clear previous results
                        st.session_state.clustering_results = {}
                        
                        # K-Means
                        if use_kmeans:
                            try:
                                kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=random_state, n_init=n_init)
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
                        
                        # OPTICS
                        if use_optics:
                            try:
                                optics = OPTICS(min_samples=min_samples_optics, xi=xi_optics)
                                optics_labels = optics.fit_predict(features_scaled)
                                n_clusters_optics = len(set(optics_labels)) - (1 if -1 in optics_labels else 0)
                                st.session_state.clustering_results['OPTICS'] = {
                                    'labels': optics_labels,
                                    'model': optics,
                                    'n_clusters': n_clusters_optics
                                }
                                noise_points = np.sum(optics_labels == -1)
                                st.success(f"✅ OPTICS completed with {n_clusters_optics} clusters ({noise_points} noise points)")
                            except Exception as e:
                                st.error(f"❌ OPTICS error: {str(e)}")
                        
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
            features_scaled = scaler.fit_transform(preprocessed_data[selected_features])
            
            # PCA for 2D visualization
            if len(selected_features) >= 2:
                pca = PCA(n_components=2, random_state=42)
                features_2d = pca.fit_transform(features_scaled)
                explained_variance = pca.explained_variance_ratio_.sum()
                st.info(f"📊 PCA explained variance: {explained_variance:.2%}")
            else:
                features_2d = np.column_stack([features_scaled.flatten(), np.zeros_like(features_scaled.flatten())])
                explained_variance = 0
            
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
                
                results_df[f'{model_name}_Cluster'] = labels
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # NEW: Best model recommendation
            successful_models = metrics_df[metrics_df['Status'] == '✅ Success']
            if not successful_models.empty:
                best_silhouette = successful_models.loc[successful_models['Silhouette Score'].astype(float).idxmax()]
                st.markdown(f"🏆 **Recommended Model:** {best_silhouette['Model']} (Silhouette: {best_silhouette['Silhouette Score']})")
            
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
                        title=f'{model_name} - PCA Visualization (2D)<br><sub>Explained variance: {explained_variance:.1%}</sub>',
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
                
                overall_means = results_df[selected_features].mean()
                
                for cluster_id in sorted(results_df[cluster_col].unique()):
                    cluster_data = results_df[results_df[cluster_col] == cluster_id]
                    cluster_size = len(cluster_data)
                    size_pct = (cluster_size / len(results_df)) * 100
                    
                    if cluster_id == -1:
                        insights.append(f"**🔵 Noise Points (DBSCAN)** ({cluster_size} records, {size_pct:.1f}%)")
                    else:
                        insights.append(f"**🟢 Cluster {cluster_id}** ({cluster_size} records, {size_pct:.1f}%)")
                    
                    cluster_means = cluster_data[selected_features].mean()
                    insights.append("- Key characteristics:")
                    deviations = []
                    for feature in selected_features:
                        cluster_mean = cluster_means[feature]
                        overall_mean = overall_means[feature]
                        if overall_mean != 0:
                            deviation = ((cluster_mean - overall_mean) / overall_mean) * 100
                            deviations.append((feature, deviation, cluster_mean))
                    
                    deviations.sort(key=lambda x: abs(x[1]), reverse=True)
                    for feature, deviation, cluster_mean in deviations[:3]:
                        direction = "↑" if deviation > 0 else "↓"
                        color = "green" if deviation > 0 else "red"
                        insights.append(f"  • **{feature}**: {cluster_mean:.2f} <span style='color:{color}'>{direction} {abs(deviation):.1f}%</span>")
                    
                    insights.append("")
                
                insight_text = "<br>".join(insights)
                st.markdown(f'<div class="insight-box">{insight_text}</div>', unsafe_allow_html=True)
                
                # Detailed cluster statistics
                with st.expander(f"📊 View detailed statistics for {model_name}"):
                    cluster_stats = results_df.groupby(cluster_col)[selected_features].agg([
                        'count', 'mean', 'median', 'std', 'min', 'max'
                    ]).round(2)
                    st.dataframe(cluster_stats, use_container_width=True)
            
            # NEW: 3D PCA Visualization
            if len(selected_features) >= 3:
                st.markdown("**🎮 3D PCA Visualization:**")
                pca_3d = PCA(n_components=3, random_state=42)
                features_3d = pca_3d.fit_transform(features_scaled)
                
                model_for_3d = st.selectbox("Select model for 3D visualization", list(results.keys()))
                if model_for_3d:
                    fig = px.scatter_3d(
                        x=features_3d[:, 0],
                        y=features_3d[:, 1],
                        z=features_3d[:, 2],
                        color=results_df[f'{model_for_3d}_Cluster'],
                        title=f'3D PCA - {model_for_3d}',
                        labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Export results
            st.markdown("**💾 Download Clustering Results:**")
            
            # NEW: Multiple export formats
            if export_format == "CSV":
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                data_to_export = csv_buffer.getvalue()
                mime_type = "text/csv"
                file_ext = "csv"
            elif export_format == "Excel":
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Clustering Results', index=False)
                    metrics_df.to_excel(writer, sheet_name='Model Metrics', index=False)
                data_to_export = excel_buffer.getvalue()
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                file_ext = "xlsx"
            else:  # JSON
                import json
                json_data = {
                    'clustering_results': results_df.to_dict('records'),
                    'model_metrics': metrics_df.to_dict('records'),
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'scaling_method': scaling_method,
                        'features_used': selected_features
                    }
                }
                data_to_export = json.dumps(json_data, indent=2)
                mime_type = "application/json"
                file_ext = "json"
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label=f"📥 Download Results as {export_format}",
                    data=data_to_export,
                    file_name=f"clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                    mime=mime_type,
                    type="primary",
                    use_container_width=True
                )
            
            # NEW: Export visualizations
            st.markdown("**🎨 Export Visualizations:**")
            export_viz = st.checkbox("Export visualizations as HTML (interactive)")
            if export_viz:
                # Save PCA plot
                fig_html = fig.to_html()
                b64 = base64.b64encode(fig_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="pca_plot.html">📊 Download PCA Plot</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        else:
            st.warning("⚠️ No clustering results available. Please run clustering models in the 'Clustering' tab.")

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center;">Enhanced Clustering POC | Built with 🤖 | Streamlit, Scikit-Learn & Plotly</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    st.info("👋 Welcome! Upload a CSV file to get started with advanced automated clustering analysis.")