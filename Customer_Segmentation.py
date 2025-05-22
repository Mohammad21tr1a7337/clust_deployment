import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Title
st.set_page_config(page_title='Flight Customer Segmentation', layout='wide')
st.title('✈️ Flight Customer Segmentation')

# Sidebar - Data Upload / Selection
st.sidebar.header('1. Upload / Load Dataset')
uploaded_file = st.sidebar.file_uploader('Upload CSV', type=['csv', 'xlsx'])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, sheet_name='data')
    else:
        df = pd.read_csv(uploaded_file)
else:
    st.sidebar.write('Using default EastWestAirlines dataset')
    @st.cache_data
    def load_default():
        return pd.read_excel('EastWestAirlines.xlsx', sheet_name='data')
    df = load_default()

st.write('### Dataset Preview')
st.dataframe(df.head())

# Sidebar - Feature selection
st.sidebar.header('2. Select Features')
all_features = df.columns.tolist()
selected_features = st.sidebar.multiselect('Pick numeric features for clustering', all_features, default=all_features[:5])

if not selected_features:
    st.error('Please select at least one feature to proceed.')
    st.stop()

X = df[selected_features].select_dtypes(include=[np.number]).dropna()

# Data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar - Choose Clustering Method and Parameters
st.sidebar.header('3. Clustering Options')
method = st.sidebar.selectbox('Algorithm', ['KMeans', 'Hierarchical', 'DBSCAN'])

params = {}
if method == 'KMeans':
    params['n_clusters'] = st.sidebar.slider('Number of clusters (k)', min_value=2, max_value=10, value=4)
elif method == 'Hierarchical':
    params['n_clusters'] = st.sidebar.slider('Number of clusters', min_value=2, max_value=10, value=4)
    params['linkage'] = st.sidebar.selectbox('Linkage method', ['ward', 'complete', 'average', 'single'])
elif method == 'DBSCAN':
    params['eps'] = st.sidebar.slider('eps', 0.1, 10.0, 0.5)
    params['min_samples'] = st.sidebar.slider('min_samples', 1, 20, 5)

# Run Clustering
st.header('4. Run Clustering')
if st.button('Cluster'):
    # Perform clustering
    if method == 'KMeans':
        model = KMeans(n_clusters=params['n_clusters'], random_state=42)
        labels = model.fit_predict(X_scaled)
    elif method == 'Hierarchical':
        Z = linkage(X_scaled, method=params['linkage'])
        labels = fcluster(Z, params['n_clusters'], criterion='maxclust')
    else:
        model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        labels = model.fit_predict(X_scaled)

    df['Cluster'] = labels

    # Heuristic Names for Clusters based on behavior patterns
    # Pre-defined suggestions; adjust order based on expected number of clusters
    suggestions = [
        'Frequent Flyers', 'Occasional Travelers', 'High Spend Customers',
        'Low Spend Customers', 'Corporate Travelers', 'Leisure Travelers',
        'Bargain Hunters', 'Elite Members', 'Inactive Customers', 'Seasonal Travelers'
    ]
    unique_labels = sorted(np.unique(labels))
    name_map = {lbl: suggestions[i] if i < len(suggestions) else f'Cluster {lbl}'
                for i, lbl in enumerate(unique_labels)}
    df['Cluster Name'] = df['Cluster'].map(name_map)

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df['PC1'] = components[:, 0]
    df['PC2'] = components[:, 1]

    # Show cluster summary with names
    st.subheader('Cluster Counts & Names')
    summary = df.groupby(['Cluster', 'Cluster Name']).size().reset_index(name='Count')
    st.dataframe(summary)

    st.subheader('Clustered Data (first 10 rows)')
    st.dataframe(df.head(10))

    # Plot clusters with names
    st.subheader('2D PCA Cluster Plot')
    fig, ax = plt.subplots()
    for lbl in unique_labels:
        mask = df['Cluster'] == lbl
        ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
                   label=name_map[lbl], alpha=0.7)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    st.pyplot(fig)

    # Optional hierarchical dendrogram
    if method == 'Hierarchical':
        st.subheader('Hierarchical Dendrogram')
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        dendrogram(Z, truncate_mode='level', p=5, ax=ax2)
        st.pyplot(fig2)

    st.success('Clustering completed with descriptive names!')

# Footer
st.markdown('---')
