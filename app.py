import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Used Car Analysis", layout="wide")

# --- 1. VERİ ÖN İŞLEME (AKILLI SÜTUN BULUCU) ---
@st.cache_data
def load_data():
    df = pd.read_csv("vehicle.csv")
    
    # Sütun isimlerini küçük harfe çevir ve boşlukları temizle
    df.columns = df.columns.str.lower().str.strip()
    
    # --- AKILLI SÜTUN EŞLEŞTİRME ---
    renames = {
        'make': 'manufacturer',
        'brand': 'manufacturer',
        'company': 'manufacturer',
        'mileage': 'odometer',
        'kms_driven': 'odometer',
        'fueltype': 'fuel',
        'fuel_type': 'fuel',
        'transmission_type': 'transmission',
        'model_year': 'year'
    }
    df = df.rename(columns=renames)
    
    # Eğer manufacturer yoksa ilk metin sütununu alalım
    if 'manufacturer' not in df.columns:
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            df = df.rename(columns={text_cols[0]: 'manufacturer'})

    # Eksik verileri temizle
    df = df.dropna()
    
    # Veri tipi düzeltme (Fiyat içindeki $ veya , işaretlerini sil)
    if 'price' in df.columns and df['price'].dtype == 'object':
        df['price'] = df['price'].astype(str).str.replace(r'[$,]', '', regex=True)
        
    # Sayısala çevirme
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')
    
    # Temizlik sonrası tekrar NaN kontrolü
    df = df.dropna(subset=['price', 'year', 'odometer'])

    # Aykırı Değer Temizliği
    df = df[(df['price'] > 500) & (df['price'] < 500000)]
    df = df[df['year'] > 1990]
    df = df[df['odometer'] < 500000]
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- SIDEBAR (FİLTRELER) ---
st.sidebar.header("Dashboard Filters")

min_year, max_year = int(df['year'].min()), int(df['year'].max())
year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (2010, 2020))

all_brands = sorted(df['manufacturer'].unique())
selected_brands = st.sidebar.multiselect("Select Brands", all_brands, default=all_brands[:5])

# Filtreleme
if selected_brands:
    filtered_df = df[(df['year'].between(*year_range)) & (df['manufacturer'].isin(selected_brands))]
else:
    filtered_df = df[df['year'].between(*year_range)]

# --- BAŞLIK VE GİRİŞ ---
st.title("🚗 Used Car Price Analysis Dashboard")
st.markdown("""
This project is designed to analyze price dynamics in the used car market. 
The dataset has been cleaned and presented with interactive visualizations.
""")
st.info(f"Number of Records Displayed: {len(filtered_df)} (Filtered)")

# --- SEKMELER (TABS) ---
tab1, tab2, tab3 = st.tabs(["Ali Sait Öz (Hierarchy)", "Berfin Öztürk (Trends)", "Arda Murat Abay (ML & Stats)"])

# =============================================================================
# 👤 TAB 1: ALİ SAİT ÖZ
# =============================================================================
with tab1:
    st.header("Categorical and Hierarchical Analysis - Ali Sait Öz")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Market Share by Brand & Transmission (Treemap)")
        # DÜZELTME: 'model' yerine 'transmission' kullanıyoruz çünkü model sütunu yok.
        fig_treemap = px.treemap(filtered_df, path=['manufacturer', 'transmission'], values='price', color='price',
                                 color_continuous_scale='RdBu', title="Market Share by Brand and Transmission")
        st.plotly_chart(fig_treemap, use_container_width=True)
        
    with col2:
        st.subheader("2. Hierarchy: Brand > Fuel > Transmission")
        fig_sunburst = px.sunburst(filtered_df.head(5000), path=['manufacturer', 'fuel', 'transmission'], 
                                   title="Distribution of Brand - Fuel - Transmission")
        st.plotly_chart(fig_sunburst, use_container_width=True)

    st.subheader("3. Average Price by Brand (Bar Chart)")
    # DÜZELTME: 'model' sütunu olmadığı için markaya (manufacturer) göre sıraladık.
    top_expensive = filtered_df.groupby('manufacturer')['price'].mean().sort_values(ascending=False).head(10).reset_index()
    fig_bar = px.bar(top_expensive, x='price', y='manufacturer', orientation='h', title="Top 10 Brands with Highest Average Price")
    st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================================
# 👤 TAB 2: BERFİN ÖZTÜRK
# =============================================================================
with tab2:
    st.header("Trend and Time Series Analysis - Berfin Öztürk")
    
    st.subheader("4. Price vs Mileage Evolution over Time (Animation)")
    st.caption("Press the Play button to watch the evolution over the years.")
    
    anim_df = filtered_df.sort_values('year')
    fig_anim = px.scatter(anim_df, x="odometer", y="price", animation_frame="year", 
                          color="manufacturer", size_max=60, range_x=[0,300000], range_y=[0,100000],
                          title="Evolution of Price vs Mileage Over Years")
    st.plotly_chart(fig_anim, use_container_width=True)
    
    st.subheader("5. Parallel Coordinates Plot")
    st.caption("Multidimensional relationship between Price, Year, and Odometer.")
    fig_parallel = px.parallel_coordinates(filtered_df.head(500), dimensions=['price', 'year', 'odometer'],
                                           color="price", title="Multivariate Analysis (First 500 Cars)")
    st.plotly_chart(fig_parallel, use_container_width=True)

    st.subheader("6. Average Price Trend (Line Chart)")
    yearly_trend = filtered_df.groupby('year')['price'].mean().reset_index()
    fig_line = px.line(yearly_trend, x='year', y='price', title="Average Price Change Over Years")
    st.plotly_chart(fig_line, use_container_width=True)

# =============================================================================
# 👤 TAB 3: ARDA MURAT ABAY
# =============================================================================
with tab3:
    st.header("Statistical Analysis and ML - Arda Murat Abay")
    
    st.subheader("7. K-Means Clustering (ML Segmentation)")
    st.write("We segment cars into 3 categories (Economy, Mid-range, Luxury) based on Price and Odometer features.")
    
    ml_df = filtered_df[['price', 'odometer']].dropna()
    
    if len(ml_df) > 0:
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
        ml_df['cluster'] = kmeans.fit_predict(ml_df)
        ml_df['cluster'] = ml_df['cluster'].astype(str)
        # Kümeleri isimlendirme 
        cluster_means = ml_df.groupby('cluster')['price'].mean().sort_values()
        cluster_map = {
            cluster_means.index[0]: 'Economy',
            cluster_means.index[1]: 'Mid-range',
            cluster_means.index[2]: 'Luxury'
        }
        ml_df['cluster'] = ml_df['cluster'].map(cluster_map)
        
        fig_cluster = px.scatter(ml_df, x='odometer', y='price', color='cluster', 
                                 title="Car Segmentation (Clustering Analysis)",
                                 labels={'cluster': 'Segment'})
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.warning("Not enough data for clustering.")

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("8. Price vs Odometer Density (Heatmap)")
        fig_heatmap = px.density_heatmap(filtered_df, x="odometer", y="price", nbinsx=20, nbinsy=20, 
                                         title="Price and Odometer Density Heatmap")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    with col4:
        st.subheader("9. Price Distribution by Fuel Type")
        fig_box = px.box(filtered_df, x="fuel", y="price", color="fuel", title="Price Distribution by Fuel Type")
        st.plotly_chart(fig_box, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("CEN445 Project - 2025 | Github Repository: [https://github.com/berfinozturk/CEN445-Car-Analysis]")

