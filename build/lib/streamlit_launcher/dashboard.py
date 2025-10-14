import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
from wordcloud import WordCloud
import plotly.figure_factory as ff
warnings.filterwarnings('ignore')
import seaborn as sns

# --- Konfigurasi halaman ---
st.set_page_config(
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📊 Dashboard Analisis Data Lanjutan")

# CSS kustom untuk styling
st.markdown("""
<meta name="google-site-verification" content="ryAdKrOiPgVE9lQjxBAPCNbxtfCOJkDg_pvo7dzlp4U" />
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .data-table th, .data-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .data-table th {
        background-color: #f2f2f2;
    }
    .data-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .up-trend {
        color: green;
        font-weight: bold;
    }
    .down-trend {
        color: red;
        font-weight: bold;
    }
    .stock-summary {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border-left: 5px solid #1f77b4;
    }
    .chart-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .chart-description {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 4px solid #1f77b4;
    }
    .slider-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .kpi-title {
        font-size: 1rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Cache untuk performa
@st.cache_data(show_spinner=False)
def process_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"CSV berhasil dibaca: {uploaded_file.name}")
            
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            # Coba beberapa engine untuk membaca Excel
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            except:
                try:
                    df = pd.read_excel(uploaded_file, engine='xlrd')
                except:
                    st.error("Tidak dapat membaca file Excel. Pastikan openpyxl atau xlrd terinstall.")
                    return None
            
            st.sidebar.success(f"Excel berhasil dibaca: {uploaded_file.name}")
                
        else:
            st.error("Format file tidak didukung. Harap unggah file CSV atau Excel.")
            return None
        
        df.columns = df.columns.str.strip()
        df = auto_convert_dates(df)
        return df
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {str(e)}")
        st.info("Untuk file Excel, pastikan dependensi 'openpyxl' terinstall. Jalankan: pip install openpyxl")
        return None

@st.cache_data(show_spinner=False)
def auto_convert_dates(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                temp_col = pd.to_datetime(df[col], errors='coerce')
                if temp_col.notna().sum() / len(df) > 0.7:
                    df[col] = temp_col
                    st.sidebar.info(f"Kolom '{col}' dikonversi otomatis ke format tanggal")
            except:
                pass
    return df

@st.cache_data(show_spinner=False)
def merge_datasets(datasets, merge_method='concat'):
    if not datasets:
        return None
    
    if merge_method == 'concat':
        combined_df = pd.concat(datasets, ignore_index=True)
        st.sidebar.success(f"Berhasil menggabungkan {len(datasets)} dataset (concat)")
        return combined_df
    else:
        common_columns = set(datasets[0].columns)
        for dataset in datasets[1:]:
            common_columns = common_columns.intersection(set(dataset.columns))
        
        if not common_columns:
            st.error("Tidak ada kolom yang sama untuk melakukan penggabungan.")
            return None
        
        merge_key = list(common_columns)[0]
        st.sidebar.info(f"Menggunakan kolom '{merge_key}' sebagai kunci penggabungan")
        
        merged_df = datasets[0]
        for i in range(1, len(datasets)):
            try:
                merged_df = pd.merge(merged_df, datasets[i], how=merge_method, on=merge_key, suffixes=('', f'_{i}'))
            except Exception as e:
                st.error(f"Error saat menggabungkan dataset: {str(e)}")
                return None
        
        st.sidebar.success(f"Berhasil menggabungkan {len(datasets)} dataset ({merge_method} join)")
        return merged_df

# Fungsi untuk membuat semua jenis visualisasi
def create_all_visualizations(df):
    if df is None or df.empty:
        st.error("Data tidak tersedia atau kosong")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.error("Tidak ditemukan kolom numerik dalam dataset.")
        return
    
    # Sidebar untuk konfigurasi chart
    st.sidebar.header("🎛️ Konfigurasi Visualisasi")
    
    # Pilihan jenis chart yang diperluas
    chart_types = [
        "📈 Grafik Garis (Line Chart)",
        "📊 Grafik Batang (Bar Chart)", 
        "📋 Histogram Responsif Berwarna",  # DIUBAH NAMA DI SINI
        "🔄 Kombinasi Grafik Garis & Batang",
        "🔵 Scatter Plot",
        "🫧 Grafik Gelembung (Bubble Chart)",
        "🎯 Grafik Gauge (Speedometer)",
        "🕷️ Grafik Radar (Spider Chart)",
        "📦 Diagram Bingkai (Box Plot)",
        "🍾 Grafik Corong (Funnel Chart)",
        "🥧 Pie Chart dengan Slider",
        "📊 Scorecard / KPI Indicator",
        "🎯 Bullet Chart",
        "🌳 Treemap",
        "☁️ Word Cloud",
        "📅 Grafik Gantt (Gantt Chart)",
        "🗺️ Grafik Peta (Map Chart)",
        "🌊 Grafik Peta Aliran (Flow Map)",
        "🔥 Heatmap",
        "📈 Multiple Line Chart"
    ]
    
    chart_type = st.sidebar.selectbox("Pilih Jenis Chart", chart_types, key="chart_type_select")
    
    try:
        # Container untuk chart
        chart_container = st.container()
        
        with chart_container:
            st.markdown(f"### {chart_type}")
            
            if chart_type == "📈 Grafik Garis (Line Chart)":
                create_line_chart(df, numeric_cols, non_numeric_cols)
                
            elif chart_type == "📊 Grafik Batang (Bar Chart)":
                create_bar_chart(df, numeric_cols, non_numeric_cols)
                
            elif chart_type == "📋 Histogram Responsif Berwarna":  # DIUBAH DI SINI
                create_responsive_histogram(df, numeric_cols)  # PASTIKAN NAMA FUNGSI SESUAI
                
            elif chart_type == "🔄 Kombinasi Grafik Garis & Batang":
                create_combined_chart(df, numeric_cols, non_numeric_cols)
                
            elif chart_type == "🔵 Scatter Plot":
                create_scatter_plot(df, numeric_cols, non_numeric_cols)
                
            elif chart_type == "🫧 Grafik Gelembung (Bubble Chart)":
                create_bubble_chart(df, numeric_cols, non_numeric_cols)
                
            elif chart_type == "🎯 Grafik Gauge (Speedometer)":
                create_gauge_chart(df, numeric_cols)
                
            elif chart_type == "🕷️ Grafik Radar (Spider Chart)":
                create_radar_chart(df, numeric_cols, non_numeric_cols)
                
            elif chart_type == "📦 Diagram Bingkai (Box Plot)":
                create_box_plot(df, numeric_cols)
                
            elif chart_type == "🍾 Grafik Corong (Funnel Chart)":
                create_funnel_chart(df, numeric_cols, non_numeric_cols)
                
            elif chart_type == "🥧 Pie Chart dengan Slider":
                create_pie_chart_with_slider(df, numeric_cols, non_numeric_cols)
                
            elif chart_type == "📊 Scorecard / KPI Indicator":
                create_kpi_scorecard(df, numeric_cols)
                
            elif chart_type == "🎯 Bullet Chart":
                create_bullet_chart(df, numeric_cols)
                
            elif chart_type == "🌳 Treemap":
                create_treemap(df, numeric_cols, non_numeric_cols)
                
            elif chart_type == "☁️ Word Cloud":
                create_wordcloud(df, non_numeric_cols)
                
            elif chart_type == "📅 Grafik Gantt (Gantt Chart)":
                create_gantt_chart(df)
                
            elif chart_type == "🗺️ Grafik Peta (Map Chart)":
                create_map_chart(df)
                
            elif chart_type == "🌊 Grafik Peta Aliran (Flow Map)":
                create_flow_map(df)
                
            elif chart_type == "🔥 Heatmap":
                create_heatmap(df, numeric_cols)
                
            elif chart_type == "📈 Multiple Line Chart":
                create_multiple_line_chart(df, numeric_cols, non_numeric_cols)
                
    except Exception as e:
        st.error(f"Error dalam membuat visualisasi: {str(e)}")
        st.error("Pastikan semua library yang diperlukan sudah diimport")

def create_responsive_histogram(df, numeric_cols):
    
    # Deteksi ukuran data
    data_size = len(df)
    if data_size > 100000:
        st.info(f"⚡ Mode Optimasi: Data besar ({data_size:,} rows) - Menggunakan sampling otomatis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_col = st.selectbox("Pilih kolom numerik", numeric_cols, key="histogram_col")
    
    with col2:
        num_bins = st.slider("Jumlah bins", min_value=5, max_value=100, value=min(30, data_size//1000), 
                           key="hist_bins")
    
    with col3:
        color_theme = st.selectbox("Tema warna", 
                                 ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Blues"],
                                 key="hist_color")
    
    with col4:
        # Pengaturan optimasi
        optimization_mode = st.selectbox(
            "Mode Optimasi",
            ["Auto", "Fast", "Balanced", "Detailed"],
            index=0 if data_size > 100000 else 2,
            key="hist_optim"
        )

    if selected_col:
        try:
            with st.spinner("🔄 Memproses data histogram..."):
                # OPTIMASI 1: Filter dan sampling data
                clean_data = optimize_histogram_data(df[selected_col], data_size, optimization_mode)
                
                if len(clean_data) == 0:
                    st.warning(f"Tidak ada data valid untuk kolom {selected_col}")
                    return
                
                # OPTIMASI 2: Batasi bins untuk data sangat besar
                if len(clean_data) > 100000:
                    num_bins = min(num_bins, 50)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Buat histogram yang dioptimalkan
                    fig = create_optimized_histogram(clean_data, selected_col, num_bins, color_theme, data_size)
                    st.plotly_chart(fig, use_container_width=True, 
                                  config={'displayModeBar': True, 'responsive': True})
                
                with col2:
                    # Statistik cepat
                    display_quick_statistics(clean_data, selected_col)

                # OPTIMASI 3: Multiple histogram dengan data terbatas
                if data_size <= 50000:  # Hanya tampilkan untuk data tidak terlalu besar
                    with st.expander("🔍 Bandingkan Distribusi", expanded=False):
                        compare_cols = st.multiselect(
                            "Pilih kolom untuk perbandingan", 
                            numeric_cols[:5],  # Batasi pilihan
                            default=[selected_col],
                            key="hist_compare"
                        )
                        
                        if len(compare_cols) > 1:
                            fig_compare = create_comparison_histogram(df, compare_cols, optimization_mode)
                            st.plotly_chart(fig_compare, use_container_width=True)
                
                # Tampilkan info optimasi
                show_histogram_optimization_info(data_size, len(clean_data), optimization_mode)

        except Exception as e:
            st.error(f"Error membuat histogram: {str(e)}")
            # Fallback ke metode sederhana
            create_simple_histogram_fallback(df[selected_col].dropna(), selected_col)

def optimize_histogram_data(data_series, data_size, optimization_mode):
    """Optimasi data untuk histogram dengan sampling yang tepat"""
    clean_data = data_series.dropna()
    
    if len(clean_data) == 0:
        return clean_data
    
    # Tentukan target sample size berdasarkan mode optimasi
    target_sizes = {
        "Auto": min(10000, data_size) if data_size > 50000 else data_size,
        "Fast": min(5000, data_size),
        "Balanced": min(20000, data_size),
        "Detailed": data_size
    }
    
    target_size = target_sizes[optimization_mode]
    
    # Jika data lebih besar dari target, lakukan sampling
    if len(clean_data) > target_size:
        if optimization_mode == "Fast":
            # Systematic sampling untuk performa maksimal
            step = len(clean_data) // target_size
            sampled_data = clean_data.iloc[::step]
        else:
            # Stratified sampling untuk mempertahankan distribusi
            try:
                # Bin data terlebih dahulu, lalu sample dari setiap bin
                n_bins = min(100, target_size // 10)
                bins = pd.cut(clean_data, bins=n_bins)
                stratified_sample = clean_data.groupby(bins, observed=False, group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), max(1, target_size // n_bins)), random_state=42)
                )
                sampled_data = stratified_sample
            except:
                # Fallback ke random sampling
                sampled_data = clean_data.sample(n=target_size, random_state=42)
        
        return sampled_data
    
    return clean_data

def create_optimized_histogram(data, column_name, num_bins, color_theme, original_size):
    """Buat histogram dengan optimasi performa"""
    
    # Mapping warna yang dioptimalkan
    color_sequences = {
        "Viridis": ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
        "Plasma": ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921'],
        "Inferno": ['#000004', '#3b0f70', '#8c2981', '#de4968', '#fe9f6d'],
        "Magma": ['#000004', '#4a0c6b', '#a52c60', '#e95e3c', '#feca8d'],
        "Cividis": ['#00204d', '#31446b', '#666870', '#958f78', '#ffea46'],
        "Blues": ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6']
    }
    
    selected_color = color_sequences.get(color_theme, color_sequences["Viridis"])[2]
    
    # OPTIMASI: Gunakan numpy untuk perhitungan histogram yang lebih cepat
    hist, bin_edges = np.histogram(data, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    fig = go.Figure()
    
    # Trace histogram yang dioptimalkan
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist,
        width=np.diff(bin_edges),
        name=column_name,
        marker_color=selected_color,
        opacity=0.8,
        hovertemplate='<b>Range: %{x:.2f}</b><br>Frekuensi: %{y}<extra></extra>'
    ))
    
    # OPTIMASI: Density plot hanya untuk data yang tidak terlalu besar
    if len(data) <= 10000:
        try:
            from scipy.stats import gaussian_kde
            # Sample data untuk density calculation
            if len(data) > 2000:
                density_data = data.sample(n=2000, random_state=42)
            else:
                density_data = data
                
            x_range = np.linspace(data.min(), data.max(), 100)
            density = gaussian_kde(density_data)(x_range)
            hist_area = len(data) * (data.max() - data.min()) / num_bins
            scaled_density = density * hist_area
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=scaled_density,
                mode='lines',
                line=dict(color='red', width=2),
                name='Density Estimate',
                hovertemplate='<b>Density: %{y:.2f}</b><extra></extra>'
            ))
        except:
            pass  # Skip density plot jika error
    
    # Layout yang dioptimalkan
    fig.update_layout(
        title=f"Distribusi {column_name} ({len(data):,} data points)",
        height=450,
        showlegend=True,
        bargap=0.05,
        xaxis_title=column_name,
        yaxis_title="Frekuensi",
        margin=dict(l=50, r=50, t=60, b=50),
        plot_bgcolor='white'
    )
    
    return fig

def display_quick_statistics(data, column_name):
    """Tampilkan statistik cepat yang dioptimalkan"""
    st.markdown("### 📊 Statistik Cepat")
    
    # Hitung statistik dasar dengan numpy (lebih cepat)
    stats_data = {
        "Rata-rata": f"{np.mean(data):.2f}",
        "Median": f"{np.median(data):.2f}",
        "Std Dev": f"{np.std(data):.2f}",
        "Min": f"{np.min(data):.2f}",
        "Max": f"{np.max(data):.2f}",
        "Jumlah Data": f"{len(data):,}"
    }
    
    for key, value in stats_data.items():
        st.metric(key, value)
    
    # Hitung skewness dengan optimasi
    try:
        if len(data) > 2:
            skew_val = (3 * (np.mean(data) - np.median(data))) / np.std(data)  # Approximation
            st.metric("Skewness", f"{skew_val:.2f}")
            
            # Info distribusi
            st.markdown("### 📈 Info Distribusi")
            if abs(skew_val) < 0.5:
                st.success("**Normal**")
            elif skew_val > 0.5:
                st.warning("**Right-skewed**")
            else:
                st.warning("**Left-skewed**")
    except:
        pass

def create_comparison_histogram(df, compare_cols, optimization_mode):
    """Buat histogram perbandingan yang dioptimalkan"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, col in enumerate(compare_cols[:4]):  # Maksimal 4 kolom
        clean_data = optimize_histogram_data(df[col], len(df), optimization_mode)
        if len(clean_data) > 0:
            fig.add_trace(go.Histogram(
                x=clean_data,
                name=col,
                opacity=0.6,
                nbinsx=20,  # Fixed bins untuk performa
                marker_color=colors[i % len(colors)],
                hovertemplate=f'<b>{col}</b><br>%{{x}}</b><br>Frekuensi: %{{y}}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Perbandingan Distribusi (Optimized)",
        barmode='overlay',
        height=400,
        xaxis_title="Nilai",
        yaxis_title="Frekuensi",
        showlegend=True
    )
    
    return fig

def show_histogram_optimization_info(original_size, processed_size, optimization_mode):
    """Tampilkan informasi optimasi"""
    reduction_pct = ((original_size - processed_size) / original_size) * 100 if original_size > 0 else 0
    
    if reduction_pct > 10:
        with st.expander("⚡ Info Optimasi Performa", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Original", f"{original_size:,}")
            with col2:
                st.metric("Data Diproses", f"{processed_size:,}")
            with col3:
                st.metric("Reduksi", f"{reduction_pct:.1f}%")
            
            st.info(f"**Mode {optimization_mode}**: Histogram dioptimalkan untuk kecepatan rendering")

def create_simple_histogram_fallback(data, column_name):
    """Fallback method untuk data yang bermasalah"""
    st.warning("Menggunakan metode fallback yang sederhana...")
    
    if len(data) > 1000:
        data = data.sample(n=1000, random_state=42)
    
    fig = px.histogram(x=data, nbins=20, title=f"Simple Histogram - {column_name}")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Versi ultra-ringan untuk data ekstrem
def create_ultra_fast_histogram(df, numeric_cols):
    """Versi ultra-ringan untuk data > 500k rows"""
    
    col1, col2 = st.columns(2)
    with col1:
        selected_col = st.selectbox("Pilih kolom", numeric_cols[:10], key="ultra_hist_col")
    with col2:
        num_bins = st.slider("Bins", 5, 50, 20, key="ultra_bins")
    
    if selected_col:
        # Sampling agresif
        if len(df) > 5000:
            data = df[selected_col].dropna().sample(n=5000, random_state=42)
        else:
            data = df[selected_col].dropna()
        
        # Histogram sederhana
        fig = px.histogram(x=data, nbins=num_bins, 
                         title=f"Ultra-Fast: {selected_col} (5,000 samples)")
        fig.update_layout(height=350, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.info(f"📊 Menampilkan 5,000 sample dari {len(df[selected_col].dropna()):,} data points")

# FUNGSI BARU: Pie Chart dengan Slider
def create_pie_chart_with_slider(df, numeric_cols, non_numeric_cols):
    
    col1, col2 = st.columns(2)
    
    with col1:
        category_col = st.selectbox("Pilih kolom kategori", non_numeric_cols, key="pie_category")
    with col2:
        value_col = st.selectbox("Pilih kolom nilai", numeric_cols, key="pie_value")
    
    if category_col and value_col:
        # Agregasi data
        pie_data = df.groupby(category_col)[value_col].sum().reset_index()
        pie_data = pie_data.sort_values(value_col, ascending=False)
        
        # Hitung persentase
        total_value = pie_data[value_col].sum()
        pie_data['percentage'] = (pie_data[value_col] / total_value * 100).round(2)
        
        # Slider untuk threshold persentase
        st.markdown('<div class="slider-container">', unsafe_allow_html=True)
        threshold = st.slider(
            "Threshold Persentase untuk 'Lainnya' (%)", 
            min_value=0.0, 
            max_value=20.0, 
            value=2.0, 
            step=0.5,
            help="Kategori dengan persentase di bawah nilai ini akan digabung menjadi 'Lainnya'"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Proses data berdasarkan threshold
        main_categories = pie_data[pie_data['percentage'] >= threshold]
        other_categories = pie_data[pie_data['percentage'] < threshold]
        
        if len(other_categories) > 0:
            other_total = other_categories[value_col].sum()
            other_percentage = other_categories['percentage'].sum()
            
            final_data = pd.concat([
                main_categories,
                pd.DataFrame({
                    category_col: ['Lainnya'],
                    value_col: [other_total],
                    'percentage': [other_percentage]
                })
            ], ignore_index=True)
        else:
            final_data = main_categories
        
        # Buat pie chart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(
                final_data, 
                values=value_col, 
                names=category_col,
                title=f"Distribusi {value_col} per {category_col}",
                hover_data=['percentage'],
                labels={'percentage': 'Persentase (%)'}
            )
            
            # Customisasi tampilan
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Nilai: %{value}<br>Persentase: %{percent}<extra></extra>'
            )
            
            fig.update_layout(
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Detail Kategori")
            st.markdown(f"**Total {value_col}:** {total_value:,.2f}")
            st.markdown(f"**Jumlah Kategori:** {len(pie_data)}")
            st.markdown(f"**Kategori ditampilkan:** {len(main_categories)}")
            if len(other_categories) > 0:
                st.markdown(f"**Kategori di 'Lainnya':** {len(other_categories)}")
            
            st.markdown("---")
            st.markdown("**Top 5 Kategori:**")
            for i, row in pie_data.head().iterrows():
                st.markdown(f"• {row[category_col]}: {row['percentage']:.1f}%")
        
        # Tampilkan tabel data detail
        with st.expander("📋 Lihat Data Detail"):
            display_data = final_data.copy()
            display_data[value_col] = display_data[value_col].round(2)
            display_data['percentage'] = display_data['percentage'].round(2)
            st.dataframe(display_data[[category_col, value_col, 'percentage']], use_container_width=True)
        
        # Keterangan
        with st.expander("ℹ️ Keterangan Pie Chart dengan Slider"):
            st.markdown("""
            **Pie Chart dengan Slider** memungkinkan Anda mengontrol tampilan kategori berdasarkan persentase.
            - **Slider Threshold**: Mengatur batas minimum persentase untuk menampilkan kategori secara individual
            - **Kategori 'Lainnya'**: Semua kategori di bawah threshold akan digabung
            - **Kelebihan**: Fleksibel dalam menampilkan data, menghindari chart yang terlalu penuh
            - **Penggunaan**: Distribusi data dengan banyak kategori, analisis komposisi
            """)

def create_kpi_scorecard(df, numeric_cols):
    
    # Deteksi ukuran data
    data_size = len(df)
    if data_size > 100000:
        st.info(f"⚡ Mode Optimasi: Data besar ({data_size:,} rows) - Menggunakan kalkulasi cepat")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        kpi_col = st.selectbox("Pilih kolom untuk KPI", numeric_cols, key="kpi_col")
    
    with col2:
        calculation_type = st.selectbox("Jenis perhitungan", 
                                      ["Mean", "Sum", "Median", "Max", "Min", "Percentile"], 
                                      key="kpi_calc")
        
        if calculation_type == "Percentile":
            percentile_val = st.slider("Percentile", 0, 100, 90, key="kpi_percentile")
    
    with col3:
        optimization_mode = st.selectbox(
            "Mode Optimasi",
            ["Auto", "Fast", "Balanced", "Detailed"],
            index=0 if data_size > 50000 else 2,
            key="kpi_optim"
        )
    
    if kpi_col:
        try:
            with st.spinner("🔄 Menghitung KPI metrics..."):
                # OPTIMASI 1: Kalkulasi nilai KPI yang efisien
                kpi_results = calculate_kpi_values(df, kpi_col, calculation_type, 
                                                 percentile_val if 'percentile_val' in locals() else None,
                                                 data_size, optimization_mode)
                
                if kpi_results is None:
                    st.warning(f"Tidak ada data valid untuk kolom {kpi_col}")
                    return
                
                # Tampilkan KPI cards utama
                display_main_kpi_cards(kpi_results, kpi_col, calculation_type)
                
                # Tampilkan trend analysis
                display_trend_analysis(df, kpi_col, kpi_results, data_size, optimization_mode)
                
                # Tampilkan additional metrics
                display_additional_metrics(kpi_results, data_size)
                
                # Tampilkan info optimasi
                show_kpi_optimization_info(data_size, kpi_results['sample_size'], optimization_mode)
                
        except Exception as e:
            st.error(f"Error menghitung KPI: {str(e)}")
            # Fallback ke metode sederhana
            create_simple_kpi_fallback(df, kpi_col)

def calculate_kpi_values(df, kpi_col, calculation_type, percentile_val, data_size, optimization_mode):
    """Hitung nilai KPI dengan optimasi untuk data besar"""
    
    # OPTIMASI: Sampling untuk data besar
    if data_size > 100000:
        target_sizes = {
            "Auto": min(10000, data_size),
            "Fast": min(5000, data_size),
            "Balanced": min(20000, data_size),
            "Detailed": min(50000, data_size)
        }
        
        target_size = target_sizes[optimization_mode]
        
        if data_size > target_size:
            # Systematic sampling untuk performa
            step = data_size // target_size
            sample_data = df[kpi_col].dropna().iloc[::step]
            sample_info = f"Sample: {len(sample_data):,} dari {data_size:,}"
        else:
            sample_data = df[kpi_col].dropna()
            sample_info = f"Full data: {len(sample_data):,}"
    else:
        sample_data = df[kpi_col].dropna()
        sample_info = f"Full data: {len(sample_data):,}"
    
    if len(sample_data) == 0:
        return None
    
    # OPTIMASI: Gunakan numpy untuk kalkulasi yang lebih cepat
    data_values = sample_data.values
    
    # Hitung nilai utama berdasarkan tipe kalkulasi
    if calculation_type == "Mean":
        main_value = np.mean(data_values)
    elif calculation_type == "Sum":
        if data_size > 100000:
            # Scale sum untuk data sample
            scale_factor = data_size / len(sample_data)
            main_value = np.sum(data_values) * scale_factor
        else:
            main_value = np.sum(data_values)
    elif calculation_type == "Median":
        main_value = np.median(data_values)
    elif calculation_type == "Max":
        main_value = np.max(data_values)
    elif calculation_type == "Min":
        main_value = np.min(data_values)
    elif calculation_type == "Percentile":
        main_value = np.percentile(data_values, percentile_val)
    else:
        main_value = np.mean(data_values)
    
    # Hitung statistik tambahan
    count = len(sample_data)
    std_dev = np.std(data_values)
    mean_val = np.mean(data_values)
    cv = (std_dev / mean_val * 100) if mean_val != 0 else 0
    
    # Hitung quartiles untuk trend analysis
    q1 = np.percentile(data_values, 25)
    q3 = np.percentile(data_values, 75)
    
    return {
        'main_value': float(main_value),
        'count': count,
        'std_dev': float(std_dev),
        'cv': float(cv),
        'mean': float(mean_val),
        'q1': float(q1),
        'q3': float(q3),
        'min': float(np.min(data_values)),
        'max': float(np.max(data_values)),
        'sample_size': len(sample_data),
        'sample_info': sample_info,
        'data_size': data_size
    }

def display_main_kpi_cards(kpi_results, kpi_col, calculation_type):
    """Tampilkan KPI cards utama"""
    
    # CSS untuk KPI cards
    st.markdown("""
    <style>
    .kpi-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 5px solid #4CAF50;
        margin: 5px;
    }
    .kpi-title {
        font-size: 14px;
        color: #666;
        margin-bottom: 5px;
        font-weight: 500;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin: 10px 0;
    }
    .kpi-trend-up {
        color: #4CAF50;
        font-weight: bold;
    }
    .kpi-trend-down {
        color: #f44336;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns(4)
    
    with cols[0]:
        # Main KPI value
        value = kpi_results['main_value']
        display_value = f"{value:,.2f}" if abs(value) < 1000000 else f"{value/1000000:.2f}M"
        
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #4CAF50;">
            <div class="kpi-title">{calculation_type}</div>
            <div class="kpi-value">{display_value}</div>
            <div class="kpi-title">{kpi_col}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        # Data points
        count = kpi_results['count']
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #2196F3;">
            <div class="kpi-title">Data Points</div>
            <div class="kpi-value">{count:,}</div>
            <div class="kpi-title">{kpi_results['sample_info']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        # Variability
        std_dev = kpi_results['std_dev']
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #FF9800;">
            <div class="kpi-title">Standard Deviation</div>
            <div class="kpi-value">{std_dev:.2f}</div>
            <div class="kpi-title">Variability</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        # Coefficient of Variation
        cv = kpi_results['cv']
        cv_status = "Low" if cv < 30 else "Medium" if cv < 70 else "High"
        
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #9C27B0;">
            <div class="kpi-title">Coef of Variation</div>
            <div class="kpi-value">{cv:.1f}%</div>
            <div class="kpi-title">{cv_status} Variability</div>
        </div>
        """, unsafe_allow_html=True)

def display_trend_analysis(df, kpi_col, kpi_results, data_size, optimization_mode):
    """Tampilkan analisis trend"""
    
    st.subheader("📈 Trend Analysis")
    
    trend_cols = st.columns(3)
    
    with trend_cols[0]:
        # Distribution skewness
        try:
            if kpi_results['sample_size'] > 2:
                # Approximation skewness untuk performa
                skewness = (3 * (kpi_results['mean'] - kpi_results['main_value'])) / kpi_results['std_dev']
                skewness = 0 if abs(skewness) > 10 else skewness  # Handle outliers
                
                trend_icon = "📊" if abs(skewness) < 0.5 else "📈" if skewness > 0 else "📉"
                skew_label = "Normal" if abs(skewness) < 0.5 else "Right-skewed" if skewness > 0 else "Left-skewed"
                
                st.metric(
                    f"Distribution {trend_icon}",
                    skew_label,
                    delta=f"Skew: {skewness:.2f}"
                )
        except:
            st.metric("Distribution", "Normal", delta="Skew: N/A")
    
    with trend_cols[1]:
        # Data range efficiency
        data_range = kpi_results['max'] - kpi_results['min']
        if data_range > 0:
            iqr = kpi_results['q3'] - kpi_results['q1']
            range_efficiency = (iqr / data_range) * 100
            efficiency_status = "Good" if range_efficiency > 50 else "Moderate"
            
            st.metric(
                "Data Concentration",
                efficiency_status,
                delta=f"{range_efficiency:.1f}% in IQR"
            )
        else:
            st.metric("Data Concentration", "Constant", delta="No variation")
    
    with trend_cols[2]:
        # Data quality
        completeness = (kpi_results['count'] / kpi_results['data_size']) * 100
        quality_status = "Excellent" if completeness > 95 else "Good" if completeness > 80 else "Poor"
        
        st.metric(
            "Data Quality",
            quality_status,
            delta=f"{completeness:.1f}% complete"
        )

def display_additional_metrics(kpi_results, data_size):
    """Tampilkan metrics tambahan"""
    
    with st.expander("📊 Additional Metrics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Minimum", f"{kpi_results['min']:.2f}")
        with col2:
            st.metric("Q1 (25%)", f"{kpi_results['q1']:.2f}")
        with col3:
            st.metric("Q3 (75%)", f"{kpi_results['q3']:.2f}")
        with col4:
            st.metric("Maximum", f"{kpi_results['max']:.2f}")
        
        # Progress bars untuk visualisasi
        col5, col6 = st.columns(2)
        
        with col5:
            # Value distribution dalam IQR
            if kpi_results['max'] > kpi_results['min']:
                iqr_range = kpi_results['q3'] - kpi_results['q1']
                total_range = kpi_results['max'] - kpi_results['min']
                iqr_percentage = (iqr_range / total_range) * 100
                
                st.markdown(f"**IQR Coverage:** {iqr_percentage:.1f}%")
                st.progress(iqr_percentage/100)
        
        with col6:
            # Data completeness
            completeness = (kpi_results['count'] / data_size) * 100
            st.markdown(f"**Data Completeness:** {completeness:.1f}%")
            st.progress(completeness/100)

def show_kpi_optimization_info(original_size, processed_size, optimization_mode):
    """Tampilkan informasi optimasi"""
    
    reduction_pct = ((original_size - processed_size) / original_size) * 100 if original_size > 0 else 0
    
    if reduction_pct > 10:
        with st.expander("⚡ Info Optimasi Performa", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Original", f"{original_size:,}")
            with col2:
                st.metric("Data Diproses", f"{processed_size:,}")
            with col3:
                st.metric("Reduksi", f"{reduction_pct:.1f}%")
            
            optimization_strategies = {
                "Fast": "• ✅ **Aggressive sampling**\n• ✅ **Approximation methods**\n• ✅ **Minimal calculations**",
                "Balanced": "• ✅ **Smart sampling**\n• ✅ **Efficient numpy operations**\n• ✅ **Basic trend analysis**",
                "Detailed": "• ✅ **Maximum data retention**\n• ✅ **Comprehensive metrics**\n• ✅ **Full analysis**"
            }
            
            st.info(f"**Mode {optimization_mode}**: {optimization_strategies.get(optimization_mode, 'Custom optimization')}")

def create_simple_kpi_fallback(df, kpi_col):
    """Fallback method untuk data yang bermasalah"""
    st.warning("Menggunakan metode fallback sederhana...")
    
    # Kalkulasi sederhana dengan sample kecil
    sample_data = df[kpi_col].dropna().head(1000)
    
    if len(sample_data) == 0:
        st.error("Tidak ada data valid")
        return
    
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Mean", f"{sample_data.mean():.2f}")
    with cols[1]:
        st.metric("Count", f"{len(sample_data):,}")
    with cols[2]:
        st.metric("Std Dev", f"{sample_data.std():.2f}")

# Versi ultra-ringan untuk data ekstrem
def create_ultra_fast_kpi(df, numeric_cols):
    """Versi ultra-ringan untuk data > 500k rows"""
    st.subheader("🚀 KPI Scorecard Ultra-Fast")
    
    kpi_col = st.selectbox("Pilih kolom KPI", numeric_cols[:8], key="ultra_kpi_col")
    
    if kpi_col:
        # Sampling sangat agresif
        sample_data = df[kpi_col].dropna()
        if len(sample_data) > 5000:
            sample_data = sample_data.sample(n=5000, random_state=42)
        
        if len(sample_data) > 0:
            cols = st.columns(4)
            
            with cols[0]:
                st.metric("Mean", f"{sample_data.mean():.2f}")
            with cols[1]:
                st.metric("Count", f"{len(sample_data):,}")
            with cols[2]:
                st.metric("Std Dev", f"{sample_data.std():.2f}")
            with cols[3]:
                cv = (sample_data.std() / sample_data.mean() * 100) if sample_data.mean() != 0 else 0
                st.metric("CV", f"{cv:.1f}%")
            
            st.info(f"📊 Ultra-Fast Mode: 5,000 samples dari {len(df[kpi_col].dropna()):,} data points")

def create_bullet_chart(df, numeric_cols):
    
    # Deteksi ukuran data
    data_size = len(df)
    if data_size > 100000:
        st.info(f"⚡ Mode Optimasi: Data besar ({data_size:,} rows) - Menggunakan kalkulasi cepat")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        value_col = st.selectbox("Pilih kolom nilai", numeric_cols, key="bullet_value")
    
    with col2:
        target_col = st.selectbox("Pilih kolom target (opsional)", 
                                [None] + numeric_cols, 
                                key="bullet_target")
    
    with col3:
        performance_bands = st.slider("Jumlah performance bands", 2, 5, 3, key="bullet_bands")
    
    with col4:
        optimization_mode = st.selectbox(
            "Mode Optimasi",
            ["Auto", "Fast", "Balanced", "Detailed"],
            index=0 if data_size > 50000 else 2,
            key="bullet_optim"
        )
    
    # Pengaturan lanjutan
    with st.expander("⚙️ Pengaturan Lanjutan", expanded=False):
        col5, col6, col7 = st.columns(3)
        with col5:
            band_type = st.selectbox(
                "Tipe Performance Band",
                ["Auto Ranges", "Fixed Ranges", "Percentile Based"],
                key="bullet_band_type"
            )
        with col6:
            if band_type == "Fixed Ranges":
                good_range = st.slider("Good Range (%)", 80, 120, 100, key="bullet_good")
                excellent_range = st.slider("Excellent Range (%)", 100, 150, 120, key="bullet_excellent")
        with col7:
            marker_style = st.selectbox(
                "Style Marker",
                ["diamond", "circle", "square", "triangle-up"],
                key="bullet_marker"
            )
    
    if value_col:
        try:
            with st.spinner("🔄 Menghitung nilai bullet chart..."):
                # OPTIMASI 1: Kalkulasi nilai yang efisien
                bullet_data = calculate_bullet_values(
                    df, value_col, target_col, data_size, optimization_mode
                )
                
                if bullet_data is None:
                    st.warning(f"Tidak ada data valid untuk kolom {value_col}")
                    return
                
                # OPTIMASI 2: Buat bullet chart yang dioptimalkan
                fig = create_optimized_bullet_chart(
                    bullet_data, value_col, target_col, performance_bands, 
                    band_type, marker_style, optimization_mode
                )
                
                # OPTIMASI 3: Konfigurasi plotly yang ringan
                config = {
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestGl2d'],
                    'responsive': True
                }
                
                st.plotly_chart(fig, use_container_width=True, config=config)
                
                # Tampilkan performance summary
                display_performance_summary(bullet_data, value_col, target_col)
                
                # Tampilkan info optimasi
                show_bullet_optimization_info(data_size, bullet_data['sample_size'], optimization_mode)
                
        except Exception as e:
            st.error(f"Error membuat bullet chart: {str(e)}")
            # Fallback ke metode sederhana
            create_simple_bullet_fallback(df, value_col, target_col)

def calculate_bullet_values(df, value_col, target_col, data_size, optimization_mode):
    """Hitung nilai bullet chart dengan optimasi untuk data besar"""
    
    # OPTIMASI: Sampling untuk data besar
    if data_size > 100000:
        target_sizes = {
            "Auto": min(10000, data_size),
            "Fast": min(5000, data_size),
            "Balanced": min(20000, data_size),
            "Detailed": min(50000, data_size)
        }
        
        target_size = target_sizes[optimization_mode]
        
        if data_size > target_size:
            # Systematic sampling untuk performa
            step = data_size // target_size
            sample_df = df.iloc[::step]
            sample_info = f"Sample: {len(sample_df):,} dari {data_size:,}"
        else:
            sample_df = df
            sample_info = f"Full data: {len(sample_df):,}"
    else:
        sample_df = df
        sample_info = f"Full data: {len(sample_df):,}"
    
    # Filter data yang valid
    value_data = sample_df[value_col].dropna()
    if len(value_data) == 0:
        return None
    
    # Hitung nilai current
    current_value = np.mean(value_data.values)
    
    # Hitung nilai target
    if target_col:
        target_data = sample_df[target_col].dropna()
        if len(target_data) > 0:
            target_value = np.mean(target_data.values)
        else:
            target_value = current_value * 1.1
    else:
        # Auto-calculate target berdasarkan data
        target_value = np.percentile(value_data.values, 75)  # 75th percentile sebagai target
    
    # Hitung statistik tambahan untuk ranges
    min_val = np.min(value_data.values)
    max_val = np.max(value_data.values)
    std_dev = np.std(value_data.values)
    
    return {
        'current_value': float(current_value),
        'target_value': float(target_value),
        'min_value': float(min_val),
        'max_value': float(max_val),
        'std_dev': float(std_dev),
        'sample_size': len(sample_df),
        'sample_info': sample_info,
        'data_size': data_size
    }

def create_optimized_bullet_chart(bullet_data, value_col, target_col, performance_bands, band_type, marker_style, optimization_mode):
    """Buat bullet chart yang dioptimalkan"""
    
    fig = go.Figure()
    
    current_value = bullet_data['current_value']
    target_value = bullet_data['target_value']
    
    # OPTIMASI: Tentukan ranges berdasarkan tipe band
    ranges, colors, labels = calculate_performance_ranges(
        bullet_data, performance_bands, band_type, optimization_mode
    )
    
    # Add performance ranges (stacked bars)
    for i in range(performance_bands):
        range_width = ranges[i+1] - ranges[i]
        
        fig.add_trace(go.Bar(
            x=[range_width],
            y=["Performance"],
            orientation='h',
            marker=dict(
                color=colors[i % len(colors)], 
                opacity=0.3,
                line=dict(width=0)  # No border untuk performa
            ),
            name=labels[i] if i < len(labels) else f'Range {i+1}',
            base=ranges[i],
            hovertemplate=f'<b>{labels[i] if i < len(labels) else f"Range {i+1}"}</b><br>Range: {ranges[i]:.1f} - {ranges[i+1]:.1f}<extra></extra>',
            showlegend=performance_bands <= 4  # Sembunyikan legend jika terlalu banyak bands
        ))
    
    # Add target line
    fig.add_trace(go.Scatter(
        x=[target_value, target_value],
        y=["Performance", "Performance"],
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),
        name='Target',
        hovertemplate=f'<b>Target</b><br>Value: {target_value:.2f}<extra></extra>'
    ))
    
    # Add current value marker
    marker_size = 12 if optimization_mode in ["Balanced", "Detailed"] else 10
    fig.add_trace(go.Scatter(
        x=[current_value],
        y=["Performance"],
        mode='markers',
        marker=dict(
            color='black', 
            size=marker_size, 
            symbol=marker_style,
            line=dict(color='white', width=2)
        ),
        name='Current Value',
        hovertemplate=f'<b>Current Value</b><br>{current_value:.2f}<br>Achievement: {(current_value/target_value*100):.1f}%<extra></extra>'
    ))
    
    # Layout yang dioptimalkan
    layout_config = {
        'title': f"Bullet Chart: {value_col}" + (f" vs {target_col}" if target_col else ""),
        'xaxis': dict(
            title="Nilai",
            showgrid=True,
            gridcolor='lightgray',
            range=[bullet_data['min_value'] * 0.9, max(bullet_data['max_value'], target_value) * 1.1]
        ),
        'yaxis': dict(
            showticklabels=False,
            showgrid=False
        ),
        'showlegend': performance_bands <= 4,
        'height': 200 if optimization_mode == "Fast" else 250,
        'margin': dict(l=50, r=50, t=60, b=50),
        'plot_bgcolor': 'white',
        'barmode': 'stack'
    }
    
    fig.update_layout(**layout_config)
    
    return fig

def calculate_performance_ranges(bullet_data, performance_bands, band_type, optimization_mode):
    """Hitung performance ranges yang optimal"""
    
    target_value = bullet_data['target_value']
    min_val = bullet_data['min_value']
    max_val = bullet_data['max_value']
    std_dev = bullet_data['std_dev']
    
    # Warna berdasarkan jumlah bands
    if performance_bands == 2:
        colors = ['lightcoral', 'lightgreen']
        labels = ['Below Target', 'Above Target']
    elif performance_bands == 3:
        colors = ['lightcoral', 'lightyellow', 'lightgreen']
        labels = ['Poor', 'Good', 'Excellent']
    elif performance_bands == 4:
        colors = ['lightcoral', 'lightsalmon', 'lightyellow', 'lightgreen']
        labels = ['Poor', 'Fair', 'Good', 'Excellent']
    else:  # 5 bands
        colors = ['lightcoral', 'lightsalmon', 'lightyellow', 'lightblue', 'lightgreen']
        labels = ['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent']
    
    # Tentukan ranges berdasarkan tipe band
    if band_type == "Fixed Ranges":
        # Gunakan fixed percentage ranges
        ranges = [min_val]
        for i in range(1, performance_bands):
            percentage = (i / performance_bands) * 100
            ranges.append(target_value * (percentage / 100))
        ranges.append(max(max_val, target_value * 1.2))
        
    elif band_type == "Percentile Based":
        # Berdasarkan distribusi data
        ranges = [min_val]
        for i in range(1, performance_bands):
            percentile = (i / performance_bands) * 100
            ranges.append(np.percentile([min_val, target_value, max_val], percentile))
        ranges.append(max_val)
        
    else:  # Auto Ranges
        # Ranges otomatis berdasarkan target dan std dev
        ranges = [min_val]
        step_size = target_value / performance_bands
        
        for i in range(1, performance_bands):
            ranges.append(step_size * i)
        ranges.append(max(max_val, target_value * 1.2))
    
    return ranges, colors, labels

def display_performance_summary(bullet_data, value_col, target_col):
    """Tampilkan performance summary"""
    
    current_value = bullet_data['current_value']
    target_value = bullet_data['target_value']
    
    # Hitung performance metrics
    performance_ratio = (current_value / target_value * 100) if target_value != 0 else 0
    absolute_diff = current_value - target_value
    
    # PERBAIKAN: Gunakan delta_color yang valid
    if performance_ratio >= 100:
        status = "✅ Exceeded Target"
        delta_color = "normal"  # Hijau untuk positif
    elif performance_ratio >= 80:
        status = "⚠️ Near Target"
        delta_color = "off"     # Abu-abu untuk netral
    else:
        status = "❌ Below Target"
        delta_color = "inverse" # Merah untuk negatif
    
    st.subheader("📊 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Value", 
            f"{current_value:.2f}",
            delta=f"{absolute_diff:+.2f}" if abs(absolute_diff) > 0.01 else "0.00",
            delta_color=delta_color
        )
    
    with col2:
        st.metric("Target Value", f"{target_value:.2f}")
    
    with col3:
        # Untuk performance ratio, kita gunakan custom styling
        st.markdown(f"""
        <div style="background: {'#d4edda' if performance_ratio >= 100 else '#fff3cd' if performance_ratio >= 80 else '#f8d7da'}; 
                    padding: 10px; border-radius: 5px; border-left: 5px solid {'#28a745' if performance_ratio >= 100 else '#ffc107' if performance_ratio >= 80 else '#dc3545'};">
            <div style="font-size: 14px; color: #666;">Performance</div>
            <div style="font-size: 24px; font-weight: bold; color: #333;">{performance_ratio:.1f}%</div>
            <div style="font-size: 12px; color: #666;">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.metric("Data Points", f"{bullet_data['sample_size']:,}")
    
    # Progress bar visual
    progress_ratio = min(performance_ratio / 100, 1.0)
    st.progress(
        float(progress_ratio), 
        text=f"Achievement: {performance_ratio:.1f}% of target"
    )
    
    # Additional insights
    with st.expander("🔍 Additional Insights", expanded=False):
        col5, col6 = st.columns(2)
        
        with col5:
            # Variability analysis
            cv = (bullet_data['std_dev'] / current_value * 100) if current_value != 0 else 0
            st.metric("Coefficient of Variation", f"{cv:.1f}%")
            
            if cv < 10:
                st.info("✅ Low variability - consistent performance")
            elif cv < 30:
                st.warning("⚠️ Moderate variability")
            else:
                st.error("❌ High variability - inconsistent performance")
        
        with col6:
            # Target achievement confidence
            if bullet_data['std_dev'] > 0:
                z_score = (current_value - target_value) / bullet_data['std_dev']
                confidence = "High" if z_score > 1 else "Medium" if z_score > 0 else "Low"
                st.metric("Achievement Confidence", confidence)
            else:
                st.metric("Achievement Confidence", "N/A")

def show_bullet_optimization_info(data_size, sample_size, optimization_mode):
    """Tampilkan informasi optimasi"""
    
    reduction_pct = ((data_size - sample_size) / data_size) * 100 if data_size > 0 else 0
    
    if reduction_pct > 10:
        with st.expander("⚡ Info Optimasi Performa", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Original", f"{data_size:,}")
            with col2:
                st.metric("Data Diproses", f"{sample_size:,}")
            with col3:
                st.metric("Reduksi", f"{reduction_pct:.1f}%")
            
            optimization_strategies = {
                "Fast": "• ✅ **Aggressive sampling**\n• ✅ **Simple ranges**\n• ✅ **Minimal styling**",
                "Balanced": "• ✅ **Smart sampling**\n• ✅ **Optimized ranges**\n• ✅ **Enhanced visuals**",
                "Detailed": "• ✅ **Maximum data retention**\n• ✅ **Advanced analysis**\n• ✅ **Full features**"
            }
            
            st.info(f"**Mode {optimization_mode}**: {optimization_strategies.get(optimization_mode, 'Custom optimization')}")

def create_simple_bullet_fallback(df, value_col, target_col):
    """Fallback method untuk data yang bermasalah"""
    st.warning("Menggunakan metode fallback sederhana...")
    
    # Kalkulasi sederhana dengan sample kecil
    sample_data = df[[value_col] + ([target_col] if target_col else [])].dropna().head(1000)
    
    if len(sample_data) == 0:
        st.error("Tidak ada data valid")
        return
    
    current_value = sample_data[value_col].mean()
    target_value = sample_data[target_col].mean() if target_col else current_value * 1.1
    
    # Simple bullet chart
    fig = go.Figure()
    
    # Simple ranges
    ranges = [0, target_value * 0.5, target_value * 0.8, target_value * 1.2]
    colors = ['lightcoral', 'lightyellow', 'lightgreen']
    
    for i in range(3):
        fig.add_trace(go.Bar(
            x=[ranges[i+1] - ranges[i]],
            y=["Performance"],
            orientation='h',
            marker=dict(color=colors[i], opacity=0.3),
            base=ranges[i],
            showlegend=False
        ))
    
    fig.add_trace(go.Scatter(
        x=[target_value, target_value],
        y=["Performance", "Performance"],
        mode='lines',
        line=dict(color='red', width=2),
        name='Target'
    ))
    
    fig.add_trace(go.Scatter(
        x=[current_value],
        y=["Performance"],
        mode='markers',
        marker=dict(color='black', size=10, symbol='diamond'),
        name='Current'
    ))
    
    fig.update_layout(
        title=f"Simple Bullet: {value_col}",
        height=200,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tampilkan metrics sederhana
    performance_ratio = (current_value / target_value * 100) if target_value != 0 else 0
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current", f"{current_value:.2f}")
    with col2:
        st.metric("Target", f"{target_value:.2f}")
    with col3:
        st.metric("Performance", f"{performance_ratio:.1f}%")

# FUNGSI BARU: Treemap yang dioptimalkan
def create_treemap(df, numeric_cols, non_numeric_cols):
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hierarchy_1 = st.selectbox("Level hierarki 1", non_numeric_cols, key="tree_1")
    with col2:
        hierarchy_2 = st.selectbox("Level hierarki 2 (opsional)", 
                                 [None] + [col for col in non_numeric_cols if col != hierarchy_1], 
                                 key="tree_2")
    with col3:
        value_col = st.selectbox("Kolom nilai", numeric_cols, key="tree_value")
    
    if hierarchy_1 and value_col:
        # Optimasi: Gunakan cache untuk data yang besar
        @st.cache_data(ttl=3600)
        def aggregate_tree_data(_df, group_cols, value_column):
            return _df.groupby(group_cols)[value_column].sum().reset_index()
        
        # Tentukan kolom grouping
        group_cols = [hierarchy_1]
        if hierarchy_2 and hierarchy_2 != 'None':
            group_cols.append(hierarchy_2)
        
        # Agregasi data dengan cache
        with st.spinner("Memproses data..."):
            tree_data = aggregate_tree_data(df, group_cols, value_col)
        
        # Optimasi: Batasi jumlah kategori jika terlalu banyak
        max_categories = 50
        if len(tree_data) > max_categories:
            st.warning(f"⚠️ Data terlalu banyak ({len(tree_data)} kategori). Menampilkan {max_categories} kategori teratas.")
            
            # Ambil top categories berdasarkan value
            top_data = tree_data.nlargest(max_categories, value_col)
            other_data = tree_data.nsmallest(len(tree_data) - max_categories, value_col)
            
            # Gabungkan kategori kecil menjadi "Lainnya"
            if len(other_data) > 0:
                other_sum = other_data[value_col].sum()
                other_row = {group_cols[0]: "Lainnya", value_col: other_sum}
                if len(group_cols) > 1:
                    other_row[group_cols[1]] = "Lainnya"
                top_data = pd.concat([top_data, pd.DataFrame([other_row])], ignore_index=True)
            
            tree_data = top_data
        
        # Buat treemap
        fig = px.treemap(
            tree_data,
            path=group_cols,
            values=value_col,
            title=f"Treemap: {value_col} by {hierarchy_1}" + (f" and {hierarchy_2}" if hierarchy_2 and hierarchy_2 != 'None' else ""),
            color=value_col,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=600,
            margin=dict(t=50, l=25, r=25, b=25)
        )
        
        # Optimasi: Gunakan container width dengan config
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Tampilkan data summary dengan pagination
        with st.expander("📊 Data Summary"):
            st.write(f"Total Kategori: {len(tree_data)}")
            
            # Tambahkan filter untuk data summary
            col_a, col_b = st.columns(2)
            with col_a:
                min_value = st.number_input(
                    f"Minimum {value_col}", 
                    min_value=float(tree_data[value_col].min()), 
                    max_value=float(tree_data[value_col].max()),
                    value=float(tree_data[value_col].min()),
                    key="min_tree"
                )
            with col_b:
                sort_order = st.selectbox("Urutkan", ["Descending", "Ascending"], key="sort_tree")
            
            # Filter dan sort data
            filtered_data = tree_data[tree_data[value_col] >= min_value]
            filtered_data = filtered_data.sort_values(
                value_col, 
                ascending=(sort_order == "Ascending")
            )
            
            # Tampilkan dengan pagination
            page_size = 10
            total_pages = max(1, len(filtered_data) // page_size + (1 if len(filtered_data) % page_size > 0 else 0))
            
            page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, key="page_tree")
            
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, len(filtered_data))
            
            st.dataframe(
                filtered_data.iloc[start_idx:end_idx], 
                use_container_width=True,
                hide_index=True
            )
            
            st.write(f"Menampilkan {start_idx + 1}-{end_idx} dari {len(filtered_data)} baris")
            
            # Download option
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Data sebagai CSV",
                data=csv,
                file_name=f"treemap_data_{value_col}.csv",
                mime="text/csv"
            )

# Alternatif lebih sederhana untuk data sangat besar
def create_treemap_fast(df, numeric_cols, non_numeric_cols):
    
    # Pilihan kolom
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hierarchy_1 = st.selectbox("Level hierarki 1", non_numeric_cols, key="tree_fast_1")
    with col2:
        hierarchy_2 = st.selectbox("Level hierarki 2 (opsional)", 
                                 [None] + [col for col in non_numeric_cols if col != hierarchy_1], 
                                 key="tree_fast_2")
    with col3:
        value_col = st.selectbox("Kolom nilai", numeric_cols, key="tree_fast_value")
    
    if hierarchy_1 and value_col:
        # Sampling untuk data sangat besar
        if len(df) > 10000:
            st.info("🔍 Menggunakan sample data untuk performa lebih baik")
            sample_df = df.sample(n=10000, random_state=42)
        else:
            sample_df = df
        
        # Agregasi langsung tanpa cache (lebih cepat untuk data kecil)
        group_cols = [hierarchy_1]
        if hierarchy_2 and hierarchy_2 != 'None':
            group_cols.append(hierarchy_2)
        
        tree_data = sample_df.groupby(group_cols)[value_col].sum().reset_index()
        
        # Batasi kategori
        if len(tree_data) > 30:
            top_data = tree_data.nlargest(30, value_col)
            st.warning(f"Menampilkan 30 kategori teratas dari {len(tree_data)} total kategori")
            tree_data = top_data
        
        # Buat treemap sederhana
        fig = px.treemap(
            tree_data,
            path=group_cols,
            values=value_col,
            title=f"Treemap: {value_col} by {hierarchy_1}" + (f" and {hierarchy_2}" if hierarchy_2 and hierarchy_2 != 'None' else ""),
            color=value_col,
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_line_chart(df, numeric_cols, non_numeric_cols):
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("Pilih kolom untuk sumbu X", 
                           [df.index.name if df.index.name else "index"] + non_numeric_cols + numeric_cols, 
                           key="line_x_col")
    with col2:
        y_col = st.selectbox("Pilih kolom untuk sumbu Y", numeric_cols, key="line_y_col")
    
    # Optimasi: Pengaturan performa
    with st.expander("⚙️ Pengaturan Performa", expanded=False):
        col3, col4 = st.columns(2)
        with col3:
            max_points = st.slider("Maksimum titik data", 
                                 min_value=500, max_value=10000, value=2000, 
                                 key="line_max_points")
            use_sampling = st.checkbox("Gunakan sampling", value=True, key="line_sampling")
        with col4:
            aggregation = st.selectbox("Aggregasi data", 
                                     ["none", "mean", "sum", "max"], 
                                     key="line_aggregation")
            show_range_slider = st.checkbox("Tampilkan range slider", value=True, key="line_range_slider")
    
    if x_col and y_col:
        try:
            with st.spinner("Memproses data line chart..."):
                # Optimasi 1: Persiapan data dasar
                processed_df = df[[y_col]].copy()
                
                if x_col == "index":
                    x_data = df.index
                    x_label = "Index"
                    processed_df['x_axis'] = x_data
                else:
                    x_data = df[x_col]
                    x_label = x_col
                    processed_df['x_axis'] = x_data
                
                # Optimasi 2: Sampling untuk data besar
                if use_sampling and len(processed_df) > max_points:
                    if pd.api.types.is_datetime64_any_dtype(processed_df['x_axis']):
                        # Untuk time series, sampling terstruktur
                        processed_df = processed_df.sort_values('x_axis')
                        sample_frac = max_points / len(processed_df)
                        processed_df = processed_df.sample(frac=sample_frac, random_state=42)
                        processed_df = processed_df.sort_values('x_axis')
                    else:
                        # Untuk data non-time series, sampling sederhana
                        processed_df = processed_df.sample(n=max_points, random_state=42)
                    
                    st.info(f"📊 Data disampling: {len(processed_df):,} dari {len(df):,} titik data")
                
                # Optimasi 3: Aggregasi untuk data yang masih banyak
                if len(processed_df) > max_points and aggregation != "none":
                    if pd.api.types.is_datetime64_any_dtype(processed_df['x_axis']):
                        # Aggregasi time series
                        processed_df = processed_df.set_index('x_axis')
                        if aggregation == "mean":
                            processed_df = processed_df.resample('D').mean()
                        elif aggregation == "sum":
                            processed_df = processed_df.resample('D').sum()
                        elif aggregation == "max":
                            processed_df = processed_df.resample('D').max()
                        processed_df = processed_df.reset_index()
                        st.info(f"📈 Data diaggregasi per hari ({aggregation})")
                    else:
                        # Aggregasi non-time series
                        bins = min(1000, len(processed_df) // 10)
                        processed_df['x_bins'] = pd.cut(processed_df['x_axis'], bins=bins)
                        agg_df = processed_df.groupby('x_bins', observed=True).agg({
                            'x_axis': 'mean',
                            y_col: aggregation
                        }).reset_index()
                        processed_df = agg_df
                        st.info(f"📈 Data diaggregasi menjadi {bins} bin ({aggregation})")
                
                # Optimasi 4: Batasi titik data akhir
                if len(processed_df) > max_points:
                    processed_df = processed_df.head(max_points)
                    st.warning(f"⚠️ Data dibatasi hingga {max_points} titik pertama")
                
                # Optimasi 5: Cache figure creation
                @st.cache_data(ttl=300)
                def create_line_figure(data, x_col, y_col, title, is_datetime, show_slider):
                    fig = px.line(
                        data, 
                        x='x_axis', 
                        y=y_col, 
                        title=title,
                        line_shape='linear'  # Lebih cepat daripada 'spline'
                    )
                    
                    # Optimasi layout
                    layout_config = {
                        'height': 500,
                        'showlegend': False,
                        'margin': dict(l=50, r=50, t=60, b=80),
                        'plot_bgcolor': 'rgba(0,0,0,0)',
                        'xaxis': dict(
                            title=x_col,
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray'
                        ),
                        'yaxis': dict(
                            title=y_col,
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray'
                        )
                    }
                    
                    # Optimasi untuk time series
                    if is_datetime and show_slider:
                        layout_config['xaxis'].update({
                            'rangeslider': dict(visible=True, thickness=0.05),
                            'rangeselector': dict(
                                buttons=list([
                                    dict(count=1, label="1m", step="month", stepmode="backward"),
                                    dict(count=6, label="6m", step="month", stepmode="backward"),
                                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                                    dict(count=1, label="1y", step="year", stepmode="backward"),
                                    dict(step="all")
                                ])
                            )
                        })
                    
                    fig.update_layout(**layout_config)
                    
                    # Optimasi trace untuk performa
                    fig.update_traces(
                        hovertemplate=f'<b>{y_col}</b><br>{x_col}: %{{x}}<br>Nilai: %{{y:.2f}}<extra></extra>',
                        line=dict(width=1.2),
                        connectgaps=False  # Lebih cepat
                    )
                    
                    return fig
                
                # Deteksi tipe data
                is_datetime = pd.api.types.is_datetime64_any_dtype(processed_df['x_axis'])
                
                title = f"Grafik Garis: {y_col} over {x_label}"
                fig = create_line_figure(processed_df, x_label, y_col, title, is_datetime, show_range_slider)
                
                # Optimasi 6: Plotly config yang ringan
                config = {
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestGl2d'],
                    'scrollZoom': True,
                    'responsive': True
                }
                
                st.plotly_chart(fig, use_container_width=True, config=config)
            
            # Tampilkan statistik
            with st.expander("📊 Statistik Data"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Titik Data", len(processed_df))
                with col2:
                    st.metric(f"Rata-rata {y_col}", f"{processed_df[y_col].mean():.2f}")
                with col3:
                    st.metric("Rentang Waktu" if is_datetime else "Rentang Nilai", 
                             f"{len(processed_df['x_axis'].unique())} titik")
                
                st.dataframe(processed_df[['x_axis', y_col]].describe(), use_container_width=True)
                
        except ValueError as e:
            if "date" in str(e).lower():
                st.warning("Format tanggal tidak dikenali. Coba konversi kolom tanggal ke format datetime terlebih dahulu.")
            else:
                st.error(f"Error nilai: {str(e)}")
        except KeyError as e:
            st.error(f"Kolom tidak ditemukan: {str(e)}")
        except Exception as e:
            st.error(f"Error membuat line chart: {str(e)}")
            st.info("Tips: Pastikan data sumbu X dan Y valid dan tidak mengandung nilai NaN")

# Alternatif: Versi ultra-ringan untuk data sangat besar
def create_line_chart_lightweight(df, numeric_cols, non_numeric_cols):
    """Versi yang lebih ringan untuk dataset sangat besar (>100k records)"""
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("Pilih kolom untuk sumbu X", 
                           [df.index.name if df.index.name else "index"] + non_numeric_cols + numeric_cols, 
                           key="line_x_light")
    with col2:
        y_col = st.selectbox("Pilih kolom untuk sumbu Y", numeric_cols, key="line_y_light")
    
    if x_col and y_col:
        try:
            # Aggregasi langsung untuk performa maksimal
            if x_col == "index":
                x_data = df.index
            else:
                x_data = df[x_col]
            
            # Resample otomatis untuk data besar
            sample_df = df[[y_col]].copy()
            sample_df['x_axis'] = x_data
            
            if len(sample_df) > 1000:
                if pd.api.types.is_datetime64_any_dtype(sample_df['x_axis']):
                    sample_df = sample_df.set_index('x_axis')
                    sample_df = sample_df.resample('H').mean().head(1000)
                    sample_df = sample_df.reset_index()
                else:
                    sample_df = sample_df.sample(n=1000, random_state=42)
            
            # Plot sederhana
            fig = px.line(sample_df, x='x_axis', y=y_col, 
                         title=f"Grafik Garis: {y_col} (Data: {len(sample_df):,} titik)")
            
            fig.update_layout(height=400, margin=dict(l=50, r=50, t=50, b=80))
            
            st.plotly_chart(fig, use_container_width=True, 
                           config={'displayModeBar': False})
            
        except Exception as e:
            st.error(f"Error membuat chart: {str(e)}")

def create_bar_chart(df, numeric_cols, non_numeric_cols):
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("Pilih kolom untuk sumbu X", non_numeric_cols if non_numeric_cols else numeric_cols, 
                           key="bar_x_col")
    with col2:
        y_col = st.selectbox("Pilih kolom untuk sumbu Y", numeric_cols, key="bar_y_col")
    
    # Optimasi: Pengaturan performa
    col3, col4 = st.columns(2)
    with col3:
        max_categories = st.slider("Maksimum kategori ditampilkan", 
                                 min_value=5, max_value=50, value=20, key="bar_max_categories")
    with col4:
        sort_data = st.checkbox("Urutkan data", value=True, key="bar_sort")
        use_sampling = st.checkbox("Gunakan sampling untuk data besar", value=True, key="bar_sampling")
    
    if x_col and y_col:
        with st.spinner("Memproses data..."):
            # Optimasi 1: Sampling untuk data besar
            processed_df = df.copy()
            if use_sampling and len(df) > 10000:
                sample_size = min(10000, len(df))
                processed_df = df.sample(n=sample_size, random_state=42)
                st.info(f"📊 Data disampling: {sample_size:,} dari {len(df):,} records")
            
            # Optimasi 2: Aggregasi yang efisien
            if x_col in non_numeric_cols:
                # Untuk data kategorikal, gunakan observed=True dan batasi kategori
                bar_data = (processed_df.groupby(x_col, observed=True)[y_col]
                          .agg(['mean', 'count'])
                          .round(2)
                          .nlargest(max_categories, 'count')
                          .reset_index())
                bar_data.columns = [x_col, y_col, 'count']
            else:
                # Untuk data numerik, buat bins
                bar_data = (processed_df.groupby(pd.cut(processed_df[x_col], bins=min(20, len(processed_df[x_col].unique())), 
                                                      include_lowest=True))[y_col]
                          .mean()
                          .reset_index())
                bar_data.columns = [x_col, y_col]
            
            # Optimasi 3: Sorting jika diperlukan
            if sort_data:
                bar_data = bar_data.sort_values(y_col, ascending=False)
            
            # Optimasi 4: Cache figure creation
            @st.cache_data(ttl=300)
            def create_bar_figure(data, x_col, y_col, title):
                fig = px.bar(
                    data, 
                    x=x_col, 
                    y=y_col, 
                    title=title,
                    color=y_col,
                    color_continuous_scale='blues'
                )
                
                # Optimasi layout untuk performa
                fig.update_layout(
                    height=500,
                    showlegend=False,
                    margin=dict(l=50, r=50, t=60, b=100),
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # Optimasi hover data
                fig.update_traces(
                    hovertemplate=f"<b>%{{x}}</b><br>{y_col}: %{{y:.2f}}<extra></extra>"
                )
                
                return fig
            
            title = f"Grafik Batang: Rata-rata {y_col} per {x_col}"
            fig = create_bar_figure(bar_data, x_col, y_col, title)
            
            # Optimasi 5: Plotly config yang ringan
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
                'responsive': True
            }
            
            st.plotly_chart(fig, use_container_width=True, config=config)
        
        # Tampilkan data summary
        with st.expander("📊 Lihat Data Summary"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Kategori", len(bar_data))
            with col2:
                st.metric(f"Rata-rata {y_col}", f"{bar_data[y_col].mean():.2f}")
            with col3:
                st.metric("Kategori Tertinggi", bar_data.iloc[0][x_col][:20] + "...")
            
            st.dataframe(bar_data.head(10).style.format({y_col: "{:.2f}"}), use_container_width=True)
            
        with st.expander("ℹ️ Keterangan Grafik Batang"):
            st.markdown(f"""
            **Grafik Batang (Bar Chart)** digunakan untuk membandingkan nilai antar kategori.
            
            **Statistik Dataset:**
            - Total kategori yang ditampilkan: **{len(bar_data)}**
            - Rentang nilai: **{bar_data[y_col].min():.2f}** hingga **{bar_data[y_col].max():.2f}**
            - Standar deviasi: **{bar_data[y_col].std():.2f}**
            
            **Kelebihan**: 
            - Mudah membandingkan nilai antar kategori
            - Visualisasi yang intuitif
            
            **Kekurangan**: 
            - Tidak efektif untuk data dengan banyak kategori
            - Dapat menjadi lambat dengan data sangat besar
            
            **Penggunaan**: Perbandingan kategori, ranking, distribusi kategorikal
            
            **Optimasi yang diterapkan:**
            ✅ Sampling otomatis untuk data besar  
            ✅ Batasan jumlah kategori  
            ✅ Caching untuk performa  
            ✅ Aggregasi yang efisien  
            """)

# Alternatif: Versi ultra-ringan untuk data sangat besar
def create_bar_chart_lightweight(df, numeric_cols, non_numeric_cols):
    """Versi yang lebih ringan untuk dataset sangat besar"""
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("Pilih kolom untuk sumbu X", non_numeric_cols if non_numeric_cols else numeric_cols, 
                           key="bar_x_light")
    with col2:
        y_col = st.selectbox("Pilih kolom untuk sumbu Y", numeric_cols, key="bar_y_light")
    
    if x_col and y_col:
        # Aggregasi langsung tanpa sampling tambahan
        bar_data = (df.groupby(x_col, observed=True)[y_col]
                  .mean()
                  .nlargest(15)
                  .reset_index())
        
        # Plot sederhana
        fig = px.bar(bar_data, x=x_col, y=y_col, 
                    title=f"Grafik Batang: Top 15 {y_col} per {x_col}")
        
        fig.update_layout(height=400, margin=dict(l=50, r=50, t=50, b=100))
        fig.update_xaxes(tickangle=-45)
        
        st.plotly_chart(fig, use_container_width=True, 
                       config={'displayModeBar': False})

def create_combined_chart(df, numeric_cols, non_numeric_cols):
    
    # Deteksi ukuran data dan berikan rekomendasi
    data_size = len(df)
    if data_size > 1000000:
        st.warning(f"🚨 Data sangat besar ({data_size:,} rows). Menggunakan mode ultra-fast...")
        default_optimization = "Super Fast"
    elif data_size > 100000:
        st.info(f"📊 Data besar ({data_size:,} rows). Optimasi otomatis diaktifkan.")
        default_optimization = "Fast"
    else:
        default_optimization = "Balanced"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_col = st.selectbox("Pilih kolom untuk sumbu X", 
                           [df.index.name if df.index.name else "index"] + non_numeric_cols + numeric_cols, 
                           key="comb_x_col")
    
    with col2:
        line_col = st.selectbox("Pilih kolom untuk garis", numeric_cols, key="line_col")
    
    with col3:
        bar_col = st.selectbox("Pilih kolom untuk batang", numeric_cols, key="bar_col")
    
    with col4:
        optimization_level = st.selectbox("Level Optimasi", 
                                        ["Super Fast", "Fast", "Balanced"],
                                        index=0 if data_size > 100000 else 1,
                                        key="optim_level")
    
    if x_col and line_col and bar_col:
        try:
            with st.spinner("🔄 Mengoptimalkan data besar..."):
                # OPTIMASI UTAMA UNTUK DATA BESAR
                display_df = optimize_dataframe_advanced(df, x_col, line_col, bar_col, optimization_level, data_size)
                
                # Buat chart dengan konfigurasi performa tinggi
                fig = create_ultra_optimized_chart(display_df, x_col, line_col, bar_col, optimization_level, data_size)
            
            # Konfigurasi plotly yang sangat ringan
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestGl2d', 'toggleSpikelines'],
                'scrollZoom': True,
                'responsive': True
            }
            
            st.plotly_chart(fig, use_container_width=True, config=config)
            
            # Tampilkan info optimasi
            show_optimization_info(df, display_df, optimization_level)
            
        except Exception as e:
            create_fallback_chart(df, x_col, line_col, bar_col)

def optimize_dataframe_advanced(df, x_col, line_col, bar_col, optimization_level, original_size):
    """
    Fungsi optimasi agresif dengan multiple strategies
    """
    # Target sizes berdasarkan level optimasi
    target_sizes = {
        "Super Fast": min(500, original_size),
        "Fast": min(2000, original_size),
        "Balanced": min(10000, original_size)
    }
    
    target_size = target_sizes[optimization_level]
    
    # Jika data sudah kecil, return langsung
    if original_size <= target_size:
        return df[[x_col, line_col, bar_col]].copy().dropna()
    
    # Pilih strategi berdasarkan tipe data dan ukuran
    sample_df = select_sampling_strategy(df, x_col, line_col, bar_col, target_size, original_size)
    
    return sample_df.dropna()

def select_sampling_strategy(df, x_col, line_col, bar_col, target_size, original_size):
    """
    Pilih strategi sampling terbaik berdasarkan karakteristik data
    """
    sample_df = df[[x_col, line_col, bar_col]].copy()
    
    # Strategy 1: Time series data
    if pd.api.types.is_datetime64_any_dtype(sample_df[x_col]):
        return time_series_sampling(sample_df, x_col, target_size, original_size)
    
    # Strategy 2: Numeric X data
    elif pd.api.types.is_numeric_dtype(sample_df[x_col]):
        return numeric_sampling(sample_df, x_col, target_size, original_size)
    
    # Strategy 3: Categorical data
    else:
        return categorical_sampling(sample_df, x_col, target_size, original_size)

def time_series_sampling(df, x_col, target_size, original_size):
    """Optimasi untuk data time series"""
    df_sorted = df.sort_values(x_col).set_index(x_col)
    
    # Tentukan frekuensi resampling berdasarkan target size
    if original_size > 1000000:
        freq = '1H'  # 1 hour untuk data sangat besar
    elif original_size > 100000:
        freq = '30T'  # 30 menit
    elif original_size > 50000:
        freq = '10T'  # 10 menit
    else:
        freq = '1T'   # 1 menit
    
    try:
        resampled = df_sorted.resample(freq).mean().reset_index()
        # Jika masih terlalu besar, ambil sample
        if len(resampled) > target_size:
            step = max(1, len(resampled) // target_size)
            resampled = resampled.iloc[::step]
        return resampled
    except:
        # Fallback: systematic sampling
        step = max(1, original_size // target_size)
        return df.iloc[::step].copy()

def numeric_sampling(df, x_col, target_size, original_size):
    """Optimasi untuk data numerik"""
    # Adaptive binning berdasarkan ukuran data
    n_bins = min(target_size, 500)
    
    # Buat bins yang adaptif
    df_clean = df.dropna()
    if len(df_clean) == 0:
        return df.head(target_size)
    
    try:
        bins = pd.cut(df_clean[x_col], bins=n_bins, duplicates='drop')
        aggregated = df_clean.groupby(bins, observed=False).agg({
            x_col: 'mean',
            **{col: 'mean' for col in df_clean.columns if col != x_col}
        }).reset_index(drop=True)
        
        # Pastikan tidak melebihi target size
        return aggregated.head(target_size)
    except:
        # Fallback: random sampling
        return df_clean.sample(n=min(target_size, len(df_clean)), random_state=42)

def categorical_sampling(df, x_col, target_size, original_size):
    """Optimasi untuk data kategorikal"""
    # Ambil kategori paling meaningful (dengan data terbanyak)
    value_counts = df[x_col].value_counts()
    
    if len(value_counts) > target_size:
        # Ambil top categories
        top_categories = value_counts.head(target_size).index
        sampled = df[df[x_col].isin(top_categories)]
    else:
        # Jika kategori tidak terlalu banyak, sampling random
        sampled = df.sample(n=min(target_size, len(df)), random_state=42)
    
    return sampled

def create_ultra_optimized_chart(display_df, x_col, line_col, bar_col, optimization_level, original_size):
    """
    Buat chart dengan optimasi maksimal untuk performa
    """
    fig = go.Figure()
    
    # Konfigurasi berdasarkan level optimasi dan ukuran data
    config = get_render_config(optimization_level, len(display_df), original_size)
    
    # Trace untuk garis - selalu gunakan scattergl untuk WebGL
    fig.add_trace(go.Scattergl(
        x=display_df[x_col],
        y=display_df[line_col],
        mode=config['line_mode'],
        name=line_col,
        yaxis='y1',
        line=dict(
            color='#1f77b4', 
            width=config['line_width'],
            simplify=True  # Optimasi: simplify line
        ),
        marker=dict(size=config['marker_size']),
        hovertemplate=config['hover_template'].format(x_col=x_col, col=line_col),
        opacity=0.9
    ))
    
    # Trace untuk batang - hanya untuk data tidak terlalu banyak
    if len(display_df) <= 1000 or optimization_level == "Balanced":
        fig.add_trace(go.Bar(
            x=display_df[x_col],
            y=display_df[bar_col],
            name=bar_col,
            yaxis='y2',
            marker=dict(
                color='#ff7f0e', 
                opacity=config['bar_opacity'],
                line=dict(width=0)  # No border untuk performa
            ),
            hovertemplate=config['hover_template'].format(x_col=x_col, col=bar_col),
            opacity=config['bar_opacity']
        ))
    else:
        # Untuk data banyak, gunakan line juga untuk bar data
        fig.add_trace(go.Scattergl(
            x=display_df[x_col],
            y=display_df[bar_col],
            name=f"{bar_col} (line)",
            yaxis='y2',
            line=dict(color='#ff7f0e', width=2, dash='dot'),
            hovertemplate=config['hover_template'].format(x_col=x_col, col=bar_col),
            opacity=0.7
        ))
    
    # Layout ultra-optimized
    layout_config = {
        'title': f"Kombinasi: {line_col} vs {bar_col} - {optimization_level} Mode",
        'xaxis': {
            'title': x_col,
            'tickangle': -45 if len(display_df) > 20 else 0,
            'gridcolor': '#f0f0f0',
            'showgrid': True,
        },
        'yaxis': {
            'title': line_col, 
            'side': 'left',
            'gridcolor': '#f0f0f0',
        },
        'yaxis2': {
            'title': bar_col, 
            'side': 'right', 
            'overlaying': 'y',
            'gridcolor': '#f0f0f0',
        },
        'height': 450,
        'showlegend': config['show_legend'],
        'margin': dict(l=60, r=60, t=60, b=80),
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        # Optimasi performa critical
        'uirevision': 'constant',
    }
    
    # Tambahkan range slider hanya untuk time series dengan data cukup
    if (pd.api.types.is_datetime64_any_dtype(display_df[x_col]) and 
        len(display_df) > 100):
        layout_config['xaxis']['rangeslider'] = {
            'visible': True, 
            'thickness': 0.03,
            'bgcolor': '#f8f9fa'
        }
    
    fig.update_layout(**layout_config)
    
    return fig

def get_render_config(optimization_level, display_size, original_size):
    """Konfigurasi rendering berdasarkan level optimasi"""
    configs = {
        "Super Fast": {
            'line_mode': 'lines',
            'marker_size': 0,
            'line_width': 1,
            'bar_opacity': 0.6,
            'show_legend': False,
            'hover_template': '<b>%{x}</b><br>%{y:.1f}<extra></extra>'
        },
        "Fast": {
            'line_mode': 'lines',
            'marker_size': 0,
            'line_width': 1.5,
            'bar_opacity': 0.7,
            'show_legend': True,
            'hover_template': '<b>%{x}</b><br>%{y:.2f}<extra></extra>'
        }
    }
    return configs[optimization_level]

def show_optimization_info(original_df, optimized_df, optimization_level):
    """Tampilkan informasi optimasi"""
    reduction_pct = ((len(original_df) - len(optimized_df)) / len(original_df)) * 100
    
    with st.expander("📊 Info Optimasi Performa", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Original", f"{len(original_df):,}")
        with col2:
            st.metric("Data Ditampilkan", f"{len(optimized_df):,}")
        with col3:
            st.metric("Reduksi", f"{reduction_pct:.1f}%")
        with col4:
            st.metric("Optimasi", optimization_level)
        
        # Progress bar untuk visualisasi reduksi
        st.progress(min(100, 100 - reduction_pct)/100, text="Tingkat Reduksi Data")
        
        st.markdown("""
        **Strategi Optimasi:**
        - ✅ **WebGL Rendering** untuk performa terbaik
        - ✅ **Adaptive Sampling** berdasarkan tipe data
        - ✅ **Smart Aggregation** untuk mempertahankan pattern
        - ✅ **Minimal Hover Effects** untuk rendering cepat
        """)

def create_fallback_chart(df, x_col, line_col, bar_col):
    """Fallback method untuk data yang bermasalah"""
    st.warning("Menggunakan metode fallback...")
    
    # Sample kecil untuk memastikan bisa render
    sample_df = df[[x_col, line_col, bar_col]].dropna().head(1000)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_df[x_col], y=sample_df[line_col], 
        mode='lines', name=line_col, yaxis='y1'
    ))
    fig.add_trace(go.Bar(
        x=sample_df[x_col], y=sample_df[bar_col], 
        name=bar_col, yaxis='y2', opacity=0.6
    ))
    
    fig.update_layout(
        title=f"Fallback Chart: {line_col} vs {bar_col}",
        yaxis=dict(title=line_col, side='left'),
        yaxis2=dict(title=bar_col, side='right', overlaying='y'),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Versi ultra-ringan untuk data ekstrem (>1 juta rows)
def create_combined_chart_ultralight(df, numeric_cols, non_numeric_cols):
    """Versi ultra-ringan untuk data > 1 juta rows"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_col = st.selectbox("Sumbu X", 
                           [df.index.name if df.index.name else "index"] + non_numeric_cols[:3], 
                           key="ultralight_x")
    with col2:
        line_col = st.selectbox("Garis", numeric_cols[:5], key="ultralight_line")
    with col3:
        bar_col = st.selectbox("Batang", numeric_cols[:5], key="ultralight_bar")
    
    if x_col and line_col and bar_col:
        # Aggressive sampling - hanya 500 data points
        if len(df) > 1000:
            step = len(df) // 500
            display_df = df.iloc[::step][[x_col, line_col, bar_col]].dropna()
        else:
            display_df = df[[x_col, line_col, bar_col]].dropna()
        
        # Simple chart dengan WebGL
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=display_df[x_col], y=display_df[line_col],
            mode='lines', name=line_col, line=dict(width=1)
        ))
        fig.add_trace(go.Scattergl(
            x=display_df[x_col], y=display_df[bar_col],
            mode='lines', name=bar_col, line=dict(width=1, dash='dot'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            height=400,
            yaxis2=dict(side='right', overlaying='y'),
            showlegend=True,
            margin=dict(l=50, r=50, t=30, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.info(f"Ultra-Light Mode: {len(display_df):,} points dari {len(df):,}")

def create_scatter_plot(df, numeric_cols, non_numeric_cols):
    
    # Deteksi ukuran data
    data_size = len(df)
    if data_size > 100000:
        st.info(f"⚡ Mode Optimasi: Data besar ({data_size:,} rows) - Menggunakan sampling otomatis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_col = st.selectbox("Pilih kolom X", numeric_cols, key="scatter_x")
    with col2:
        y_col = st.selectbox("Pilih kolom Y", numeric_cols, key="scatter_y")
    with col3:
        color_col = st.selectbox("Pilih kolom warna", [None] + non_numeric_cols, key="scatter_color")
    with col4:
        # Pengaturan optimasi
        optimization_mode = st.selectbox(
            "Mode Optimasi",
            ["Auto", "Fast", "Balanced", "Detailed"],
            index=0 if data_size > 100000 else 2,
            key="scatter_optim"
        )
    
    # Opsi tambahan untuk data besar
    with st.expander("⚙️ Pengaturan Lanjutan", expanded=False):
        col5, col6, col7 = st.columns(3)
        with col5:
            max_points = st.slider(
                "Maksimum titik data",
                min_value=1000,
                max_value=20000,
                value=5000 if data_size > 100000 else 10000,
                key="scatter_max_points"
            )
        with col6:
            point_size = st.slider(
                "Ukuran titik",
                min_value=1,
                max_value=10,
                value=3 if data_size > 50000 else 5,
                key="scatter_point_size"
            )
        with col7:
            opacity = st.slider(
                "Transparansi",
                min_value=0.1,
                max_value=1.0,
                value=0.6 if data_size > 50000 else 0.8,
                key="scatter_opacity"
            )
    
    if x_col and y_col:
        try:
            with st.spinner("🔄 Memproses data scatter plot..."):
                # OPTIMASI 1: Filter data dan sampling
                plot_data = optimize_scatter_data(df, x_col, y_col, color_col, data_size, optimization_mode, max_points)
                
                if len(plot_data) == 0:
                    st.warning("Tidak ada data valid untuk plot")
                    return
                
                # OPTIMASI 2: Buat scatter plot yang dioptimalkan
                fig = create_optimized_scatter(plot_data, x_col, y_col, color_col, point_size, opacity, data_size)
                
                # OPTIMASI 3: Konfigurasi plotly yang ringan
                config = {
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestGl2d'],
                    'scrollZoom': True,
                    'responsive': True
                }
                
                st.plotly_chart(fig, use_container_width=True, config=config)
                
                # Tampilkan statistik korelasi
                display_correlation_stats(plot_data, x_col, y_col)
                
                # Tampilkan info optimasi
                show_scatter_optimization_info(data_size, len(plot_data), optimization_mode)
                
        except Exception as e:
            st.error(f"Error membuat scatter plot: {str(e)}")
            # Fallback ke metode sederhana
            create_simple_scatter_fallback(df, x_col, y_col, color_col)

def optimize_scatter_data(df, x_col, y_col, color_col, data_size, optimization_mode, max_points):
    """Optimasi data untuk scatter plot dengan sampling yang tepat"""
    # Pilih kolom yang diperlukan
    columns_needed = [x_col, y_col]
    if color_col:
        columns_needed.append(color_col)
    
    # Filter data yang valid
    plot_data = df[columns_needed].dropna()
    
    if len(plot_data) == 0:
        return plot_data
    
    # Tentukan target sample size
    target_sizes = {
        "Auto": min(max_points, data_size),
        "Fast": min(3000, data_size),
        "Balanced": min(10000, data_size),
        "Detailed": min(20000, data_size)
    }
    
    target_size = target_sizes[optimization_mode]
    
    # Jika data lebih besar dari target, lakukan sampling
    if len(plot_data) > target_size:
        if optimization_mode == "Fast":
            # Systematic sampling untuk performa maksimal
            step = len(plot_data) // target_size
            sampled_data = plot_data.iloc[::step]
        elif optimization_mode == "Balanced":
            # Stratified sampling berdasarkan quadrant
            try:
                x_median = plot_data[x_col].median()
                y_median = plot_data[y_col].median()
                
                # Bagi data menjadi 4 quadrant
                quadrants = [
                    (plot_data[x_col] <= x_median) & (plot_data[y_col] <= y_median),
                    (plot_data[x_col] <= x_median) & (plot_data[y_col] > y_median),
                    (plot_data[x_col] > x_median) & (plot_data[y_col] <= y_median),
                    (plot_data[x_col] > x_median) & (plot_data[y_col] > y_median)
                ]
                
                samples_per_quadrant = target_size // 4
                sampled_dfs = []
                
                for quadrant in quadrants:
                    quadrant_data = plot_data[quadrant]
                    if len(quadrant_data) > 0:
                        sample_size = min(samples_per_quadrant, len(quadrant_data))
                        sampled_dfs.append(quadrant_data.sample(n=sample_size, random_state=42))
                
                sampled_data = pd.concat(sampled_dfs, ignore_index=True)
                
                # Jika masih kurang, tambahkan random sampling
                if len(sampled_data) < target_size:
                    remaining = target_size - len(sampled_data)
                    additional_samples = plot_data.sample(n=remaining, random_state=42)
                    sampled_data = pd.concat([sampled_data, additional_samples], ignore_index=True)
                    
            except:
                # Fallback ke random sampling
                sampled_data = plot_data.sample(n=target_size, random_state=42)
        else:
            # Random sampling untuk mode lain
            sampled_data = plot_data.sample(n=target_size, random_state=42)
        
        return sampled_data
    
    return plot_data

def create_optimized_scatter(plot_data, x_col, y_col, color_col, point_size, opacity, original_size):
    """Buat scatter plot dengan optimasi performa"""
    
    # OPTIMASI: Gunakan scattergl (WebGL) untuk data banyak
    if len(plot_data) > 5000:
        scatter_function = px.scatter_gl
        scatter_name = "ScatterGL"
    else:
        scatter_function = px.scatter
        scatter_name = "Scatter"
    
    # Buat plot
    fig = scatter_function(
        plot_data,
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"{scatter_name}: {y_col} vs {x_col} ({len(plot_data):,} points)",
        opacity=opacity,
        size_max=point_size * 2
    )
    
    # Update marker size dan style
    fig.update_traces(
        marker=dict(
            size=point_size,
            line=dict(width=0)  # No border untuk performa
        ),
        selector=dict(mode='markers')
    )
    
    # OPTIMASI: Sederhanakan hover template untuk performa
    if len(plot_data) > 5000:
        hover_template = f'{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>'
    else:
        if color_col:
            hover_template = f'{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<br>{color_col}: %{{marker.color}}<extra></extra>'
        else:
            hover_template = f'{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>'
    
    fig.update_traces(hovertemplate=hover_template)
    
    # Layout yang dioptimalkan
    fig.update_layout(
        height=500,
        showlegend=len(plot_data) <= 10000,  # Sembunyikan legend untuk data sangat banyak
        margin=dict(l=50, r=50, t=60, b=50),
        plot_bgcolor='white'
    )
    
    # Tambahkan trendline untuk data yang tidak terlalu banyak
    if len(plot_data) <= 10000 and len(plot_data) > 10:
        try:
            # Hitung regression line
            z = np.polyfit(plot_data[x_col], plot_data[y_col], 1)
            p = np.poly1d(z)
            
            # Buat trendline
            x_trend = np.linspace(plot_data[x_col].min(), plot_data[x_col].max(), 100)
            y_trend = p(x_trend)
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Trend Line',
                hovertemplate='Trend: %{y:.2f}<extra></extra>'
            ))
        except:
            pass  # Skip trendline jika error
    
    return fig

def display_correlation_stats(plot_data, x_col, y_col):
    """Tampilkan statistik korelasi"""
    with st.expander("📈 Analisis Korelasi", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        try:
            # Hitung korelasi
            correlation = plot_data[x_col].corr(plot_data[y_col])
            
            with col1:
                st.metric("Korelasi Pearson", f"{correlation:.3f}")
            
            with col2:
                # Interpretasi korelasi
                if abs(correlation) < 0.3:
                    st.metric("Kekuatan", "Lemah")
                elif abs(correlation) < 0.7:
                    st.metric("Kekuatan", "Sedang")
                else:
                    st.metric("Kekuatan", "Kuat")
            
            with col3:
                # Arah korelasi
                if correlation > 0:
                    st.metric("Arah", "Positif")
                else:
                    st.metric("Arah", "Negatif")
            
            # Additional stats
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Jumlah Titik", len(plot_data))
            with col5:
                st.metric(f"Rata2 {x_col}", f"{plot_data[x_col].mean():.2f}")
            with col6:
                st.metric(f"Rata2 {y_col}", f"{plot_data[y_col].mean():.2f}")
                
        except Exception as e:
            st.warning(f"Tidak dapat menghitung korelasi: {str(e)}")

def show_scatter_optimization_info(original_size, processed_size, optimization_mode):
    """Tampilkan informasi optimasi"""
    reduction_pct = ((original_size - processed_size) / original_size) * 100 if original_size > 0 else 0
    
    if reduction_pct > 10:
        with st.expander("⚡ Info Optimasi Performa", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Original", f"{original_size:,}")
            with col2:
                st.metric("Data Ditampilkan", f"{processed_size:,}")
            with col3:
                st.metric("Reduksi", f"{reduction_pct:.1f}%")
            
            st.info(f"**Mode {optimization_mode}**: Scatter plot dioptimalkan untuk kecepatan rendering")
            
            if optimization_mode == "Fast":
                st.markdown("• ✅ **WebGL Rendering** (ScatterGL)")
                st.markdown("• ✅ **Systematic Sampling**")
                st.markdown("• ✅ **Minimal Hover Effects**")
            elif optimization_mode == "Balanced":
                st.markdown("• ✅ **Stratified Sampling** (per quadrant)")
                st.markdown("• ✅ **Trend Line Analysis**")
                st.markdown("• ✅ **Optimized Hover**")

def create_simple_scatter_fallback(df, x_col, y_col, color_col):
    """Fallback method untuk data yang bermasalah"""
    st.warning("Menggunakan metode fallback yang sederhana...")
    
    # Sample kecil untuk memastikan bisa render
    sample_size = min(1000, len(df))
    plot_data = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna().head(sample_size)
    
    fig = px.scatter(
        plot_data,
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"Simple Scatter: {y_col} vs {x_col} ({len(plot_data)} points)"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Versi ultra-ringan untuk data ekstrem
def create_ultra_fast_scatter(df, numeric_cols, non_numeric_cols):
    """Versi ultra-ringan untuk data > 500k rows"""
    
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Pilih kolom X", numeric_cols[:8], key="ultra_scatter_x")
    with col2:
        y_col = st.selectbox("Pilih kolom Y", numeric_cols[:8], key="ultra_scatter_y")
    
    if x_col and y_col:
        # Sampling agresif - hanya 2000 points
        if len(df) > 2000:
            plot_data = df[[x_col, y_col]].dropna().sample(n=2000, random_state=42)
        else:
            plot_data = df[[x_col, y_col]].dropna()
        
        # ScatterGL dengan konfigurasi minimal
        fig = px.scatter_gl(
            plot_data,
            x=x_col,
            y=y_col,
            title=f"Ultra-Fast: {y_col} vs {x_col} (2,000 samples)"
        )
        
        fig.update_traces(
            marker=dict(size=2, opacity=0.5),
            hovertemplate='%{x:.1f}, %{y:.1f}<extra></extra>'
        )
        
        fig.update_layout(height=350, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.info(f"📊 Menampilkan 2,000 sample dari {len(df[[x_col, y_col]].dropna()):,} data points")

def create_bubble_chart(df, numeric_cols, non_numeric_cols):
    
    # Deteksi ukuran data
    data_size = len(df)
    if data_size > 50000:
        st.info(f"⚡ Mode Optimasi: Data besar ({data_size:,} rows) - Menggunakan sampling otomatis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_col = st.selectbox("Pilih kolom X", numeric_cols, key="bubble_x")
    with col2:
        y_col = st.selectbox("Pilih kolom Y", numeric_cols, key="bubble_y")
    with col3:
        size_col = st.selectbox("Pilih kolom ukuran", numeric_cols, key="bubble_size")
    with col4:
        color_col = st.selectbox("Pilih kolom warna", [None] + non_numeric_cols, key="bubble_color")
    
    # Pengaturan optimasi
    with st.expander("⚙️ Pengaturan Optimasi", expanded=False):
        col5, col6, col7 = st.columns(3)
        with col5:
            optimization_mode = st.selectbox(
                "Mode Optimasi",
                ["Auto", "Fast", "Balanced", "Detailed"],
                index=0 if data_size > 50000 else 2,
                key="bubble_optim"
            )
        with col6:
            max_bubbles = st.slider(
                "Maksimum gelembung",
                min_value=100,
                max_value=2000,
                value=500 if data_size > 50000 else 1000,
                key="bubble_max_points"
            )
        with col7:
            size_factor = st.slider(
                "Faktor ukuran gelembung",
                min_value=1,
                max_value=20,
                value=5 if data_size > 50000 else 10,
                key="bubble_size_factor"
            )
    
    if x_col and y_col and size_col:
        try:
            with st.spinner("🔄 Memproses data bubble chart..."):
                # OPTIMASI 1: Filter data dan sampling
                plot_data = optimize_bubble_data(df, x_col, y_col, size_col, color_col, data_size, optimization_mode, max_bubbles)
                
                if len(plot_data) == 0:
                    st.warning("Tidak ada data valid untuk bubble chart")
                    return
                
                # OPTIMASI 2: Normalisasi ukuran bubble untuk visualisasi yang lebih baik
                plot_data = normalize_bubble_sizes(plot_data, size_col, size_factor)
                
                # OPTIMASI 3: Buat bubble chart yang dioptimalkan
                fig = create_optimized_bubble_chart(plot_data, x_col, y_col, size_col, color_col, data_size)
                
                # OPTIMASI 4: Konfigurasi plotly yang ringan
                config = {
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestGl2d'],
                    'scrollZoom': True,
                    'responsive': True
                }
                
                st.plotly_chart(fig, use_container_width=True, config=config)
                
                # Tampilkan statistik
                display_bubble_statistics(plot_data, x_col, y_col, size_col)
                
                # Tampilkan info optimasi
                show_bubble_optimization_info(data_size, len(plot_data), optimization_mode)
                
        except Exception as e:
            st.error(f"Error membuat bubble chart: {str(e)}")
            # Fallback ke metode sederhana
            create_simple_bubble_fallback(df, x_col, y_col, size_col, color_col)

def optimize_bubble_data(df, x_col, y_col, size_col, color_col, data_size, optimization_mode, max_bubbles):
    """Optimasi data untuk bubble chart dengan sampling yang tepat"""
    # Pilih kolom yang diperlukan
    columns_needed = [x_col, y_col, size_col]
    if color_col:
        columns_needed.append(color_col)
    
    # Filter data yang valid
    plot_data = df[columns_needed].dropna()
    
    if len(plot_data) == 0:
        return plot_data
    
    # Tentukan target sample size
    target_sizes = {
        "Auto": min(max_bubbles, data_size),
        "Fast": min(300, data_size),
        "Balanced": min(800, data_size),
        "Detailed": min(1500, data_size)
    }
    
    target_size = target_sizes[optimization_mode]
    
    # Jika data lebih besar dari target, lakukan sampling
    if len(plot_data) > target_size:
        if optimization_mode == "Fast":
            # Sampling berdasarkan ukuran (ambil yang paling signifikan)
            plot_data_sorted = plot_data.nlargest(target_size, size_col)
            return plot_data_sorted
            
        elif optimization_mode == "Balanced":
            # Stratified sampling berdasarkan size quantile
            try:
                quantiles = pd.qcut(plot_data[size_col], q=4, duplicates='drop')
                samples_per_quantile = target_size // 4
                sampled_dfs = []
                
                for quantile in quantiles.cat.categories:
                    quantile_data = plot_data[quantiles == quantile]
                    if len(quantile_data) > 0:
                        sample_size = min(samples_per_quantile, len(quantile_data))
                        sampled_dfs.append(quantile_data.sample(n=sample_size, random_state=42))
                
                sampled_data = pd.concat(sampled_dfs, ignore_index=True)
                
                # Jika masih kurang, tambahkan berdasarkan size
                if len(sampled_data) < target_size:
                    remaining = target_size - len(sampled_data)
                    additional_samples = plot_data.nlargest(remaining, size_col)
                    sampled_data = pd.concat([sampled_data, additional_samples], ignore_index=True)
                    
                return sampled_data
            except:
                # Fallback ke size-based sampling
                return plot_data.nlargest(target_size, size_col)
        else:
            # Random sampling dengan prioritas size besar
            if optimization_mode == "Detailed":
                # Gabungkan random sampling dengan size-based
                size_based = plot_data.nlargest(target_size // 2, size_col)
                random_samples = plot_data.sample(n=target_size - len(size_based), random_state=42)
                return pd.concat([size_based, random_samples], ignore_index=True)
            else:
                return plot_data.sample(n=target_size, random_state=42)
    
    return plot_data

def normalize_bubble_sizes(plot_data, size_col, size_factor):
    """Normalisasi ukuran bubble untuk visualisasi yang lebih baik"""
    plot_data = plot_data.copy()
    
    # Normalisasi ukuran antara 5-50 untuk visualisasi optimal
    min_size = plot_data[size_col].min()
    max_size = plot_data[size_col].max()
    
    if max_size > min_size:
        # Scale ke range yang reasonable
        scaled_sizes = (plot_data[size_col] - min_size) / (max_size - min_size)
        plot_data['bubble_size_normalized'] = 5 + scaled_sizes * (size_factor * 5)
    else:
        plot_data['bubble_size_normalized'] = 10  # Default size
    
    return plot_data

def create_optimized_bubble_chart(plot_data, x_col, y_col, size_col, color_col, original_size):
    """Buat bubble chart dengan optimasi performa"""
    
    # OPTIMASI: Gunakan scattergl untuk data banyak
    use_webgl = len(plot_data) > 500
    if use_webgl:
        scatter_function = px.scatter_gl
        chart_type = "Bubble Chart (WebGL)"
    else:
        scatter_function = px.scatter
        chart_type = "Bubble Chart"
    
    # Buat plot
    fig = scatter_function(
        plot_data,
        x=x_col,
        y=y_col,
        size='bubble_size_normalized',
        color=color_col,
        title=f"{chart_type}: {y_col} vs {x_col} - Size: {size_col} ({len(plot_data):,} bubbles)",
        hover_name=plot_data.index if plot_data.index.name else None,
        size_max=20,  # Batasi ukuran maksimum
        opacity=0.7
    )
    
    # OPTIMASI: Update marker untuk performa
    fig.update_traces(
        marker=dict(
            line=dict(width=0),  # No border untuk performa
            sizemode='diameter'
        ),
        selector=dict(mode='markers')
    )
    
    # OPTIMASI: Sederhanakan hover template
    if len(plot_data) > 500:
        hover_template = (f'{x_col}: %{{x:.2f}}<br>'
                         f'{y_col}: %{{y:.2f}}<br>'
                         f'{size_col}: %{{marker.size:.1f}}<extra></extra>')
    else:
        if color_col:
            hover_template = (f'{x_col}: %{{x:.2f}}<br>'
                             f'{y_col}: %{{y:.2f}}<br>'
                             f'{size_col}: %{{marker.size:.1f}}<br>'
                             f'{color_col}: %{{marker.color}}<extra></extra>')
        else:
            hover_template = (f'{x_col}: %{{x:.2f}}<br>'
                             f'{y_col}: %{{y:.2f}}<br>'
                             f'{size_col}: %{{marker.size:.1f}}<extra></extra>')
    
    fig.update_traces(hovertemplate=hover_template)
    
    # Layout yang dioptimalkan
    fig.update_layout(
        height=500,
        showlegend=len(plot_data) <= 200,  # Sembunyikan legend untuk banyak bubbles
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ) if len(plot_data) <= 200 else None
    )
    
    return fig

def display_bubble_statistics(plot_data, x_col, y_col, size_col):
    """Tampilkan statistik bubble chart"""
    with st.expander("📊 Statistik Bubble Chart", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Jumlah Gelembung", len(plot_data))
        with col2:
            st.metric(f"Rata2 {x_col}", f"{plot_data[x_col].mean():.2f}")
        with col3:
            st.metric(f"Rata2 {y_col}", f"{plot_data[y_col].mean():.2f}")
        with col4:
            st.metric(f"Rata2 {size_col}", f"{plot_data[size_col].mean():.2f}")
        
        # Top 5 largest bubbles
        st.markdown("**🔍 Gelembung Terbesar:**")
        largest_bubbles = plot_data.nlargest(5, size_col)[[x_col, y_col, size_col]]
        st.dataframe(largest_bubbles.style.format({
            x_col: "{:.2f}",
            y_col: "{:.2f}", 
            size_col: "{:.2f}"
        }), use_container_width=True)
        
        # Korelasi antar variabel
        try:
            corr_xy = plot_data[x_col].corr(plot_data[y_col])
            corr_xsize = plot_data[x_col].corr(plot_data[size_col])
            corr_ysize = plot_data[y_col].corr(plot_data[size_col])
            
            col5, col6, col7 = st.columns(3)
            with col5:
                st.metric("Korelasi X-Y", f"{corr_xy:.3f}")
            with col6:
                st.metric("Korelasi X-Size", f"{corr_xsize:.3f}")
            with col7:
                st.metric("Korelasi Y-Size", f"{corr_ysize:.3f}")
        except:
            st.info("Tidak dapat menghitung korelasi")

def show_bubble_optimization_info(original_size, processed_size, optimization_mode):
    """Tampilkan informasi optimasi"""
    reduction_pct = ((original_size - processed_size) / original_size) * 100 if original_size > 0 else 0
    
    if reduction_pct > 10:
        with st.expander("⚡ Info Optimasi Performa", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Original", f"{original_size:,}")
            with col2:
                st.metric("Gelembung Ditampilkan", f"{processed_size:,}")
            with col3:
                st.metric("Reduksi", f"{reduction_pct:.1f}%")
            
            optimization_strategies = {
                "Fast": "• ✅ **Size-based sampling** (ambil yang terbesar)\n• ✅ **WebGL Rendering**\n• ✅ **Minimal hover effects**",
                "Balanced": "• ✅ **Stratified sampling** (berdasarkan quantile size)\n• ✅ **Size normalization**\n• ✅ **Optimized bubble sizes**",
                "Detailed": "• ✅ **Hybrid sampling** (size + random)\n• ✅ **Full features**\n• ✅ **Detailed hover info**"
            }
            
            st.info(f"**Mode {optimization_mode}**: {optimization_strategies.get(optimization_mode, 'Custom optimization')}")

def create_simple_bubble_fallback(df, x_col, y_col, size_col, color_col):
    """Fallback method untuk data yang bermasalah"""
    st.warning("Menggunakan metode fallback yang sederhana...")
    
    # Sample kecil untuk memastikan bisa render
    sample_size = min(200, len(df))
    plot_data = df[[x_col, y_col, size_col] + ([color_col] if color_col else [])].dropna().head(sample_size)
    
    # Normalisasi sederhana
    if len(plot_data) > 0:
        min_size = plot_data[size_col].min()
        max_size = plot_data[size_col].max()
        if max_size > min_size:
            plot_data = plot_data.copy()
            plot_data['bubble_size_norm'] = 10 + ((plot_data[size_col] - min_size) / (max_size - min_size)) * 30
    
    fig = px.scatter(
        plot_data,
        x=x_col,
        y=y_col,
        size='bubble_size_norm' if 'bubble_size_norm' in plot_data.columns else size_col,
        color=color_col,
        title=f"Simple Bubble: {y_col} vs {x_col} ({len(plot_data)} bubbles)"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Versi ultra-ringan untuk data ekstrem
def create_ultra_fast_bubble_chart(df, numeric_cols, non_numeric_cols):
    """Versi ultra-ringan untuk data > 100k rows"""
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_col = st.selectbox("Pilih kolom X", numeric_cols[:6], key="ultra_bubble_x")
    with col2:
        y_col = st.selectbox("Pilih kolom Y", numeric_cols[:6], key="ultra_bubble_y")
    with col3:
        size_col = st.selectbox("Pilih kolom ukuran", numeric_cols[:6], key="ultra_bubble_size")
    
    if x_col and y_col and size_col:
        # Sampling agresif - hanya 150 bubbles terbesar
        plot_data = df[[x_col, y_col, size_col]].dropna().nlargest(150, size_col)
        
        if len(plot_data) > 0:
            # Normalisasi ukuran
            min_size = plot_data[size_col].min()
            max_size = plot_data[size_col].max()
            if max_size > min_size:
                plot_data = plot_data.copy()
                plot_data['bubble_size_norm'] = 5 + ((plot_data[size_col] - min_size) / (max_size - min_size)) * 15
            
            # Bubble chart sederhana dengan WebGL
            fig = px.scatter_gl(
                plot_data,
                x=x_col,
                y=y_col,
                size='bubble_size_norm' if 'bubble_size_norm' in plot_data.columns else size_col,
                title=f"Ultra-Fast Bubble: Top 150 by {size_col}"
            )
            
            fig.update_traces(
                marker=dict(opacity=0.6, line=dict(width=0)),
                hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Size: %{marker.size:.1f}<extra></extra>'
            )
            
            fig.update_layout(height=350, showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.info(f"📊 Menampilkan 150 gelembung terbesar dari {len(df[[x_col, y_col, size_col]].dropna()):,} data points")
        
        with st.expander("ℹ️ Keterangan Bubble Chart"):
            st.markdown("""
            **Bubble Chart** adalah scatter plot dengan dimensi ketiga (ukuran gelembung).
            - **Kelebihan**: Menampilkan tiga dimensi data sekaligus
            - **Kekurangan**: Bisa sulit dibaca jika terlalu banyak gelembung
            - **Penggunaan**: Analisis tiga variabel, comparison dengan multiple dimensions
            
            **Optimasi untuk Data Besar:**
            • Size-based sampling untuk mempertahankan insight
            • WebGL rendering untuk performa
            • Normalisasi ukuran untuk visualisasi optimal
            """)

def create_gauge_chart(df, numeric_cols):
    
    # Deteksi ukuran data
    data_size = len(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        value_col = st.selectbox("Pilih kolom nilai", numeric_cols, key="gauge_value")
    
    with col2:
        # Auto-calculate max value atau manual input
        auto_max = st.checkbox("Auto calculate max", value=True, key="gauge_auto_max")
        if auto_max and value_col:
            max_val = df[value_col].max() * 1.1  # Tambah 10% buffer
            st.info(f"Max: {max_val:.2f}")
        else:
            max_val = st.number_input("Nilai maksimum gauge", 
                                    value=100.0, 
                                    min_value=0.1,
                                    key="gauge_max")
    
    with col3:
        calculation_method = st.selectbox(
            "Metode kalkulasi",
            ["Mean", "Median", "Sum", "Last Value", "Custom Percentile"],
            key="gauge_calc_method"
        )
        
        if calculation_method == "Custom Percentile":
            percentile = st.slider("Percentile", 0, 100, 90, key="gauge_percentile")
    
    if value_col:
        try:
            with st.spinner("🔄 Menghitung nilai gauge..."):
                # OPTIMASI 1: Kalkulasi nilai yang efisien
                gauge_value, reference_value = calculate_gauge_values(
                    df, value_col, calculation_method, 
                    percentile if 'percentile' in locals() else None,
                    data_size
                )
                
                # OPTIMASI 2: Buat gauge chart yang dioptimalkan
                fig = create_optimized_gauge_chart(
                    gauge_value, reference_value, value_col, 
                    max_val, calculation_method, data_size
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Tampilkan statistik tambahan
                display_gauge_statistics(df, value_col, gauge_value, reference_value, data_size)
                
        except Exception as e:
            st.error(f"Error membuat gauge chart: {str(e)}")
            # Fallback ke metode sederhana
            create_simple_gauge_fallback(df, value_col)

def calculate_gauge_values(df, value_col, calculation_method, percentile, data_size):
    """Hitung nilai gauge dengan optimasi untuk data besar"""
    
    # OPTIMASI: Sampling untuk data sangat besar
    if data_size > 100000:
        # Gunakan sample representatif
        sample_size = min(10000, data_size)
        sample_df = df[value_col].dropna().sample(n=sample_size, random_state=42)
        st.info(f"📊 Menggunakan sample {sample_size:,} dari {data_size:,} data points")
    else:
        sample_df = df[value_col].dropna()
    
    if len(sample_df) == 0:
        return 0, 0
    
    # Kalkulasi berdasarkan metode yang dipilih
    if calculation_method == "Mean":
        value = sample_df.mean()
        reference = sample_df.median()
    elif calculation_method == "Median":
        value = sample_df.median()
        reference = sample_df.mean()
    elif calculation_method == "Sum":
        # Scale sum untuk data sample
        if data_size > 100000:
            scale_factor = data_size / len(sample_df)
            value = sample_df.sum() * scale_factor
        else:
            value = sample_df.sum()
        reference = value * 0.8  # Reference 80% dari total
    elif calculation_method == "Last Value":
        value = df[value_col].iloc[-1] if len(df) > 0 else 0
        reference = sample_df.mean()
    elif calculation_method == "Custom Percentile":
        value = sample_df.quantile(percentile/100)
        reference = sample_df.median()
    else:
        value = sample_df.mean()
        reference = sample_df.median()
    
    return float(value), float(reference)

def create_optimized_gauge_chart(value, reference, value_col, max_value, calculation_method, data_size):
    """Buat gauge chart yang dioptimalkan"""
    
    # OPTIMASI: Tentukan warna berdasarkan nilai
    value_ratio = value / max_value if max_value > 0 else 0
    
    if value_ratio < 0.3:
        bar_color = "red"
        threshold_color = "darkred"
    elif value_ratio < 0.7:
        bar_color = "orange"
        threshold_color = "darkorange"
    else:
        bar_color = "green"
        threshold_color = "darkgreen"
    
    # OPTIMASI: Steps dengan warna yang meaningful
    steps = [
        {'range': [0, max_value * 0.3], 'color': "lightcoral"},
        {'range': [max_value * 0.3, max_value * 0.7], 'color': "lightyellow"},
        {'range': [max_value * 0.7, max_value], 'color': "lightgreen"}
    ]
    
    # Buat gauge figure
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"{calculation_method} {value_col}",
            'font': {'size': 16}
        },
        number = {
            'valueformat': ".2f",
            'font': {'size': 24}
        },
        delta = {
            'reference': reference,
            'increasing': {'color': "green"},
            'decreasing': {'color': "red"},
            'valueformat': ".2f"
        },
        gauge = {
            'axis': {
                'range': [0, max_value],
                'tickwidth': 1,
                'tickcolor': "darkblue"
            },
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': steps,
            'threshold': {
                'line': {'color': threshold_color, 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    # Layout yang dioptimalkan
    fig.update_layout(
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='white',
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def display_gauge_statistics(df, value_col, gauge_value, reference_value, data_size):
    """Tampilkan statistik tambahan untuk gauge chart"""
    
    with st.expander("📈 Statistik Detail", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        # Kalkulasi statistik dengan sampling untuk data besar
        if data_size > 50000:
            sample_data = df[value_col].dropna().sample(n=min(10000, data_size), random_state=42)
        else:
            sample_data = df[value_col].dropna()
        
        with col1:
            st.metric("Nilai Gauge", f"{gauge_value:.2f}")
        with col2:
            st.metric("Nilai Referensi", f"{reference_value:.2f}")
        with col3:
            diff = gauge_value - reference_value
            diff_pct = (diff / reference_value * 100) if reference_value != 0 else 0
            st.metric("Selisih", f"{diff:+.2f}", f"{diff_pct:+.1f}%")
        with col4:
            if len(sample_data) > 0:
                completion_pct = (gauge_value / sample_data.max() * 100) if sample_data.max() > 0 else 0
                st.metric("Progress", f"{completion_pct:.1f}%")
        
        # Progress bar visual
        if len(sample_data) > 0:
            max_val = sample_data.max()
            progress_ratio = min(gauge_value / max_val, 1.0) if max_val > 0 else 0
            st.progress(float(progress_ratio), 
                       text=f"Progress: {progress_ratio*100:.1f}% dari maksimum {max_val:.2f}")
        
        # Quick stats
        if len(sample_data) > 0:
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("Data Points", f"{len(sample_data):,}")
            with col6:
                st.metric("Std Dev", f"{sample_data.std():.2f}")
            with col7:
                st.metric("Min", f"{sample_data.min():.2f}")
            with col8:
                st.metric("Max", f"{sample_data.max():.2f}")


def create_simple_gauge_fallback(df, value_col):
    """Fallback method untuk data yang bermasalah"""
    st.warning("Menggunakan metode fallback sederhana...")
    
    # Kalkulasi sederhana
    clean_data = df[value_col].dropna().head(1000)  # Batasi data
    if len(clean_data) == 0:
        st.error("Tidak ada data valid")
        return
    
    value = clean_data.mean()
    max_val = clean_data.max() * 1.1
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Simple Gauge: {value_col}"},
        gauge = {
            'axis': {'range': [0, max_val]},
            'bar': {'color': "darkblue"},
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Versi multiple gauges untuk dashboard
def create_multi_gauge_dashboard(df, numeric_cols):
    """Dashboard dengan multiple gauge charts"""
    
    # Pilih hingga 4 metrik
    selected_metrics = st.multiselect(
        "Pilih metrik untuk dashboard",
        numeric_cols[:8],  # Batasi pilihan
        default=numeric_cols[:min(4, len(numeric_cols))],
        key="multi_gauge_metrics"
    )
    
    if selected_metrics:
        # Tentukan layout berdasarkan jumlah metrik
        n_metrics = len(selected_metrics)
        if n_metrics == 1:
            cols = [1]
        elif n_metrics == 2:
            cols = st.columns(2)
        elif n_metrics == 3:
            cols = st.columns(3)
        else:
            cols = st.columns(2)
        
        # Buat gauge untuk setiap metrik
        for i, metric in enumerate(selected_metrics):
            if n_metrics <= 3:
                with cols[i]:
                    create_single_gauge_compact(df, metric)
            else:
                # Untuk 4 metrik, buat 2x2 grid
                row_idx = i // 2
                col_idx = i % 2
                if col_idx == 0:
                    col1, col2 = st.columns(2)
                with col1 if col_idx == 0 else col2:
                    create_single_gauge_compact(df, metric)

def create_single_gauge_compact(df, metric):
    """Buat gauge chart compact untuk dashboard"""
    try:
        # Kalkulasi cepat
        sample_data = df[metric].dropna()
        if len(sample_data) == 0:
            st.warning(f"No data for {metric}")
            return
        
        value = sample_data.mean()
        max_val = sample_data.max() * 1.1
        value_ratio = value / max_val if max > 0 else 0
        
        # Tentukan warna
        bar_color = "green" if value_ratio > 0.7 else "orange" if value_ratio > 0.3 else "red"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': metric[:20], 'font': {'size': 12}},
            number = {'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, max_val]},
                'bar': {'color': bar_color},
                'steps': [
                    {'range': [0, max_val*0.3], 'color': "lightcoral"},
                    {'range': [max_val*0.3, max_val*0.7], 'color': "lightyellow"},
                    {'range': [max_val*0.7, max_val], 'color': "lightgreen"}
                ]
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=30))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
    except Exception as e:
        st.error(f"Error creating gauge for {metric}")



def create_radar_chart(df, numeric_cols, non_numeric_cols):
    
    # Deteksi ukuran data
    data_size = len(df)
    if data_size > 100000:
        st.info(f"⚡ Mode Optimasi: Data besar ({data_size:,} rows) - Menggunakan sampling otomatis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        category_col = st.selectbox("Pilih kolom kategori", non_numeric_cols, key="radar_category")
    
    with col2:
        max_metrics = st.slider("Maksimum metrik", 
                              min_value=3, max_value=10, value=6,
                              key="radar_max_metrics")
    
    # Filter numeric cols yang feasible untuk radar chart
    suitable_numeric_cols = [col for col in numeric_cols 
                           if df[col].nunique() > 1 and df[col].dtype in ['float64', 'int64']]
    
    selected_cols = st.multiselect(
        "Pilih kolom nilai", 
        suitable_numeric_cols[:20],  # Batasi pilihan
        default=suitable_numeric_cols[:min(max_metrics, len(suitable_numeric_cols))], 
        key="radar_values"
    )
    
    # Pengaturan optimasi
    with st.expander("⚙️ Pengaturan Optimasi", expanded=False):
        col3, col4, col5 = st.columns(3)
        with col3:
            optimization_mode = st.selectbox(
                "Mode Optimasi",
                ["Auto", "Fast", "Balanced", "Detailed"],
                index=0 if data_size > 50000 else 2,
                key="radar_optim"
            )
        with col4:
            max_categories = st.slider(
                "Maksimum kategori",
                min_value=3,
                max_value=15,
                value=8 if data_size > 50000 else 12,
                key="radar_max_categories"
            )
        with col5:
            normalize_data = st.checkbox(
                "Normalisasi data", 
                value=True,
                help="Scale data ke range 0-1 untuk perbandingan yang lebih baik"
            )
    
    if category_col and selected_cols and len(selected_cols) >= 3:
        try:
            with st.spinner("🔄 Memproses data radar chart..."):
                # OPTIMASI 1: Filter dan sampling data
                plot_data = optimize_radar_data(
                    df, category_col, selected_cols, data_size, 
                    optimization_mode, max_categories
                )
                
                if plot_data is None or len(plot_data) == 0:
                    st.warning("Tidak ada data valid untuk radar chart")
                    return
                
                # OPTIMASI 2: Normalisasi data jika diperlukan
                if normalize_data:
                    plot_data = normalize_radar_data(plot_data, selected_cols)
                
                # OPTIMASI 3: Batasi jumlah kategori yang ditampilkan
                radar_data = aggregate_radar_data(plot_data, category_col, selected_cols, max_categories)
                
                # OPTIMASI 4: Buat radar chart yang dioptimalkan
                fig = create_optimized_radar_chart(radar_data, category_col, selected_cols, data_size)
                
                # OPTIMASI 5: Konfigurasi plotly yang ringan
                config = {
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestGl2d'],
                    'responsive': True
                }
                
                st.plotly_chart(fig, use_container_width=True, config=config)
                
                # Tampilkan data table
                display_radar_data_table(radar_data, category_col, selected_cols)
                
                # Tampilkan info optimasi
                show_radar_optimization_info(data_size, len(plot_data), len(radar_data), optimization_mode)
                
        except Exception as e:
            st.error(f"Error membuat radar chart: {str(e)}")
            # Fallback ke metode sederhana
            create_simple_radar_fallback(df, category_col, selected_cols)

def optimize_radar_data(df, category_col, selected_cols, data_size, optimization_mode, max_categories):
    """Optimasi data untuk radar chart"""
    
    # Pilih kolom yang diperlukan
    columns_needed = [category_col] + selected_cols
    plot_data = df[columns_needed].dropna()
    
    if len(plot_data) == 0:
        return None
    
    # OPTIMASI: Sampling untuk data besar
    if data_size > 50000:
        target_sizes = {
            "Auto": min(10000, data_size),
            "Fast": min(5000, data_size),
            "Balanced": min(20000, data_size),
            "Detailed": min(50000, data_size)
        }
        
        target_size = target_sizes[optimization_mode]
        
        if len(plot_data) > target_size:
            # Stratified sampling berdasarkan kategori
            try:
                category_counts = plot_data[category_col].value_counts()
                top_categories = category_counts.head(max_categories).index
                filtered_data = plot_data[plot_data[category_col].isin(top_categories)]
                
                # Sample dari setiap kategori
                samples_per_category = target_size // len(top_categories)
                sampled_dfs = []
                
                for category in top_categories:
                    category_data = filtered_data[filtered_data[category_col] == category]
                    if len(category_data) > 0:
                        sample_size = min(samples_per_category, len(category_data))
                        sampled_dfs.append(category_data.sample(n=sample_size, random_state=42))
                
                if sampled_dfs:
                    plot_data = pd.concat(sampled_dfs, ignore_index=True)
                else:
                    plot_data = plot_data.sample(n=target_size, random_state=42)
                    
            except:
                # Fallback ke random sampling
                plot_data = plot_data.sample(n=target_size, random_state=42)
    
    return plot_data

def normalize_radar_data(plot_data, selected_cols):
    """Normalisasi data untuk radar chart (0-1 scaling)"""
    plot_data = plot_data.copy()
    
    for col in selected_cols:
        min_val = plot_data[col].min()
        max_val = plot_data[col].max()
        
        if max_val > min_val:
            plot_data[col] = (plot_data[col] - min_val) / (max_val - min_val)
        else:
            plot_data[col] = 0.5  # Default value jika semua sama
    
    return plot_data

def aggregate_radar_data(plot_data, category_col, selected_cols, max_categories):
    """Aggregasi data untuk radar chart"""
    
    # Ambil kategori paling banyak
    category_counts = plot_data[category_col].value_counts()
    top_categories = category_counts.head(max_categories).index
    
    # Aggregasi data
    radar_data = plot_data[plot_data[category_col].isin(top_categories)]
    radar_data = radar_data.groupby(category_col, observed=True)[selected_cols].mean().reset_index()
    
    return radar_data

def create_optimized_radar_chart(radar_data, category_col, selected_cols, original_size):
    """Buat radar chart yang dioptimalkan"""
    
    fig = go.Figure()
    
    # Warna yang dioptimalkan untuk visibility
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    
    # Batasi opacity berdasarkan jumlah kategori
    n_categories = len(radar_data)
    base_opacity = max(0.3, 0.8 - (n_categories * 0.05))
    
    for idx, row in radar_data.iterrows():
        # Siapkan data untuk radar (tutup loop dengan nilai pertama)
        r_values = row[selected_cols].values.tolist() + [row[selected_cols].values[0]]
        theta_values = selected_cols + [selected_cols[0]]
        
        # Pendekkan label jika terlalu panjang
        category_name = str(row[category_col])
        if len(category_name) > 20:
            category_name = category_name[:17] + "..."
        
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta_values,
            fill='toself' if n_categories <= 8 else 'none',  # Non-fill untuk banyak kategori
            fillcolor=colors[idx % len(colors)] if n_categories <= 8 else None,
            line=dict(
                color=colors[idx % len(colors)],
                width=2 if n_categories <= 8 else 1
            ),
            name=category_name,
            opacity=base_opacity,
            hovertemplate=(
                f'<b>{category_name}</b><br>' +
                '<br>'.join([f'{col}: %{{r:.3f}}' for col in selected_cols]) +
                '<extra></extra>'
            )
        ))
    
    # Layout yang dioptimalkan
    layout_config = {
        'polar': dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],  # Fixed range untuk normalized data
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
                rotation=90,
                direction="clockwise"
            ),
            bgcolor='rgba(0,0,0,0.02)'
        ),
        'showlegend': n_categories <= 12,  # Sembunyikan legend jika terlalu banyak
        'height': 500,
        'margin': dict(l=50, r=50, t=50, b=50),
        'paper_bgcolor': 'white'
    }
    
    # Sesuaikan legend berdasarkan jumlah kategori
    if n_categories <= 12:
        layout_config['legend'] = dict(
            orientation="v" if n_categories <= 6 else "h",
            yanchor="bottom",
            y=-0.2 if n_categories > 6 else 0.5,
            xanchor="center",
            x=0.5
        )
    
    fig.update_layout(**layout_config)
    
    return fig

def display_radar_data_table(radar_data, category_col, selected_cols):
    """Tampilkan data table untuk radar chart"""
    
    with st.expander("📊 Data Radar Chart", expanded=False):
        # Format data untuk display
        display_data = radar_data.copy()
        for col in selected_cols:
            display_data[col] = display_data[col].round(3)
        
        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )
        
        # Statistik ringkas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Kategori", len(radar_data))
        with col2:
            st.metric("Jumlah Metrik", len(selected_cols))
        with col3:
            avg_values = radar_data[selected_cols].mean().mean()
            st.metric("Rata-rata Nilai", f"{avg_values:.3f}")

def show_radar_optimization_info(original_size, processed_size, displayed_categories, optimization_mode):
    """Tampilkan informasi optimasi"""
    
    reduction_pct = ((original_size - processed_size) / original_size) * 100 if original_size > 0 else 0
    
    if reduction_pct > 10 or displayed_categories < 10:
        with st.expander("⚡ Info Optimasi Performa", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Original", f"{original_size:,}")
            with col2:
                st.metric("Data Diproses", f"{processed_size:,}")
            with col3:
                st.metric("Kategori Ditampilkan", displayed_categories)
            
            if reduction_pct > 10:
                st.metric("Reduksi Data", f"{reduction_pct:.1f}%")
            
            optimization_strategies = {
                "Fast": "• ✅ **Aggressive sampling**\n• ✅ **Limited categories**\n• ✅ **Minimal styling**",
                "Balanced": "• ✅ **Stratified sampling**\n• ✅ **Smart normalization**\n• ✅ **Optimized visuals**",
                "Detailed": "• ✅ **Maximum data retention**\n• ✅ **Full features**\n• ✅ **Detailed hover**"
            }
            
            st.info(f"**Mode {optimization_mode}**: {optimization_strategies.get(optimization_mode, 'Custom optimization')}")

def create_simple_radar_fallback(df, category_col, selected_cols):
    """Fallback method untuk data yang bermasalah"""
    st.warning("Menggunakan metode fallback sederhana...")
    
    # Ambil sample kecil dan kategori terbatas
    sample_data = df[[category_col] + selected_cols].dropna().head(1000)
    top_categories = sample_data[category_col].value_counts().head(5).index
    filtered_data = sample_data[sample_data[category_col].isin(top_categories)]
    
    if len(filtered_data) == 0:
        st.error("Tidak ada data valid setelah filtering")
        return
    
    radar_data = filtered_data.groupby(category_col)[selected_cols].mean().reset_index()
    
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, row in radar_data.iterrows():
        r_values = row[selected_cols].values.tolist() + [row[selected_cols].values[0]]
        theta_values = selected_cols + [selected_cols[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta_values,
            fill='toself',
            name=row[category_col],
            line_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Versi ultra-ringan untuk data ekstrem
def create_ultra_fast_radar(df, numeric_cols, non_numeric_cols):
    """Versi ultra-ringan untuk data > 100k rows"""
    
    col1, col2 = st.columns(2)
    with col1:
        category_col = st.selectbox("Kolom kategori", non_numeric_cols[:5], key="ultra_radar_cat")
    with col2:
        metric_count = st.slider("Jumlah metrik", 3, 6, 4, key="ultra_radar_metrics")
    
    # Pilih metrik otomatis
    suitable_cols = [col for col in numeric_cols 
                   if df[col].nunique() > 1 and df[col].dtype in ['float64', 'int64']]
    selected_cols = suitable_cols[:metric_count]
    
    if category_col and len(selected_cols) >= 3:
        # Aggregasi langsung dengan sampling
        sample_data = df[[category_col] + selected_cols].dropna()
        if len(sample_data) > 1000:
            sample_data = sample_data.sample(n=1000, random_state=42)
        
        top_categories = sample_data[category_col].value_counts().head(4).index
        radar_data = sample_data[sample_data[category_col].isin(top_categories)]
        radar_data = radar_data.groupby(category_col)[selected_cols].mean().reset_index()
        
        # Normalisasi
        for col in selected_cols:
            min_val = radar_data[col].min()
            max_val = radar_data[col].max()
            if max_val > min_val:
                radar_data[col] = (radar_data[col] - min_val) / (max_val - min_val)
        
        # Simple radar
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, row in radar_data.iterrows():
            r_values = row[selected_cols].values.tolist() + [row[selected_cols].values[0]]
            theta_values = selected_cols + [selected_cols[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                fill='toself',
                name=row[category_col][:15],
                line_color=colors[idx % len(colors)],
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=350,
            margin=dict(l=30, r=30, t=30, b=30)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.info(f"📊 Ultra-Fast Mode: {len(radar_data)} kategori, {len(selected_cols)} metrik")


def create_box_plot(df, numeric_cols):
    
    # Deteksi ukuran data
    data_size = len(df)
    if data_size > 100000:
        st.info(f"⚡ Mode Optimasi: Data besar ({data_size:,} rows) - Menggunakan sampling otomatis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        col = st.selectbox("Pilih kolom untuk box plot", numeric_cols, key="box_col")
    
    with col2:
        # Pengaturan optimasi
        optimization_mode = st.selectbox(
            "Mode Optimasi",
            ["Auto", "Fast", "Balanced", "Detailed"],
            index=0 if data_size > 50000 else 2,
            key="box_optim"
        )
    
    with col3:
        show_points = st.selectbox(
            "Tampilkan points",
            ["None", "Outliers Only", "All"],
            index=1 if data_size < 10000 else 0,
            key="box_points"
        )
    
    # Pengaturan lanjutan
    with st.expander("⚙️ Pengaturan Lanjutan", expanded=False):
        col4, col5, col6 = st.columns(3)
        with col4:
            max_points = st.slider(
                "Maksimum data points",
                min_value=1000,
                max_value=50000,
                value=10000 if data_size > 50000 else 20000,
                key="box_max_points"
            )
        with col5:
            notch = st.checkbox(
                "Tampilkan notch", 
                value=False,
                help="Menampilkan interval kepercayaan median"
            )
        with col6:
            log_scale = st.checkbox(
                "Skala logaritmik",
                value=False,
                help="Berguna untuk data dengan skew tinggi"
            )
    
    if col:
        try:
            with st.spinner("🔄 Memproses data box plot..."):
                # OPTIMASI 1: Filter dan sampling data
                plot_data = optimize_box_data(df, col, data_size, optimization_mode, max_points)
                
                if len(plot_data) == 0:
                    st.warning(f"Tidak ada data valid untuk kolom {col}")
                    return
                
                # OPTIMASI 2: Buat box plot yang dioptimalkan
                fig = create_optimized_box_plot(plot_data, col, show_points, notch, log_scale, data_size)
                
                # OPTIMASI 3: Konfigurasi plotly yang ringan
                config = {
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'responsive': True
                }
                
                st.plotly_chart(fig, use_container_width=True, config=config)
                
                # Tampilkan statistik detail
                display_box_statistics(plot_data, col, data_size)
                
                # Tampilkan info optimasi
                show_box_optimization_info(data_size, len(plot_data), optimization_mode)
                
        except Exception as e:
            st.error(f"Error membuat box plot: {str(e)}")
            # Fallback ke metode sederhana
            create_simple_box_fallback(df, col)

def optimize_box_data(df, col, data_size, optimization_mode, max_points):
    """Optimasi data untuk box plot dengan sampling yang tepat"""
    
    # Filter data yang valid
    plot_data = df[col].dropna()
    
    if len(plot_data) == 0:
        return plot_data
    
    # Tentukan target sample size
    target_sizes = {
        "Auto": min(max_points, data_size),
        "Fast": min(5000, data_size),
        "Balanced": min(20000, data_size),
        "Detailed": min(50000, data_size)
    }
    
    target_size = target_sizes[optimization_mode]
    
    # Jika data lebih besar dari target, lakukan sampling
    if len(plot_data) > target_size:
        if optimization_mode == "Fast":
            # Systematic sampling untuk performa maksimal
            step = len(plot_data) // target_size
            sampled_data = plot_data.iloc[::step]
        elif optimization_mode == "Balanced":
            # Stratified sampling untuk mempertahankan distribusi
            try:
                # Bagi data menjadi quantiles dan sample dari setiap quantile
                n_quantiles = min(10, target_size // 100)
                quantiles = pd.qcut(plot_data, q=n_quantiles, duplicates='drop')
                
                samples_per_quantile = target_size // n_quantiles
                sampled_dfs = []
                
                for quantile in quantiles.cat.categories:
                    quantile_data = plot_data[quantiles == quantile]
                    if len(quantile_data) > 0:
                        sample_size = min(samples_per_quantile, len(quantile_data))
                        sampled_dfs.append(quantile_data.sample(n=sample_size, random_state=42))
                
                sampled_data = pd.concat(sampled_dfs, ignore_index=True)
                
                # Jika masih kurang, tambahkan random sampling
                if len(sampled_data) < target_size:
                    remaining = target_size - len(sampled_data)
                    additional_samples = plot_data.sample(n=remaining, random_state=42)
                    sampled_data = pd.concat([sampled_data, additional_samples], ignore_index=True)
                    
            except:
                # Fallback ke random sampling
                sampled_data = plot_data.sample(n=target_size, random_state=42)
        else:
            # Random sampling untuk mode lain
            sampled_data = plot_data.sample(n=target_size, random_state=42)
        
        return sampled_data
    
    return plot_data

def create_optimized_box_plot(plot_data, col, show_points, notch, log_scale, original_size):
    """Buat box plot yang dioptimalkan untuk performa"""
    
    # Tentukan parameter points berdasarkan ukuran data dan pilihan user
    if show_points == "None":
        box_points = False
    elif show_points == "Outliers Only":
        box_points = 'outliers'
    else:  # "All"
        box_points = 'all' if len(plot_data) <= 5000 else 'outliers'
    
    # Buat box plot
    fig = px.box(
        plot_data, 
        y=col,
        title=f"Box Plot {col} ({len(plot_data):,} data points)",
        points=box_points,
        notched=notch
    )
    
    # OPTIMASI: Update trace untuk performa
    fig.update_traces(
        marker=dict(
            size=4 if box_points in ['all', 'outliers'] else 0,
            opacity=0.6
        ),
        line=dict(width=1.5),
        selector=dict(type='box')
    )
    
    # Skala logaritmik jika diperlukan
    if log_scale:
        fig.update_layout(yaxis_type="log")
    
    # Layout yang dioptimalkan
    fig.update_layout(
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(showticklabels=False),  # Hide x-axis labels untuk single box
        yaxis=dict(
            title=col,
            gridcolor='lightgray',
            gridwidth=1
        )
    )
    
    # Tambahkan annotation untuk statistik jika data tidak terlalu banyak
    if len(plot_data) <= 10000:
        try:
            stats = calculate_box_statistics(plot_data)
            
            # Tambahkan text annotations
            annotations = []
            y_positions = [stats['q1'], stats['median'], stats['q3']]
            labels = [f"Q1: {stats['q1']:.2f}", f"Median: {stats['median']:.2f}", f"Q3: {stats['q3']:.2f}"]
            
            for i, (y_pos, label) in enumerate(zip(y_positions, labels)):
                annotations.append(dict(
                    x=0.5,
                    y=y_pos,
                    xref="paper",
                    yref="y",
                    text=label,
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                    opacity=0.8
                ))
            
            fig.update_layout(annotations=annotations)
        except:
            pass  # Skip annotations jika error
    
    return fig

def calculate_box_statistics(data):
    """Hitung statistik box plot dengan numpy (lebih cepat)"""
    return {
        'min': np.min(data),
        'q1': np.percentile(data, 25),
        'median': np.median(data),
        'q3': np.percentile(data, 75),
        'max': np.max(data),
        'mean': np.mean(data),
        'std': np.std(data)
    }

def display_box_statistics(plot_data, col, original_size):
    """Tampilkan statistik box plot secara detail"""
    
    with st.expander("📊 Statistik Detail Box Plot", expanded=True):
        # Hitung statistik
        stats = calculate_box_statistics(plot_data)
        
        # Tampilkan metrik utama
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Jumlah Data", f"{len(plot_data):,}")
            st.metric("Minimum", f"{stats['min']:.2f}")
        with col2:
            st.metric("Q1 (25%)", f"{stats['q1']:.2f}")
            st.metric("Median", f"{stats['median']:.2f}")
        with col3:
            st.metric("Q3 (75%)", f"{stats['q3']:.2f}")
            st.metric("Maksimum", f"{stats['max']:.2f}")
        with col4:
            st.metric("Rata-rata", f"{stats['mean']:.2f}")
            st.metric("Std Dev", f"{stats['std']:.2f}")
        
        # Hitung IQR dan outlier
        iqr = stats['q3'] - stats['q1']
        lower_bound = stats['q1'] - 1.5 * iqr
        upper_bound = stats['q3'] + 1.5 * iqr
        
        outliers = plot_data[(plot_data < lower_bound) | (plot_data > upper_bound)]
        outlier_percentage = (len(outliers) / len(plot_data)) * 100
        
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("IQR", f"{iqr:.2f}")
        with col6:
            st.metric("Outliers", f"{len(outliers):,}")
        with col7:
            st.metric("% Outliers", f"{outlier_percentage:.1f}%")
        
        # Info skewness
        try:
            from scipy.stats import skew
            skewness = skew(plot_data)
            st.metric("Skewness", f"{skewness:.2f}")
            
            if abs(skewness) < 0.5:
                st.success("Distribusi: Mendekati normal")
            elif skewness > 0.5:
                st.warning("Distribusi: Right-skewed (positif)")
            elif skewness < -0.5:
                st.warning("Distribusi: Left-skewed (negatif)")
        except:
            st.info("Skewness: Tidak dapat dihitung")
        
        # Tampilkan outliers jika ada dan tidak terlalu banyak
        if len(outliers) > 0 and len(outliers) <= 50:
            st.markdown("**🔍 Daftar Outliers:**")
            outliers_df = pd.DataFrame({'Outlier Values': outliers.sort_values()})
            st.dataframe(outliers_df, use_container_width=True)

def show_box_optimization_info(original_size, processed_size, optimization_mode):
    """Tampilkan informasi optimasi"""
    reduction_pct = ((original_size - processed_size) / original_size) * 100 if original_size > 0 else 0
    
    if reduction_pct > 10:
        with st.expander("⚡ Info Optimasi Performa", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Original", f"{original_size:,}")
            with col2:
                st.metric("Data Diproses", f"{processed_size:,}")
            with col3:
                st.metric("Reduksi", f"{reduction_pct:.1f}%")
            
            optimization_strategies = {
                "Fast": "• ✅ **Systematic sampling**\n• ✅ **Outliers-only points**\n• ✅ **Minimal annotations**",
                "Balanced": "• ✅ **Stratified sampling**\n• ✅ **Smart point display**\n• ✅ **Basic statistics**",
                "Detailed": "• ✅ **Maximum data retention**\n• ✅ **Full annotations**\n• ✅ **Detailed analysis**"
            }
            
            st.info(f"**Mode {optimization_mode}**: {optimization_strategies.get(optimization_mode, 'Custom optimization')}")

def create_simple_box_fallback(df, col):
    """Fallback method untuk data yang bermasalah"""
    st.warning("Menggunakan metode fallback sederhana...")
    
    # Sample kecil untuk memastikan bisa render
    sample_data = df[col].dropna().head(2000)
    
    if len(sample_data) == 0:
        st.error("Tidak ada data valid")
        return
    
    fig = px.box(sample_data, y=col, title=f"Simple Box Plot: {col}")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Versi multiple box plots untuk perbandingan
def create_multi_box_plot(df, numeric_cols):
    """Multiple box plots untuk perbandingan"""
    
    data_size = len(df)
    
    selected_cols = st.multiselect(
        "Pilih kolom untuk perbandingan",
        numeric_cols[:10],  # Batasi pilihan
        default=numeric_cols[:min(5, len(numeric_cols))],
        key="multi_box_cols"
    )
    
    if len(selected_cols) >= 2:
        with st.spinner("🔄 Memproses multiple box plots..."):
            # Sampling untuk data besar
            if data_size > 50000:
                sample_df = df[selected_cols].dropna().sample(n=10000, random_state=42)
                st.info(f"📊 Menggunakan 10,000 sample dari {data_size:,} data points")
            else:
                sample_df = df[selected_cols].dropna()
            
            # Melt data untuk multiple box plots
            melted_df = sample_df.melt(var_name='Variable', value_name='Value')
            
            fig = px.box(
                melted_df, 
                x='Variable', 
                y='Value',
                title=f"Perbandingan Distribusi ({len(selected_cols)} variables)"
            )
            
            fig.update_traces(
                marker=dict(size=3, opacity=0.6),
                line=dict(width=1.2)
            )
            
            fig.update_layout(
                height=500,
                xaxis_title="Variable",
                yaxis_title="Value",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

def create_funnel_chart(df, numeric_cols, non_numeric_cols):
    
    stage_col = st.selectbox("Pilih kolom stage", non_numeric_cols, key="funnel_stage")
    value_col = st.selectbox("Pilih kolom nilai", numeric_cols, key="funnel_value")
    
    # Optimasi: Batasi jumlah data yang diproses
    max_stages = st.slider("Maksimum jumlah stage yang ditampilkan", 
                          min_value=5, max_value=20, value=10, key="funnel_max_stages")
    
    if stage_col and value_col:
        # Optimasi: Gunakan aggregation yang lebih efisien
        with st.spinner("Memproses data..."):
            # Group by dengan optimasi memori
            funnel_data = df.groupby(stage_col, observed=True)[value_col].sum()
            
            # Konversi ke DataFrame dan sort
            funnel_data = funnel_data.reset_index()
            funnel_data = funnel_data.nlargest(max_stages, value_col)
            
            # Cache data untuk performa
            @st.cache_data
            def create_funnel_figure(data, x_col, y_col, title):
                fig = px.funnel(data, x=x_col, y=y_col, title=title)
                fig.update_layout(
                    height=500,
                    showlegend=False,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                return fig
            
            fig = create_funnel_figure(
                funnel_data, 
                value_col, 
                stage_col, 
                f"Funnel Chart: {value_col} per {stage_col}"
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Tampilkan data summary
        with st.expander("📊 Lihat Data Summary"):
            st.dataframe(funnel_data.style.format({value_col: "{:,.0f}"}), use_container_width=True)
            
        with st.expander("ℹ️ Keterangan Funnel Chart"):
            st.markdown(f"""
            **Funnel Chart** menampilkan proses bertahap dengan attrition.
            
            **Statistik:**
            - Total {stage_col}: {len(funnel_data)}
            - Total {value_col}: {funnel_data[value_col].sum():,.0f}
            - Stage tertinggi: {funnel_data.iloc[0][stage_col]} ({funnel_data.iloc[0][value_col]:,.0f})
            - Stage terendah: {funnel_data.iloc[-1][stage_col]} ({funnel_data.iloc[-1][value_col]:,.0f})
            
            **Kelebihan**: Visualisasi proses dan konversi yang jelas
            **Kekurangan**: Hanya untuk data sequential
            **Penggunaan**: Sales funnel, conversion analysis, process flow
            """)

# Alternatif: Versi dengan sampling untuk data yang sangat besar
def create_funnel_chart_optimized(df, numeric_cols, non_numeric_cols):
    
    stage_col = st.selectbox("Pilih kolom stage", non_numeric_cols, key="funnel_stage_opt")
    value_col = st.selectbox("Pilih kolom nilai", numeric_cols, key="funnel_value_opt")
    
    # Optimasi tambahan untuk data sangat besar
    sample_size = st.slider("Sample size (%)", 10, 100, 50, key="funnel_sample")
    
    if stage_col and value_col:
        # Sampling data untuk performa
        if len(df) > 10000:
            df_sampled = df.sample(frac=sample_size/100, random_state=42)
            st.info(f"Data disampling: {len(df_sampled):,} dari {len(df):,} records ({sample_size}%)")
        else:
            df_sampled = df
        
        with st.spinner("Memproses data dengan optimasi..."):
            # Aggregasi yang lebih cepat
            funnel_data = (df_sampled
                         .groupby(stage_col, observed=True)[value_col]
                         .sum()
                         .nlargest(15)  # Batasi langsung di aggregation
                         .reset_index())
            
            # Plot yang lebih ringan
            fig = px.funnel(funnel_data, x=value_col, y=stage_col,
                          title=f"Funnel Chart: {value_col} per {stage_col}")
            
            fig.update_layout(
                height=500,
                showlegend=False,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            # Nonaktifkan beberapa fitur plotly untuk performa
            st.plotly_chart(fig, use_container_width=True, 
                          config={'displayModeBar': False, 'responsive': True})

def create_wordcloud(df, non_numeric_cols):
    
    # Deteksi ukuran data
    data_size = len(df)
    if data_size > 100000:
        st.info(f"⚡ Mode Optimasi: Data besar ({data_size:,} rows) - Menggunakan sampling otomatis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        text_col = st.selectbox("Pilih kolom teks", non_numeric_cols, key="wordcloud_col")
    
    with col2:
        # Pengaturan optimasi
        optimization_mode = st.selectbox(
            "Mode Optimasi",
            ["Auto", "Fast", "Balanced", "Detailed"],
            index=0 if data_size > 50000 else 2,
            key="wc_optim"
        )
    
    with col3:
        max_words = st.slider(
            "Maksimum kata",
            min_value=50,
            max_value=500,
            value=150 if data_size > 50000 else 200,
            key="wc_max_words"
        )
    
    # Pengaturan lanjutan
    with st.expander("⚙️ Pengaturan Lanjutan", expanded=False):
        col4, col5, col6 = st.columns(3)
        with col4:
            sample_size = st.slider(
                "Sample data size",
                min_value=1000,
                max_value=50000,
                value=10000 if data_size > 50000 else 20000,
                key="wc_sample_size"
            )
        with col5:
            width = st.slider(
                "Lebar word cloud",
                min_value=400,
                max_value=1200,
                value=800,
                key="wc_width"
            )
        with col6:
            height = st.slider(
                "Tinggi word cloud",
                min_value=200,
                max_value=800,
                value=400,
                key="wc_height"
            )
    
    # Custom stopwords dan preferences
    with st.expander("🔧 Pengaturan Teks", expanded=False):
        col7, col8 = st.columns(2)
        with col7:
            remove_stopwords = st.checkbox("Hapus stopwords", value=True, key="wc_stopwords")
            include_numbers = st.checkbox("Sertakan angka", value=False, key="wc_numbers")
        with col8:
            language = st.selectbox(
                "Bahasa stopwords",
                ["English", "Indonesian", "None"],
                index=0,
                key="wc_language"
            )
        custom_stopwords = st.text_area(
            "Stopwords kustom (pisahkan dengan koma)",
            value="",
            help="Tambahkan kata-kata yang ingin dihilangkan dari word cloud"
        )
    
    if text_col:
        try:
            with st.spinner("🔄 Memproses teks dan membuat word cloud..."):
                # OPTIMASI 1: Sampling data untuk data besar
                processed_text = optimize_text_data(df, text_col, data_size, optimization_mode, sample_size)
                
                if not processed_text or processed_text.strip() == "":
                    st.warning("Tidak ada teks yang valid untuk ditampilkan")
                    return
                
                # OPTIMASI 2: Preprocessing teks yang efisien
                cleaned_text = preprocess_text(
                    processed_text, 
                    remove_stopwords, 
                    language, 
                    custom_stopwords,
                    include_numbers
                )
                
                if not cleaned_text or cleaned_text.strip() == "":
                    st.warning("Tidak ada kata yang tersisa setelah preprocessing")
                    return
                
                # OPTIMASI 3: Buat word cloud dengan konfigurasi optimal
                fig = create_optimized_wordcloud(
                    cleaned_text, 
                    max_words, 
                    width, 
                    height,
                    optimization_mode
                )
                
                st.pyplot(fig, use_container_width=True)
                
                # OPTIMASI 4: Tampilkan analisis teks
                display_text_analysis(cleaned_text, processed_text, max_words)
                
                # Tampilkan info optimasi
                show_wordcloud_optimization_info(data_size, len(processed_text.split()), optimization_mode)
                
        except Exception as e:
            st.error(f"Error membuat word cloud: {str(e)}")
            # Fallback ke metode sederhana
            create_simple_wordcloud_fallback(df, text_col)

def optimize_text_data(df, text_col, data_size, optimization_mode, sample_size):
    """Optimasi data teks dengan sampling yang tepat"""
    
    # Filter data yang valid
    text_data = df[text_col].astype(str).dropna()
    
    if len(text_data) == 0:
        return ""
    
    # Tentukan target sample size
    target_sizes = {
        "Auto": min(sample_size, data_size),
        "Fast": min(5000, data_size),
        "Balanced": min(20000, data_size),
        "Detailed": min(50000, data_size)
    }
    
    target_size = target_sizes[optimization_mode]
    
    # Jika data lebih besar dari target, lakukan sampling
    if len(text_data) > target_size:
        if optimization_mode == "Fast":
            # Ambil sample acak
            sampled_data = text_data.sample(n=target_size, random_state=42)
        elif optimization_mode == "Balanced":
            # Prioritaskan teks yang lebih panjang (lebih informatif)
            text_lengths = text_data.str.len()
            # Ambil campuran: 70% teks terpanjang, 30% random
            n_longest = int(target_size * 0.7)
            n_random = target_size - n_longest
            
            longest_texts = text_data.iloc[text_lengths.nlargest(n_longest).index]
            random_texts = text_data.sample(n=n_random, random_state=42)
            
            sampled_data = pd.concat([longest_texts, random_texts])
        else:
            # Random sampling untuk mode lain
            sampled_data = text_data.sample(n=target_size, random_state=42)
        
        return ' '.join(sampled_data)
    
    return ' '.join(text_data)

def preprocess_text(text, remove_stopwords=True, language="English", custom_stopwords="", include_numbers=False):
    """Preprocessing teks yang efisien"""
    import re
    from collections import Counter
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers jika tidak diinginkan
    if not include_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Split into words
    words = text.split()
    
    # Remove stopwords
    if remove_stopwords:
        stopwords_set = get_stopwords_set(language, custom_stopwords)
        words = [word for word in words if word not in stopwords_set and len(word) > 2]
    
    # Remove very short and very long words
    words = [word for word in words if 2 < len(word) < 20]
    
    return ' '.join(words)

def get_stopwords_set(language, custom_stopwords):
    """Dapatkan set stopwords berdasarkan bahasa"""
    stopwords_set = set()
    
    # Basic English stopwords
    basic_stopwords = {
        'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with', 'as', 'by', 
        'at', 'an', 'be', 'this', 'that', 'it', 'are', 'from', 'or', 'but', 'not',
        'you', 'your', 'we', 'our', 'they', 'their', 'i', 'me', 'my', 'he', 'him',
        'his', 'she', 'her', 'its', 'us', 'them', 'what', 'which', 'who', 'when',
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'can', 'will', 'just', 'should', 'now'
    }
    
    # Indonesian stopwords
    indonesian_stopwords = {
        'yang', 'dan', 'di', 'dengan', 'ini', 'itu', 'dari', 'untuk', 'pada', 'ke',
        'dalam', 'tidak', 'akan', 'ada', 'atau', 'juga', 'bisa', 'saya', 'kita',
        'mereka', 'dia', 'kamu', 'kami', 'adalah', 'harus', 'sudah', 'belum',
        'pernah', 'selalu', 'sering', 'kadang', 'mungkin', 'boleh', 'harus',
        'perlu', 'bisa', 'dapat', 'boleh', 'harus', 'perlu', 'bisa', 'dapat'
    }
    
    if language == "English":
        stopwords_set.update(basic_stopwords)
    elif language == "Indonesian":
        stopwords_set.update(indonesian_stopwords)
    
    # Add custom stopwords
    if custom_stopwords:
        custom_words = [word.strip().lower() for word in custom_stopwords.split(',')]
        stopwords_set.update(custom_words)
    
    return stopwords_set

def create_optimized_wordcloud(text, max_words, width, height, optimization_mode):
    """Buat word cloud yang dioptimalkan"""
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    # Konfigurasi berdasarkan mode optimasi
    if optimization_mode == "Fast":
        colormap = 'viridis'
        background_color = 'white'
        relative_scaling = 0.5
        min_font_size = 8
    elif optimization_mode == "Balanced":
        colormap = 'plasma'
        background_color = 'white'
        relative_scaling = 0.8
        min_font_size = 6
    else:  # Detailed
        colormap = 'inferno'
        background_color = 'black'
        relative_scaling = 1.0
        min_font_size = 4
    
    # Buat word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        colormap=colormap,
        relative_scaling=relative_scaling,
        min_font_size=min_font_size,
        random_state=42
    ).generate(text)
    
    # Buat figure
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    # Optimasi layout
    plt.tight_layout(pad=0)
    
    return fig

def display_text_analysis(cleaned_text, original_text, max_words):
    """Tampilkan analisis teks"""
    
    with st.expander("📊 Analisis Teks", expanded=False):
        from collections import Counter
        import re
        
        # Hitung statistik dasar
        original_words = re.findall(r'\b\w+\b', original_text.lower())
        cleaned_words = cleaned_text.split()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Kata Original", f"{len(original_words):,}")
        with col2:
            st.metric("Total Kata Setelah Cleaning", f"{len(cleaned_words):,}")
        with col3:
            unique_ratio = len(set(cleaned_words)) / len(cleaned_words) if cleaned_words else 0
            st.metric("Unique Words Ratio", f"{unique_ratio:.2f}")
        with col4:
            avg_word_length = np.mean([len(word) for word in cleaned_words]) if cleaned_words else 0
            st.metric("Rata-rata Panjang Kata", f"{avg_word_length:.1f}")
        
        # Tampilkan top words
        if cleaned_words:
            word_freq = Counter(cleaned_words)
            top_words = word_freq.most_common(20)
            
            st.markdown("**🔝 Kata Paling Sering Muncul:**")
            
            # Buat chart untuk top words
            words, counts = zip(*top_words[:10])
            
            fig, ax = plt.subplots(figsize=(10, 4))
            y_pos = np.arange(len(words))
            ax.barh(y_pos, counts, color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.invert_yaxis()
            ax.set_xlabel('Frekuensi')
            ax.set_title('10 Kata Paling Sering Muncul')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Tabel untuk semua top words
            top_words_df = pd.DataFrame(top_words, columns=['Kata', 'Frekuensi'])
            st.dataframe(top_words_df, use_container_width=True)

def show_wordcloud_optimization_info(original_size, processed_word_count, optimization_mode):
    """Tampilkan informasi optimasi"""
    
    with st.expander("⚡ Info Optimasi Performa", expanded=False):
        st.metric("Data Points Original", f"{original_size:,}")
        st.metric("Kata Diproses", f"{processed_word_count:,}")
        
        optimization_strategies = {
            "Fast": "• ✅ **Aggressive sampling**\n• ✅ **Basic preprocessing**\n• ✅ **Fast rendering**",
            "Balanced": "• ✅ **Smart sampling** (prioritize long texts)\n• ✅ **Advanced preprocessing**\n• ✅ **Quality rendering**",
            "Detailed": "• ✅ **Maximum data retention**\n• ✅ **Comprehensive preprocessing**\n• ✅ **High-quality output**"
        }
        
        st.info(f"**Mode {optimization_mode}**: {optimization_strategies.get(optimization_mode, 'Custom optimization')}")

def create_simple_wordcloud_fallback(df, text_col):
    """Fallback method untuk data yang bermasalah"""
    st.warning("Menggunakan metode fallback sederhana...")
    
    # Ambil sample kecil
    sample_data = df[text_col].astype(str).dropna().head(1000)
    text = ' '.join(sample_data)
    
    if text.strip():
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        wordcloud = WordCloud(width=600, height=300, background_color='white', max_words=100).generate(text)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.error("Tidak ada teks yang valid")

# Versi ultra-ringan untuk data ekstrem
def create_ultra_fast_wordcloud(df, non_numeric_cols):
    """Versi ultra-ringan untuk data > 500k rows"""
    
    col1, col2 = st.columns(2)
    with col1:
        text_col = st.selectbox("Pilih kolom teks", non_numeric_cols[:5], key="ultra_wc_col")
    with col2:
        max_words = st.slider("Maks kata", 50, 200, 100, key="ultra_wc_words")
    
    if text_col:
        # Sampling sangat agresif
        sample_data = df[text_col].astype(str).dropna()
        if len(sample_data) > 5000:
            sample_data = sample_data.sample(n=5000, random_state=42)
        
        text = ' '.join(sample_data)
        
        if text.strip():
            # Preprocessing sederhana
            text = text.lower()
            words = text.split()
            words = [word for word in words if len(word) > 3 and len(word) < 15]
            processed_text = ' '.join(words)
            
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            
            wordcloud = WordCloud(
                width=600, 
                height=300, 
                background_color='white',
                max_words=max_words,
                colormap='viridis'
            ).generate(processed_text)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
            st.info(f"📊 Ultra-Fast Mode: 5,000 samples, {max_words} words")

def create_gantt_chart(df):
    
    # Deteksi ukuran data
    data_size = len(df)
    if data_size > 10000:
        st.info(f"⚡ Mode Optimasi: Data besar ({data_size:,} rows) - Menggunakan sampling otomatis")
    
    # Deteksi kolom yang tersedia
    date_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
    text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    st.info(f"🔍 **Kolom yang terdeteksi:**")
    st.write(f"- 📅 Datetime: {date_cols}")
    st.write(f"- 📝 Teks: {text_cols}")
    st.write(f"- 🔢 Numerik: {numeric_cols}")
    
    # SOLUSI: Jika hanya ada 1 kolom datetime, berikan opsi alternatif
    if len(date_cols) == 1:
        st.warning("""
        ⚠️ **Hanya 1 kolom datetime terdeteksi.** 
        Untuk Gantt chart, dibutuhkan 2 kolom datetime (start dan end date).
        
        **Solusi yang tersedia:**
        1. **Gunakan durasi tetap** - Tambahkan kolom end date berdasarkan durasi
        2. **Gunakan kolom numerik** - Konversi ke timeline relatif
        3. **Employee Timeline** - Visualisasi berdasarkan hire date saja
        """)
        
        selected_solution = st.radio(
            "Pilih tipe visualisasi:",
            ["Employee Timeline", "Duration-based Gantt", "Relative Timeline"],
            key="gantt_solution"
        )
        
        if selected_solution == "Employee Timeline":
            create_employee_timeline(df, date_cols[0], text_cols)
            return
        elif selected_solution == "Duration-based Gantt":
            create_duration_gantt(df, date_cols[0], text_cols, numeric_cols)
            return
        else:  # Relative Timeline
            create_relative_timeline(df, numeric_cols, text_cols)
            return
    
    # [Kode untuk kasus dengan 2+ kolom datetime...]
    st.success("✅ Dua atau lebih kolom datetime terdeteksi - dapat membuat Gantt chart standar")
    create_standard_gantt(df, date_cols, text_cols)

def create_standard_gantt(df, date_cols, text_cols):
    """Buat Gantt chart standar ketika ada 2+ kolom datetime"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_col = st.selectbox(
            "Pilih kolom start date",
            date_cols,
            key="start_date"
        )
    
    with col2:
        end_col = st.selectbox(
            "Pilih kolom end date",
            [col for col in date_cols if col != start_col],
            key="end_date"
        )
    
    with col3:
        task_col = st.selectbox(
            "Pilih kolom task/nama",
            text_cols,
            key="task_col"
        )
    
    # Filter data
    gantt_data = df[[start_col, end_col, task_col]].dropna()
    
    if len(gantt_data) == 0:
        st.error("❌ Tidak ada data valid setelah menghapus nilai kosong")
        return
    
    # Batasi jumlah data untuk performa
    if len(gantt_data) > 100:
        st.warning(f"⚠️ Data dibatasi dari {len(gantt_data)} menjadi 100 baris untuk performa")
        gantt_data = gantt_data.head(100)
    
    # Buat Gantt chart dengan Plotly
    fig = px.timeline(
        gantt_data,
        x_start=start_col,
        x_end=end_col,
        y=task_col,
        title="Gantt Chart"
    )
    
    # Update layout untuk responsif
    fig.update_layout(
        height=max(400, len(gantt_data) * 25),
        showlegend=False,
        xaxis_title="Timeline",
        yaxis_title="Tasks",
        margin=dict(l=50, r=50, t=80, b=50),
        autosize=True
    )
    
    # Rotasi label y-axis untuk readability
    fig.update_yaxes(tickangle=0)
    
    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})

def create_employee_timeline(df, date_col, text_cols):
    """Buat employee timeline berdasarkan hire date"""
    st.subheader("👥 Employee Timeline")
    
    st.info("""
    **Employee Timeline** menampilkan karyawan berdasarkan tanggal bergabung.
    Setiap bar mewakili 1 karyawan dengan posisi berdasarkan hire date.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        category_col = st.selectbox(
            "Pilih kolom untuk kategori warna",
            ["None"] + text_cols,
            key="timeline_category"
        )
    
    with col2:
        max_employees = st.slider(
            "Maksimum karyawan ditampilkan",
            min_value=50,
            max_value=1000,
            value=min(200, len(df)),
            key="timeline_max_employees"
        )
    
    # Siapkan data
    timeline_data = df[[date_col] + ([category_col] if category_col != "None" else [])].copy()
    timeline_data = timeline_data.dropna().head(max_employees)
    
    if len(timeline_data) == 0:
        st.error("❌ Tidak ada data valid untuk timeline")
        return
    
    # Buat timeline dengan cara yang lebih sederhana
    fig = go.Figure()
    
    # Sort data by date
    timeline_data = timeline_data.sort_values(date_col)
    
    # Buat scatter plot untuk timeline
    y_positions = list(range(len(timeline_data)))
    
    if category_col != "None":
        # Group by category untuk warna berbeda
        categories = timeline_data[category_col].unique()
        colors = px.colors.qualitative.Set3
        
        for i, category in enumerate(categories):
            category_data = timeline_data[timeline_data[category_col] == category]
            
            fig.add_trace(go.Scatter(
                x=category_data[date_col],
                y=list(range(len(category_data))),
                mode='markers',
                marker=dict(
                    size=15,
                    color=colors[i % len(colors)],
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                name=str(category),
                hovertemplate=(
                    f"<b>{category_col}: {category}</b><br>"
                    f"Date: %{{x|%d %b %Y}}<br>"
                    f"<extra></extra>"
                )
            ))
    else:
        # Semua titik dengan warna sama
        fig.add_trace(go.Scatter(
            x=timeline_data[date_col],
            y=y_positions,
            mode='markers',
            marker=dict(
                size=10,
                color='lightblue',
                line=dict(width=1, color='navy')
            ),
            hovertemplate=(
                "<b>Employee</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "<extra></extra>"
            ),
            name="Employees"
        ))
    
    # Layout responsif
    height = max(400, len(timeline_data) * 8)
    fig.update_layout(
        height=min(height, 800),
        title=f"Employee Timeline - {len(timeline_data)} Employees",
        xaxis_title="Hire Date",
        yaxis_title="Employee Index",
        showlegend=(category_col != "None"),
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='closest'
    )
    
    # Sembunyikan y-axis labels jika terlalu banyak
    if len(timeline_data) > 50:
        fig.update_yaxes(showticklabels=False)
    
    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    # Statistik
    with st.expander("📊 Employee Statistics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Employees", len(timeline_data))
        with col2:
            earliest_hire = timeline_data[date_col].min()
            st.metric("Earliest Hire", earliest_hire.strftime('%d %b %Y'))
        with col3:
            latest_hire = timeline_data[date_col].max()
            st.metric("Latest Hire", latest_hire.strftime('%d %b %Y'))
        
        if category_col != "None":
            st.write("**Distribution by Category:**")
            category_counts = timeline_data[category_col].value_counts()
            st.dataframe(category_counts, use_container_width=True)

def create_duration_gantt(df, date_col, text_cols, numeric_cols):
    """Buat Gantt chart dengan durasi dari kolom numerik"""
    st.subheader("⏱️ Duration-based Gantt Chart")
    
    st.info("""
    **Duration-based Gantt** menggunakan hire date sebagai start date 
    dan menambahkan durasi dari kolom numerik untuk membuat end date.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        task_col = st.selectbox(
            "Pilih kolom task/nama",
            text_cols,
            key="duration_task"
        )
    
    with col2:
        duration_col = st.selectbox(
            "Pilih kolom durasi (numerik)",
            ["Fixed Duration"] + numeric_cols,
            key="duration_col"
        )
    
    with col3:
        if duration_col == "Fixed Duration":
            fixed_duration = st.number_input(
                "Durasi tetap (hari)",
                min_value=1,
                max_value=3650,
                value=365,
                key="fixed_duration"
            )
        else:
            duration_multiplier = st.selectbox(
                "Satuan durasi",
                ["days", "months", "years"],
                key="duration_unit"
            )
    
    # Siapkan data
    gantt_data = []
    
    for idx, row in df.iterrows():
        if pd.notna(row[date_col]) and pd.notna(row[task_col]):
            start_date = row[date_col]
            
            # Tentukan end date berdasarkan pilihan
            if duration_col == "Fixed Duration":
                end_date = start_date + pd.Timedelta(days=fixed_duration)
                duration_days = fixed_duration
            else:
                if pd.notna(row[duration_col]):
                    duration_val = float(row[duration_col])
                    if duration_multiplier == "days":
                        end_date = start_date + pd.Timedelta(days=duration_val)
                        duration_days = duration_val
                    elif duration_multiplier == "months":
                        end_date = start_date + pd.DateOffset(months=duration_val)
                        duration_days = (end_date - start_date).days
                    else:  # years
                        end_date = start_date + pd.DateOffset(years=duration_val)
                        duration_days = (end_date - start_date).days
                else:
                    continue
            
            gantt_data.append({
                'Task': str(row[task_col]),
                'Start': start_date,
                'Finish': end_date,
                'Duration': duration_days
            })
    
    if not gantt_data:
        st.error("❌ Tidak ada data valid untuk Gantt chart")
        return
    
    # Konversi ke DataFrame
    gantt_df = pd.DataFrame(gantt_data)
    
    # Batasi jumlah data untuk performa
    if len(gantt_df) > 100:
        st.warning(f"⚠️ Data dibatasi dari {len(gantt_df)} menjadi 100 baris untuk performa")
        gantt_df = gantt_df.head(100)
    
    # Buat Gantt chart dengan Plotly timeline
    try:
        fig = px.timeline(
            gantt_df,
            x_start="Start",
            x_end="Finish",
            y="Task",
            title="Duration-based Gantt Chart"
        )
        
        # Update layout untuk responsif
        height = max(400, len(gantt_df) * 25)
        fig.update_layout(
            height=min(height, 800),
            showlegend=False,
            xaxis_title="Timeline",
            yaxis_title="Tasks",
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
        
    except Exception as e:
        st.error(f"❌ Error membuat chart: {str(e)}")
        # Fallback: tampilkan data sebagai tabel
        st.write("**Data Gantt Chart:**")
        st.dataframe(gantt_df, use_container_width=True)
    
    # Statistik
    with st.expander("📊 Duration Statistics"):
        durations = gantt_df['Duration'].tolist()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Items", len(gantt_df))
        with col2:
            st.metric("Avg Duration", f"{np.mean(durations):.1f} days")
        with col3:
            st.metric("Min Duration", f"{np.min(durations):.1f} days")
        with col4:
            st.metric("Max Duration", f"{np.max(durations):.1f} days")

def create_relative_timeline(df, numeric_cols, text_cols):
    """Buat timeline relatif berdasarkan kolom numerik"""
    st.subheader("📊 Relative Timeline Chart")
    
    st.info("""
    **Relative Timeline** menggunakan kolom numerik untuk membuat timeline relatif.
    Berguna untuk membandingkan metrik antar kategori.
    """)
    
    if not numeric_cols:
        st.error("❌ Tidak ada kolom numerik yang tersedia untuk relative timeline")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        value_col = st.selectbox(
            "Pilih kolom nilai",
            numeric_cols,
            key="relative_value"
        )
    
    with col2:
        category_col = st.selectbox(
            "Pilih kolom kategori",
            ["None"] + text_cols,
            key="relative_category"
        )
    
    with col3:
        max_items = st.slider(
            "Maksimum items",
            min_value=20,
            max_value=200,
            value=min(50, len(df)),
            key="relative_max_items"
        )
    
    # Siapkan data
    if category_col != "None":
        plot_data = df[[value_col, category_col]].dropna()
    else:
        plot_data = df[[value_col]].dropna()
        plot_data['Index'] = range(len(plot_data))
        category_col = 'Index'
    
    plot_data = plot_data.nlargest(max_items, value_col)
    
    if len(plot_data) == 0:
        st.error("❌ Tidak ada data valid")
        return
    
    # Buat bar chart horizontal (simulasi timeline)
    try:
        fig = px.bar(
            plot_data,
            x=value_col,
            y=category_col,
            color=category_col if category_col != "None" and category_col != 'Index' else None,
            orientation='h',
            title=f"Relative Timeline - {value_col}"
        )
        
        height = max(400, len(plot_data) * 20)
        fig.update_layout(
            height=min(height, 800),
            showlegend=False,
            xaxis_title=value_col,
            yaxis_title=category_col if category_col != "None" else "Items",
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
        
    except Exception as e:
        st.error(f"❌ Error membuat chart: {str(e)}")
        st.write("**Data:**")
        st.dataframe(plot_data, use_container_width=True)

# Tambahkan CSS untuk styling responsif
st.markdown("""
<style>
    /* Responsive radio buttons */
    .stRadio [role=radiogroup]{
        align-items: center;
        gap: 10px;
    }
    
    .stRadio [data-testid=stMarkdownContainer] > p {
        font-size: 16px;
    }
    
    /* Responsive containers */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Responsive charts */
    .js-plotly-plot .plotly .main-svg {
        width: 100% !important;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .stRadio [role=radiogroup] {
            flex-direction: column;
            align-items: flex-start;
        }
        
        .stSelectbox, .stSlider, .stNumberInput {
            min-width: 100% !important;
        }
    }
    
    /* Better spacing */
    .stExpander {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def create_map_chart(df):
    
    # Optimasi: Cache deteksi kolom
    @st.cache_data
    def detect_geo_columns(df):
        geo_patterns = ['lat', 'latitude', 'lon', 'long', 'longitude', 'country', 'state', 'city', 'region', 'province', 'kota', 'kabupaten', 'address', 'location']
        return [col for col in df.columns if any(geo in col.lower() for geo in geo_patterns)]
    
    possible_geo_cols = detect_geo_columns(df)
    
    if possible_geo_cols:
        st.success(f"✅ Kolom geografis terdeteksi: {', '.join(possible_geo_cols)}")
        
        # Kategorikan kolom dengan caching
        @st.cache_data
        def categorize_columns(_possible_geo_cols):
            lat_cols = [col for col in _possible_geo_cols if any(pat in col.lower() for pat in ['lat', 'latitude'])]
            lon_cols = [col for col in _possible_geo_cols if any(pat in col.lower() for pat in ['lon', 'long', 'longitude'])]
            name_cols = [col for col in _possible_geo_cols if any(pat in col.lower() for pat in ['country', 'state', 'city', 'region', 'province', 'kota', 'kabupaten', 'name'])]
            return lat_cols, lon_cols, name_cols
        
        lat_cols, lon_cols, name_cols = categorize_columns(possible_geo_cols)
        
        # Pilih kolom untuk mapping
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if lat_cols:
                lat_col = st.selectbox("Pilih kolom Latitude", lat_cols, key="map_lat")
            else:
                st.warning("Tidak ada kolom latitude terdeteksi")
                lat_col = None
                
        with col2:
            if lon_cols:
                lon_col = st.selectbox("Pilih kolom Longitude", lon_cols, key="map_lon")
            else:
                st.warning("Tidak ada kolom longitude terdeteksi")
                lon_col = None
                
        with col3:
            if name_cols:
                name_col = st.selectbox("Pilih kolom Nama Lokasi", name_cols, key="map_name")
            else:
                name_col = None
        
        # Optimasi: Sampling data untuk dataset besar
        sample_size = st.slider("Jumlah sampel data untuk peta", 
                               min_value=100, 
                               max_value=min(5000, len(df)), 
                               value=min(1000, len(df)),
                               key="map_sample")
        
        # Filter data yang valid dengan sampling
        if lat_col and lon_col:
            valid_data = df[(pd.notna(df[lat_col])) & (pd.notna(df[lon_col]))].copy()
            
            if len(valid_data) > 0:
                # Sampling untuk dataset besar
                if len(valid_data) > sample_size:
                    valid_data = valid_data.sample(n=sample_size, random_state=42)
                    st.info(f"📊 Menampilkan {sample_size} sampel acak dari {len(valid_data)} data valid")
                else:
                    st.success(f"📊 Menampilkan {len(valid_data)} titik data")
                
                # Progress bar untuk proses yang lama
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Buat peta
                try:
                    import folium
                    from streamlit_folium import folium_static
                    
                    status_text.text("Membuat peta...")
                    
                    # Hitung center map dengan caching
                    @st.cache_data
                    def calculate_center(_data, lat_col, lon_col):
                        return _data[lat_col].mean(), _data[lon_col].mean()
                    
                    center_lat, center_lon = calculate_center(valid_data, lat_col, lon_col)
                    
                    # Buat peta dasar
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                    
                    # Optimasi: Batasi jumlah marker atau gunakan clustering untuk data besar
                    if len(valid_data) > 500:
                        from folium.plugins import MarkerCluster
                        marker_cluster = MarkerCluster().add_to(m)
                    
                    # Tambahkan markers dengan progress update
                    total_rows = len(valid_data)
                    for idx, row in valid_data.iterrows():
                        if idx % 100 == 0:  # Update progress setiap 100 rows
                            progress_bar.progress(min((idx + 1) / total_rows, 1.0))
                        
                        popup_text = f"Lokasi {idx+1}"
                        if name_col and pd.notna(row[name_col]):
                            popup_text = f"{row[name_col]}"
                        
                        marker = folium.Marker(
                            [row[lat_col], row[lon_col]],
                            popup=popup_text,
                            tooltip=f"Click untuk detail"
                        )
                        
                        # Tambahkan ke cluster jika data banyak, langsung ke map jika sedikit
                        if len(valid_data) > 500:
                            marker.add_to(marker_cluster)
                        else:
                            marker.add_to(m)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Peta selesai dibuat!")
                    
                    # Tampilkan peta
                    folium_static(m, width=700, height=500)
                    
                    # Tampilkan data table dengan pagination
                    with st.expander("📋 Lihat Data Peta"):
                        display_cols = [lat_col, lon_col]
                        if name_col:
                            display_cols.append(name_col)
                        
                        # Pagination untuk data besar
                        page_size = 20
                        total_pages = max(1, len(valid_data) // page_size)
                        page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1)
                        
                        start_idx = (page - 1) * page_size
                        end_idx = min(start_idx + page_size, len(valid_data))
                        
                        st.dataframe(valid_data[display_cols].iloc[start_idx:end_idx])
                        st.caption(f"Menampilkan data {start_idx + 1}-{end_idx} dari {len(valid_data)}")
                        
                except ImportError:
                    st.error("❌ Library peta tidak tersedia. Install: pip install folium streamlit-folium")
                    
            else:
                st.error("❌ Tidak ada data dengan koordinat yang valid")
                
        else:
            st.warning("⚠️ Pilih kolom latitude dan longitude untuk menampilkan peta")
            
    else:
        st.warning("""
        ⚠️ Tidak terdeteksi kolom geografis. 
        
        **Untuk menampilkan peta, data harus mengandung:**
        - Kolom latitude (contoh: lat, latitude) 
        - Kolom longitude (contoh: lon, long, longitude)
        - Opsional: Kolom nama lokasi (country, state, city, region, etc.)
        """)

def create_flow_map(df):
    
    # Optimasi: Cache deteksi kolom
    @st.cache_data
    def detect_flow_columns(df):
        flow_patterns = ['lat', 'lon', 'long', 'latitude', 'longitude', 'origin', 'destination', 'from', 'to', 'source', 'target']
        return [col for col in df.columns if any(flow in col.lower() for flow in flow_patterns)]
    
    possible_flow_cols = detect_flow_columns(df)
    
    if possible_flow_cols:
        st.success(f"✅ Kolom flow map terdeteksi: {', '.join(possible_flow_cols)}")
        
        # Kategorikan kolom dengan caching
        @st.cache_data
        def categorize_flow_columns(_possible_flow_cols):
            origin_lat_cols = [col for col in _possible_flow_cols if any(pat in col.lower() for pat in ['origin_lat', 'from_lat', 'source_lat', 'lat_origin'])]
            origin_lon_cols = [col for col in _possible_flow_cols if any(pat in col.lower() for pat in ['origin_lon', 'from_lon', 'source_lon', 'lon_origin'])]
            dest_lat_cols = [col for col in _possible_flow_cols if any(pat in col.lower() for pat in ['dest_lat', 'to_lat', 'target_lat', 'lat_dest'])]
            dest_lon_cols = [col for col in _possible_flow_cols if any(pat in col.lower() for pat in ['dest_lon', 'to_lon', 'target_lon', 'lon_dest'])]
            return origin_lat_cols, origin_lon_cols, dest_lat_cols, dest_lon_cols
        
        origin_lat_cols, origin_lon_cols, dest_lat_cols, dest_lon_cols = categorize_flow_columns(possible_flow_cols)
        
        value_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📍 Origin Coordinates**")
            origin_lat = st.selectbox("Origin Latitude", origin_lat_cols if origin_lat_cols else possible_flow_cols, key="flow_origin_lat")
            origin_lon = st.selectbox("Origin Longitude", origin_lon_cols if origin_lon_cols else possible_flow_cols, key="flow_origin_lon")
            
        with col2:
            st.write("**🎯 Destination Coordinates**")
            dest_lat = st.selectbox("Destination Latitude", dest_lat_cols if dest_lat_cols else possible_flow_cols, key="flow_dest_lat")
            dest_lon = st.selectbox("Destination Longitude", dest_lon_cols if dest_lon_cols else possible_flow_cols, key="flow_dest_lon")
        
        # Pilih value column
        value_col = st.selectbox("📊 Kolom Value (untuk ketebalan flow)", [""] + value_cols, key="flow_value")
        
        # Customization options
        st.write("**🎨 Kustomisasi Tampilan**")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            flow_style = st.selectbox("Gaya Garis", ["Solid", "Dashed", "Dotted", "Animated"], key="flow_style")
            line_width = st.slider("Ketebalan Garis Dasar", 1, 10, 3, key="line_width")
            
        with col4:
            color_scheme = st.selectbox("Skema Warna", [
                "Viridis", "Plasma", "Inferno", "Magma", 
                "Rainbow", "Jet", "Hot", "Cool", "Red-Blue"
            ], key="color_scheme")
            
        with col5:
            map_style = st.selectbox("Style Peta", [
                "natural earth", "orthographic", "equirectangular", 
                "mercator", "azimuthal equal area"
            ], key="map_style")
        
        # Optimasi: Sampling untuk flow map
        flow_sample_size = st.slider("🚢 Jumlah sampel aliran untuk ditampilkan", 
                                    min_value=50, 
                                    max_value=min(1000, len(df)), 
                                    value=min(200, len(df)),
                                    key="flow_sample")
        
        # Validasi data
        if origin_lat and origin_lon and dest_lat and dest_lon:
            # Filter data valid dengan sampling
            valid_data = df[
                (pd.notna(df[origin_lat])) & (pd.notna(df[origin_lon])) &
                (pd.notna(df[dest_lat])) & (pd.notna(df[dest_lon]))
            ].copy()
            
            if len(valid_data) > 0:
                # Sampling untuk dataset besar
                if len(valid_data) > flow_sample_size:
                    valid_data = valid_data.sample(n=flow_sample_size, random_state=42)
                    st.info(f"📊 Menampilkan {flow_sample_size} sampel acak dari {len(valid_data)} aliran valid")
                else:
                    st.success(f"📊 Menampilkan {len(valid_data)} aliran data")
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Buat flow map dengan animasi
                try:
                    import plotly.graph_objects as go
                    import plotly.express as px
                    import numpy as np
                    
                    status_text.text("🌍 Membuat flow map 3D...")
                    
                    # Buat figure dengan layout globe
                    fig = go.Figure()
                    
                    # Generate colors berdasarkan value atau sequential
                    if value_col and value_col in valid_data.columns:
                        colors = px.colors.sample_colorscale(color_scheme.lower(), 
                                                           np.linspace(0, 1, len(valid_data)))
                        color_scale = px.colors.sequential.__dict__.get(color_scheme, px.colors.sequential.Viridis)
                    else:
                        colors = px.colors.sample_colorscale('viridis', np.linspace(0, 1, len(valid_data)))
                        color_scale = px.colors.sequential.Viridis
                    
                    # Konfigurasi garis berdasarkan style
                    dash_styles = {
                        "Solid": None,
                        "Dashed": "dash",
                        "Dotted": "dot",
                        "Animated": "dash"
                    }
                    
                    dash_style = dash_styles.get(flow_style, None)
                    
                    # Tambahkan lines untuk setiap flow dengan progress
                    for idx, row in valid_data.iterrows():
                        if idx % 20 == 0:  # Update progress setiap 20 rows
                            progress_bar.progress(min((idx + 1) / len(valid_data), 1.0))
                        
                        # Hitung ketebalan garis
                        current_line_width = line_width
                        if value_col and pd.notna(row[value_col]):
                            max_val = valid_data[value_col].max()
                            min_val = valid_data[value_col].min()
                            if max_val > min_val:
                                current_line_width = max(1, line_width + (row[value_col] - min_val) / (max_val - min_val) * 8)
                        
                        # Warna berdasarkan value atau sequential
                        if value_col and value_col in valid_data.columns:
                            color_idx = int((row[value_col] - min_val) / (max_val - min_val) * (len(colors) - 1)) if max_val > min_val else 0
                            line_color = colors[color_idx]
                        else:
                            line_color = colors[idx % len(colors)]
                        
                        # Tambahkan garis aliran
                        fig.add_trace(go.Scattergeo(
                            lon = [row[origin_lon], row[dest_lon]],
                            lat = [row[origin_lat], row[dest_lat]],
                            mode = 'lines',
                            line = dict(
                                width = current_line_width,
                                color = line_color,
                                dash = dash_style
                            ),
                            opacity = 0.7,
                            name = f"Flow {idx+1}",
                            showlegend=False,
                            hoverinfo='text',
                            hovertext = f"Origin: ({row[origin_lat]:.2f}, {row[origin_lon]:.2f})<br>"
                                      f"Dest: ({row[dest_lat]:.2f}, {row[dest_lon]:.2f})<br>"
                                      f"{f'Value: {row[value_col]}' if value_col else ''}"
                        ))
                        
                        # Tambahkan animasi kapal untuk style animated
                        if flow_style == "Animated":
                            # Buat titik animasi di sepanjang garis
                            num_points = 5
                            for i in range(num_points):
                                frac = i / (num_points - 1) if num_points > 1 else 0.5
                                anim_lat = row[origin_lat] + (row[dest_lat] - row[origin_lat]) * frac
                                anim_lon = row[origin_lon] + (row[dest_lon] - row[origin_lon]) * frac
                                
                                fig.add_trace(go.Scattergeo(
                                    lon = [anim_lon],
                                    lat = [anim_lat],
                                    mode = 'markers',
                                    marker = dict(
                                        size = 8,
                                        color = 'yellow',
                                        symbol = 'triangle-up',
                                        line = dict(width=1, color='darkorange')
                                    ),
                                    opacity = 0.6 - (i * 0.1),
                                    name = f"Ship {idx+1}",
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                    
                    # Tambahkan markers untuk origin dan destination
                    status_text.text("📍 Menambahkan markers...")
                    
                    unique_origins = valid_data[[origin_lat, origin_lon]].drop_duplicates().head(50)  # Batasi jumlah marker
                    unique_dests = valid_data[[dest_lat, dest_lon]].drop_duplicates().head(50)
                    
                    fig.add_trace(go.Scattergeo(
                        lon = unique_origins[origin_lon],
                        lat = unique_origins[origin_lat],
                        mode = 'markers',
                        marker = dict(
                            size=8, 
                            color='blue',
                            symbol='circle',
                            line=dict(width=2, color='darkblue')
                        ),
                        name = '📍 Origin',
                        text = ['Origin'] * len(unique_origins),
                        hoverinfo='text+lon+lat'
                    ))
                    
                    fig.add_trace(go.Scattergeo(
                        lon = unique_dests[dest_lon],
                        lat = unique_dests[dest_lat],
                        mode = 'markers',
                        marker = dict(
                            size=8, 
                            color='red',
                            symbol='square',
                            line=dict(width=2, color='darkred')
                        ),
                        name = '🎯 Destination',
                        text = ['Destination'] * len(unique_dests),
                        hoverinfo='text+lon+lat'
                    ))
                    
                    # Update layout dengan tampilan globe
                    fig.update_layout(
                        title_text = f'🌍 Flow Map 3D - {len(valid_data)} Aliran',
                        showlegend = True,
                        geo = dict(
                            scope = 'world',
                            projection_type = map_style,
                            showland = True,
                            landcolor = 'rgb(100, 125, 100)',
                            countrycolor = 'rgb(200, 200, 200)',
                            coastlinecolor = 'rgb(160, 160, 160)',
                            lakecolor = 'rgb(100, 150, 250)',
                            oceancolor = 'rgb(50, 100, 200)',
                            showocean = True,
                            showcountries = True,
                            showcoastlines = True,
                            showframe = False,
                            bgcolor = 'rgb(0, 0, 0)',
                        ),
                        paper_bgcolor = 'black',
                        font = dict(color='white'),
                        height = 700,
                        hovermode = 'closest'
                    )
                    
                    # Tambahkan animasi frame untuk efek kapal bergerak
                    if flow_style == "Animated":
                        frames = []
                        for frame_num in range(5):
                            frame_data = []
                            for idx, row in valid_data.iterrows():
                                frac = frame_num / 4
                                anim_lat = row[origin_lat] + (row[dest_lat] - row[origin_lat]) * frac
                                anim_lon = row[origin_lon] + (row[dest_lon] - row[origin_lon]) * frac
                                
                                frame_data.append(
                                    go.Scattergeo(
                                        lon=[anim_lon],
                                        lat=[anim_lat],
                                        mode='markers',
                                        marker=dict(size=10, color='yellow', symbol='triangle-up')
                                    )
                                )
                            
                            frames.append(go.Frame(data=frame_data, name=f"frame{frame_num}"))
                        
                        fig.frames = frames
                        
                        # Tambahkan play button untuk animasi
                        fig.update_layout(
                            updatemenus=[dict(
                                type="buttons",
                                buttons=[dict(label="▶️ Play",
                                            method="animate",
                                            args=[None, {"frame": {"duration": 500, "redraw": True},
                                                        "fromcurrent": True}])]
                            )]
                        )
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Flow map 3D selesai dibuat!")
                    
                    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
                    
                    # Legenda warna
                    if value_col and value_col in valid_data.columns:
                        st.write("**🎨 Legenda Intensitas Aliran**")
                        min_val = valid_data[value_col].min()
                        max_val = valid_data[value_col].max()
                        st.caption(f"Warna menunjukkan nilai dari {min_val:.2f} (biru) hingga {max_val:.2f} (merah)")
                    
                    # Tampilkan data table dengan pagination
                    with st.expander("📋 Lihat Data Flow"):
                        display_cols = [origin_lat, origin_lon, dest_lat, dest_lon]
                        if value_col:
                            display_cols.append(value_col)
                        
                        # Pagination
                        page_size = 20
                        total_pages = max(1, len(valid_data) // page_size)
                        page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, key="flow_page")
                        
                        start_idx = (page - 1) * page_size
                        end_idx = min(start_idx + page_size, len(valid_data))
                        
                        st.dataframe(valid_data[display_cols].iloc[start_idx:end_idx])
                        st.caption(f"Menampilkan data {start_idx + 1}-{end_idx} dari {len(valid_data)}")
                        
                except Exception as e:
                    st.error(f"❌ Error membuat flow map: {str(e)}")
                    st.info("💡 Tips: Pastikan data koordinat dalam format numerik yang valid")
                    
            else:
                st.error("❌ Tidak ada data dengan koordinat origin-destination yang valid")
                
        else:
            st.warning("⚠️ Pilih semua kolom koordinat untuk menampilkan flow map")
            
    else:
        st.warning("""
        ⚠️ Tidak terdeteksi kolom untuk flow map.
        
        **Untuk menampilkan Flow Map, data harus mengandung:**
        - Origin coordinates (latitude & longitude)
        - Destination coordinates (latitude & longitude) 
        - Opsional: Value column untuk ketebalan flow
        
        **Format kolom yang disarankan:**
        - `origin_lat`, `origin_lon`, `dest_lat`, `dest_lon`
        - `from_latitude`, `from_longitude`, `to_latitude`, `to_longitude`
        - `source_lat`, `source_lon`, `target_lat`, `target_lon`
        """)
def create_heatmap(df, numeric_cols):
    
    # Deteksi ukuran data
    data_size = len(df)
    if data_size > 100000:
        st.info(f"⚡ Mode Optimasi: Data besar ({data_size:,} rows) - Menggunakan sampling otomatis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_cols = st.slider(
            "Maksimum kolom ditampilkan",
            min_value=5,
            max_value=20,
            value=10 if data_size > 50000 else 15,
            key="heatmap_max_cols"
        )
    
    with col2:
        optimization_mode = st.selectbox(
            "Mode Optimasi",
            ["Auto", "Fast", "Balanced", "Detailed"],
            index=0 if data_size > 50000 else 2,
            key="heatmap_optim"
        )
    
    # Filter numeric columns yang feasible untuk heatmap
    suitable_cols = [col for col in numeric_cols 
                   if df[col].nunique() > 1 and df[col].dtype in ['float64', 'int64']]
    
    selected_cols = st.multiselect(
        "Pilih kolom untuk heatmap", 
        suitable_cols[:max_cols],  # Batasi pilihan
        default=suitable_cols[:min(8, len(suitable_cols))], 
        key="heatmap_cols"
    )
    
    # Pengaturan lanjutan
    with st.expander("⚙️ Pengaturan Lanjutan", expanded=False):
        col3, col4, col5 = st.columns(3)
        with col3:
            color_scale = st.selectbox(
                "Skala warna",
                ["RdBu_r", "Viridis", "Plasma", "Inferno", "Blues", "Greens"],
                key="heatmap_color"
            )
        with col4:
            show_values = st.selectbox(
                "Tampilkan nilai",
                ["Auto", "Always", "Never", "Significant Only"],
                key="heatmap_values"
            )
        with col5:
            correlation_method = st.selectbox(
                "Metode korelasi",
                ["pearson", "spearman", "kendall"],
                key="heatmap_method"
            )
    
    if len(selected_cols) >= 2:
        try:
            with st.spinner("🔄 Menghitung matriks korelasi..."):
                # OPTIMASI 1: Sampling data untuk kalkulasi korelasi
                processed_df = optimize_heatmap_data(df, selected_cols, data_size, optimization_mode)
                
                if len(processed_df) == 0:
                    st.warning("Tidak ada data valid setelah preprocessing")
                    return
                
                # OPTIMASI 2: Hitung matriks korelasi yang efisien
                corr_matrix = calculate_correlation_matrix(processed_df, selected_cols, correlation_method)
                
                # OPTIMASI 3: Buat heatmap yang dioptimalkan
                fig = create_optimized_heatmap(corr_matrix, selected_cols, color_scale, show_values, data_size)
                
                # OPTIMASI 4: Konfigurasi plotly yang ringan
                config = {
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'responsive': True
                }
                
                st.plotly_chart(fig, use_container_width=True, config=config)
                
                # Tampilkan analisis tambahan
                display_correlation_analysis(corr_matrix, processed_df, selected_cols)
                
                # Tampilkan info optimasi
                show_heatmap_optimization_info(data_size, len(processed_df), optimization_mode)
                
        except Exception as e:
            st.error(f"Error membuat heatmap: {str(e)}")
            # Fallback ke metode sederhana
            create_simple_heatmap_fallback(df, selected_cols)
    else:
        st.warning("Pilih minimal 2 kolom untuk heatmap")

def optimize_heatmap_data(df, selected_cols, data_size, optimization_mode):
    """Optimasi data untuk heatmap dengan sampling yang tepat"""
    
    # Filter data yang valid
    clean_df = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean_df) == 0:
        return clean_df
    
    # Tentukan target sample size
    target_sizes = {
        "Auto": min(10000, data_size),
        "Fast": min(5000, data_size),
        "Balanced": min(20000, data_size),
        "Detailed": min(50000, data_size)
    }
    
    target_size = target_sizes[optimization_mode]
    
    # Jika data lebih besar dari target, lakukan sampling
    if len(clean_df) > target_size:
        if optimization_mode == "Fast":
            # Systematic sampling untuk performa maksimal
            step = len(clean_df) // target_size
            sampled_df = clean_df.iloc[::step]
        elif optimization_mode == "Balanced":
            # Stratified sampling untuk mempertahankan korelasi
            try:
                # Sample berdasarkan kombinasi nilai ekstrem
                n_samples_per_quantile = target_size // 4
                sampled_dfs = []
                
                for col in selected_cols[:3]:  # Gunakan 3 kolom pertama untuk stratification
                    for quantile in [0.25, 0.5, 0.75]:
                        threshold = clean_df[col].quantile(quantile)
                        quantile_data = clean_df[clean_df[col] <= threshold].tail(n_samples_per_quantile // 3)
                        sampled_dfs.append(quantile_data)
                
                # Gabungkan dan hapus duplikat
                sampled_df = pd.concat(sampled_dfs, ignore_index=True).drop_duplicates()
                
                # Jika masih kurang, tambahkan random sampling
                if len(sampled_df) < target_size:
                    remaining = target_size - len(sampled_df)
                    additional_samples = clean_df.sample(n=remaining, random_state=42)
                    sampled_df = pd.concat([sampled_df, additional_samples], ignore_index=True)
                    
            except:
                # Fallback ke random sampling
                sampled_df = clean_df.sample(n=target_size, random_state=42)
        else:
            # Random sampling untuk mode lain
            sampled_df = clean_df.sample(n=target_size, random_state=42)
        
        return sampled_df
    
    return clean_df

def calculate_correlation_matrix(df, selected_cols, correlation_method):
    """Hitung matriks korelasi yang efisien"""
    
    # OPTIMASI: Gunakan numpy untuk kalkulasi yang lebih cepat
    data_subset = df[selected_cols]
    
    if correlation_method == "pearson":
        corr_matrix = data_subset.corr(method='pearson')
    elif correlation_method == "spearman":
        # Spearman lebih robust untuk data non-linear
        corr_matrix = data_subset.corr(method='spearman')
    else:  # kendall
        corr_matrix = data_subset.corr(method='kendall')
    
    return corr_matrix

def create_optimized_heatmap(corr_matrix, selected_cols, color_scale, show_values, data_size):
    """Buat heatmap yang dioptimalkan untuk performa"""
    
    # OPTIMASI: Tentukan apakah menampilkan nilai teks
    if show_values == "Auto":
        text_auto = True if len(selected_cols) <= 15 else False
    elif show_values == "Always":
        text_auto = True
    elif show_values == "Never":
        text_auto = False
    else:  # Significant Only
        # Hanya tampilkan nilai yang signifikan (|correlation| > 0.3)
        text_matrix = np.where(np.abs(corr_matrix.values) > 0.3, 
                              corr_matrix.values.round(2), 
                              "")
        text_auto = text_matrix
    
    # Buat heatmap
    fig = px.imshow(
        corr_matrix, 
        text_auto=text_auto,
        aspect="auto", 
        title=f"Heatmap Korelasi ({len(selected_cols)} variabel)",
        color_continuous_scale=color_scale,
        zmin=-1,  # Fixed range untuk korelasi
        zmax=1
    )
    
    # OPTIMASI: Update layout untuk performa
    fig.update_traces(
        hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Korelasi: %{z:.3f}<extra></extra>'
    )
    
    # Layout yang dioptimalkan
    height = max(400, len(selected_cols) * 30)  # Dynamic height berdasarkan jumlah kolom
    fig.update_layout(
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis=dict(tickangle=-45),
        plot_bgcolor='white'
    )
    
    # Tambahkan colorbar yang informatif
    fig.update_coloraxes(
        colorbar=dict(
            title="Korelasi",
            titleside="right",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1.0 (Strong -)", "-0.5", "0.0 (No)", "0.5", "1.0 (Strong +)"]
        )
    )
    
    return fig

def display_correlation_analysis(corr_matrix, processed_df, selected_cols):
    """Tampilkan analisis korelasi tambahan"""
    
    with st.expander("📊 Analisis Korelasi Detail", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        # Hitung statistik korelasi
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        
        with col1:
            avg_correlation = np.mean(np.abs(corr_values))
            st.metric("Rata-rata Korelasi (abs)", f"{avg_correlation:.3f}")
        
        with col2:
            strong_correlations = np.sum(np.abs(corr_values) > 0.7)
            st.metric("Korelasi Kuat (|r| > 0.7)", f"{strong_correlations}")
        
        with col3:
            weak_correlations = np.sum(np.abs(corr_values) < 0.3)
            st.metric("Korelasi Lemah (|r| < 0.3)", f"{weak_correlations}")
        
        # Top correlations
        st.subheader("🔝 Korelasi Terkuat")
        
        # Dapatkan pasangan dengan korelasi tertinggi
        corr_pairs = []
        for i in range(len(selected_cols)):
            for j in range(i+1, len(selected_cols)):
                corr_val = corr_matrix.iloc[i, j]
                corr_pairs.append({
                    'Variable 1': selected_cols[i],
                    'Variable 2': selected_cols[j],
                    'Correlation': corr_val,
                    'Strength': 'Strong' if abs(corr_val) > 0.7 else 
                               'Moderate' if abs(corr_val) > 0.3 else 'Weak'
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        top_correlations = corr_df.nlargest(10, 'Correlation')
        bottom_correlations = corr_df.nsmallest(10, 'Correlation')
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.markdown("**🔼 Korelasi Positif Tertinggi**")
            st.dataframe(
                top_correlations.style.format({'Correlation': '{:.3f}'}),
                use_container_width=True
            )
        
        with col5:
            st.markdown("**🔽 Korelasi Negatif Tertinggi**")
            st.dataframe(
                bottom_correlations.style.format({'Correlation': '{:.3f}'}),
                use_container_width=True
            )
        
        # Correlation clusters
        st.subheader("🎯 Kluster Korelasi")
        try:
            from scipy.cluster import hierarchy
            
            # Hierarchical clustering untuk mengidentifikasi pola
            corr_array = 1 - np.abs(corr_matrix.values)  # Convert to distance matrix
            linkage_matrix = hierarchy.linkage(corr_array, method='average')
            
            # Dapatkan order dari dendrogram
            dendro_order = hierarchy.dendrogram(linkage_matrix, no_plot=True)['leaves']
            clustered_cols = [selected_cols[i] for i in dendro_order]
            
            st.markdown(f"**Urutan Kluster:** {', '.join(clustered_cols[:5])}...")
            
        except Exception as e:
            st.info("Klustering tidak tersedia untuk dataset ini")

def show_heatmap_optimization_info(original_size, processed_size, optimization_mode):
    """Tampilkan informasi optimasi"""
    
    reduction_pct = ((original_size - processed_size) / original_size) * 100 if original_size > 0 else 0
    
    if reduction_pct > 10:
        with st.expander("⚡ Info Optimasi Performa", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Original", f"{original_size:,}")
            with col2:
                st.metric("Data Diproses", f"{processed_size:,}")
            with col3:
                st.metric("Reduksi", f"{reduction_pct:.1f}%")
            
            optimization_strategies = {
                "Fast": "• ✅ **Aggressive sampling**\n• ✅ **Basic correlation**\n• ✅ **Minimal text**",
                "Balanced": "• ✅ **Stratified sampling**\n• ✅ **Multiple methods**\n• ✅ **Smart text display**",
                "Detailed": "• ✅ **Maximum data retention**\n• ✅ **Advanced analysis**\n• ✅ **Full features**"
            }
            
            st.info(f"**Mode {optimization_mode}**: {optimization_strategies.get(optimization_mode, 'Custom optimization')}")

def create_simple_heatmap_fallback(df, selected_cols):
    """Fallback method untuk data yang bermasalah"""
    st.warning("Menggunakan metode fallback sederhana...")
    
    # Sample kecil untuk kalkulasi cepat
    sample_df = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna().head(2000)
    
    if len(sample_df) == 0:
        st.error("Tidak ada data valid")
        return
    
    corr_matrix = sample_df.corr()
    
    fig = px.imshow(
        corr_matrix, 
        text_auto=True,
        aspect="auto", 
        title="Simple Heatmap Korelasi",
        color_continuous_scale='RdBu_r'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Versi ultra-ringan untuk data ekstrem
def create_ultra_fast_heatmap(df, numeric_cols):
    """Versi ultra-ringan untuk data > 500k rows"""
    st.subheader("🚀 Heatmap Ultra-Fast")
    
    # Pilih kolom otomatis (max 8)
    suitable_cols = [col for col in numeric_cols 
                   if df[col].nunique() > 1 and df[col].dtype in ['float64', 'int64']]
    selected_cols = st.multiselect(
        "Pilih kolom", 
        suitable_cols[:8],
        default=suitable_cols[:min(6, len(suitable_cols))],
        key="ultra_heatmap_cols"
    )
    
    if len(selected_cols) >= 2:
        # Sampling sangat agresif
        sample_df = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sample_df) > 5000:
            sample_df = sample_df.sample(n=5000, random_state=42)
        
        corr_matrix = sample_df.corr()
        
        fig = px.imshow(
            corr_matrix, 
            text_auto=True,
            aspect="auto", 
            title=f"Ultra-Fast Heatmap ({len(selected_cols)} variables)"
        )
        
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.info(f"📊 Ultra-Fast Mode: 5,000 samples, {len(selected_cols)} variables")

def create_multiple_line_chart(df, numeric_cols, non_numeric_cols):
    
    # Deteksi ukuran data
    data_size = len(df)
    if data_size > 100000:
        st.info(f"⚡ Mode Optimasi: Data besar ({data_size:,} rows) - Menggunakan sampling otomatis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_lines = st.multiselect(
            "Pilih kolom untuk garis", 
            numeric_cols[:15],  # Batasi pilihan
            default=numeric_cols[:min(5, len(numeric_cols))],
            key="multi_line_select"
        )
    
    with col2:
        x_col = st.selectbox(
            "Pilih kolom untuk sumbu X", 
            [df.index.name if df.index.name else "index"] + non_numeric_cols + numeric_cols, 
            key="multi_x_col"
        )
    
    # Pengaturan optimasi
    with st.expander("⚙️ Pengaturan Optimasi", expanded=False):
        col3, col4, col5 = st.columns(3)
        
        with col3:
            optimization_mode = st.selectbox(
                "Mode Optimasi",
                ["Auto", "Fast", "Balanced", "Detailed"],
                index=0 if data_size > 50000 else 2,
                key="multi_line_optim"
            )
        
        with col4:
            max_points = st.slider(
                "Maksimum titik data per garis",
                min_value=500,
                max_value=10000,
                value=2000 if data_size > 100000 else 5000,
                key="multi_line_max_points"
            )
        
        with col5:
            line_style = st.selectbox(
                "Style garis",
                ["Solid", "Dashed", "Dotted", "Dash-Dot"],
                key="multi_line_style"
            )
    
    # Pengaturan lanjutan
    with st.expander("🔧 Pengaturan Lanjutan", expanded=False):
        col6, col7, col8 = st.columns(3)
        
        with col6:
            aggregation_method = st.selectbox(
                "Metode aggregasi",
                ["none", "mean", "median", "max", "min"],
                key="multi_line_agg"
            )
        
        with col7:
            show_confidence = st.checkbox(
                "Tampilkan confidence band", 
                value=False,
                key="multi_line_confidence"
            )
        
        with col8:
            sync_axes = st.checkbox(
                "Sinkronisasi sumbu Y", 
                value=True,
                key="multi_line_sync"
            )
    
    if selected_lines and x_col:
        try:
            with st.spinner("🔄 Memproses multiple line chart..."):
                # OPTIMASI 1: Persiapan data dasar
                plot_data = prepare_multiline_data(df, selected_lines, x_col, data_size, optimization_mode)
                
                if plot_data is None or len(plot_data) == 0:
                    st.warning("Tidak ada data valid untuk ditampilkan")
                    return
                
                # OPTIMASI 2: Sampling dan aggregasi data
                processed_data = optimize_multiline_data(plot_data, selected_lines, x_col, max_points, aggregation_method, optimization_mode)
                
                # OPTIMASI 3: Buat multiple line chart yang dioptimalkan
                fig = create_optimized_multiline_chart(processed_data, selected_lines, x_col, line_style, show_confidence, sync_axes, data_size)
                
                # OPTIMASI 4: Konfigurasi plotly yang ringan
                config = {
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverClosestGl2d'],
                    'scrollZoom': True,
                    'responsive': True
                }
                
                st.plotly_chart(fig, use_container_width=True, config=config)
                
                # Tampilkan statistik
                display_multiline_statistics(processed_data, selected_lines, x_col)
                
                # Tampilkan info optimasi
                show_multiline_optimization_info(data_size, len(processed_data), optimization_mode)
                
        except Exception as e:
            st.error(f"Error membuat multiple line chart: {str(e)}")
            # Fallback ke metode sederhana
            create_simple_multiline_fallback(df, selected_lines, x_col)

def prepare_multiline_data(df, selected_lines, x_col, data_size, optimization_mode):
    """Persiapkan data untuk multiple line chart"""
    
    # Pilih kolom yang diperlukan
    columns_needed = [x_col] + selected_lines
    plot_data = df[columns_needed].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(plot_data) == 0:
        return None
    
    # OPTIMASI: Sampling untuk data besar
    if data_size > 100000:
        target_sizes = {
            "Auto": min(20000, data_size),
            "Fast": min(5000, data_size),
            "Balanced": min(30000, data_size),
            "Detailed": min(50000, data_size)
        }
        
        target_size = target_sizes[optimization_mode]
        
        if len(plot_data) > target_size:
            if pd.api.types.is_datetime64_any_dtype(plot_data[x_col]):
                # Untuk time series: systematic sampling dengan sorting
                plot_data = plot_data.sort_values(x_col)
                step = len(plot_data) // target_size
                plot_data = plot_data.iloc[::step]
            else:
                # Untuk data non-time series: random sampling
                plot_data = plot_data.sample(n=target_size, random_state=42)
    
    return plot_data

def optimize_multiline_data(plot_data, selected_lines, x_col, max_points, aggregation_method, optimization_mode):
    """Optimasi data untuk multiple line chart"""
    
    processed_data = plot_data.copy()
    
    # Jika masih terlalu banyak points, lakukan aggregasi
    if len(processed_data) > max_points and aggregation_method != "none":
        if pd.api.types.is_datetime64_any_dtype(processed_data[x_col]):
            # Aggregasi time series
            processed_data = processed_data.set_index(x_col)
            
            # Tentukan frekuensi resampling berdasarkan jumlah data
            if len(processed_data) > 50000:
                freq = '1H'
            elif len(processed_data) > 20000:
                freq = '30T'
            else:
                freq = '10T'
            
            try:
                if aggregation_method == "mean":
                    processed_data = processed_data.resample(freq).mean()
                elif aggregation_method == "median":
                    processed_data = processed_data.resample(freq).median()
                elif aggregation_method == "max":
                    processed_data = processed_data.resample(freq).max()
                elif aggregation_method == "min":
                    processed_data = processed_data.resample(freq).min()
                
                processed_data = processed_data.reset_index()
                
            except:
                # Fallback: simple sampling
                step = len(processed_data) // max_points
                processed_data = processed_data.iloc[::step]
        
        else:
            # Aggregasi non-time series: binning
            n_bins = min(max_points, 1000)
            processed_data['x_bins'] = pd.cut(processed_data[x_col], bins=n_bins)
            
            if aggregation_method == "mean":
                aggregated = processed_data.groupby('x_bins').mean().reset_index()
            elif aggregation_method == "median":
                aggregated = processed_data.groupby('x_bins').median().reset_index()
            elif aggregation_method == "max":
                aggregated = processed_data.groupby('x_bins').max().reset_index()
            elif aggregation_method == "min":
                aggregated = processed_data.groupby('x_bins').min().reset_index()
            
            aggregated[x_col] = aggregated['x_bins'].apply(lambda x: x.mid)
            processed_data = aggregated[[x_col] + selected_lines]
    
    # Batasi akhir jika masih terlalu banyak
    if len(processed_data) > max_points:
        processed_data = processed_data.head(max_points)
    
    return processed_data

def create_optimized_multiline_chart(processed_data, selected_lines, x_col, line_style, show_confidence, sync_axes, original_size):
    """Buat multiple line chart yang dioptimalkan"""
    
    fig = go.Figure()
    
    # Mapping line style
    line_styles = {
        "Solid": None,
        "Dashed": "dash",
        "Dotted": "dot",
        "Dash-Dot": "dashdot"
    }
    
    dash_pattern = line_styles[line_style]
    
    # Warna untuk multiple lines
    colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    
    # OPTIMASI: Gunakan scattergl untuk data banyak
    use_webgl = len(processed_data) > 2000
    
    for i, col in enumerate(selected_lines):
        clean_data = processed_data[[x_col, col]].dropna()
        
        if len(clean_data) > 0:
            if use_webgl:
                # WebGL untuk performa tinggi
                trace = go.Scattergl(
                    x=clean_data[x_col],
                    y=clean_data[col],
                    mode='lines',
                    name=col,
                    line=dict(
                        color=colors[i % len(colors)],
                        width=2,
                        dash=dash_pattern
                    ),
                    hovertemplate=f'<b>{col}</b><br>{x_col}: %{{x}}<br>Nilai: %{{y:.2f}}<extra></extra>'
                )
            else:
                # Regular scatter untuk data sedikit
                trace = go.Scatter(
                    x=clean_data[x_col],
                    y=clean_data[col],
                    mode='lines',
                    name=col,
                    line=dict(
                        color=colors[i % len(colors)],
                        width=2,
                        dash=dash_pattern
                    ),
                    hovertemplate=f'<b>{col}</b><br>{x_col}: %{{x}}<br>Nilai: %{{y:.2f}}<extra></extra>'
                )
            
            fig.add_trace(trace)
            
            # Tambahkan confidence band jika diminta
            if show_confidence and len(clean_data) > 10:
                try:
                    y_mean = clean_data[col].mean()
                    y_std = clean_data[col].std()
                    
                    fig.add_trace(go.Scatter(
                        x=clean_data[x_col],
                        y=y_mean + y_std,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=clean_data[x_col],
                        y=y_mean - y_std,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f'rgba{(*px.colors.hex_to_rgb(colors[i % len(colors)]), 0.2)}',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                except:
                    pass  # Skip confidence band jika error
    
    # Layout yang dioptimalkan
    layout_config = {
        'title': f"Multiple Line Chart: {len(selected_lines)} Variables",
        'xaxis_title': x_col,
        'yaxis_title': "Nilai",
        'hovermode': 'x unified',
        'height': 500,
        'showlegend': len(selected_lines) <= 10,  # Sembunyikan legend jika terlalu banyak lines
        'margin': dict(l=50, r=50, t=80, b=50),
        'plot_bgcolor': 'white'
    }
    
    # Tambahkan range slider untuk time series
    if pd.api.types.is_datetime64_any_dtype(processed_data[x_col]) and len(processed_data) > 1000:
        layout_config['xaxis'] = dict(rangeslider=dict(visible=True, thickness=0.05))
    
    # Sinkronisasi sumbu Y jika diminta
    if not sync_axes and len(selected_lines) > 1:
        layout_config['yaxis'] = dict(title=selected_lines[0])
        
        # Tambahkan secondary axes untuk lines lainnya
        for i, col in enumerate(selected_lines[1:], 2):
            fig.update_layout(**{f'yaxis{i}': dict(
                title=col,
                side='right' if i % 2 == 0 else 'left',
                overlaying='y',
                position=0.95 if i % 2 == 0 else 0.05
            )})
            
            # Assign trace ke axis yang sesuai
            fig.data[i].update(yaxis=f'y{i}')
    
    fig.update_layout(**layout_config)
    
    return fig

def display_multiline_statistics(processed_data, selected_lines, x_col):
    """Tampilkan statistik multiple line chart"""
    
    with st.expander("📊 Statistik Multiple Lines", expanded=False):
        # Statistik dasar
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Jumlah Garis", len(selected_lines))
        with col2:
            st.metric("Total Data Points", len(processed_data))
        with col3:
            st.metric("Rentang " + x_col, 
                     f"{len(processed_data[x_col].unique())} unique values")
        
        # Tabel statistik per line
        stats_data = []
        for col in selected_lines:
            clean_data = processed_data[col].dropna()
            if len(clean_data) > 0:
                stats_data.append({
                    'Variable': col,
                    'Mean': clean_data.mean(),
                    'Std Dev': clean_data.std(),
                    'Min': clean_data.min(),
                    'Max': clean_data.max(),
                    'Data Points': len(clean_data)
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(
                stats_df.style.format({
                    'Mean': '{:.2f}',
                    'Std Dev': '{:.2f}',
                    'Min': '{:.2f}',
                    'Max': '{:.2f}'
                }),
                use_container_width=True
            )

def show_multiline_optimization_info(original_size, processed_size, optimization_mode):
    """Tampilkan informasi optimasi"""
    
    reduction_pct = ((original_size - processed_size) / original_size) * 100 if original_size > 0 else 0
    
    if reduction_pct > 10:
        with st.expander("⚡ Info Optimasi Performa", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Original", f"{original_size:,}")
            with col2:
                st.metric("Data Diproses", f"{processed_size:,}")
            with col3:
                st.metric("Reduksi", f"{reduction_pct:.1f}%")
            
            optimization_strategies = {
                "Fast": "• ✅ **Aggressive sampling**\n• ✅ **WebGL rendering**\n• ✅ **Minimal features**",
                "Balanced": "• ✅ **Smart sampling**\n• ✅ **Optimized aggregation**\n• ✅ **Enhanced visuals**",
                "Detailed": "• ✅ **Maximum data retention**\n• ✅ **Advanced features**\n• ✅ **Full analysis**"
            }
            
            st.info(f"**Mode {optimization_mode}**: {optimization_strategies.get(optimization_mode, 'Custom optimization')}")

def create_simple_multiline_fallback(df, selected_lines, x_col):
    """Fallback method untuk data yang bermasalah"""
    st.warning("Menggunakan metode fallback sederhana...")
    
    # Sample kecil untuk performa
    sample_data = df[[x_col] + selected_lines].dropna().head(1000)
    
    if len(sample_data) == 0:
        st.error("Tidak ada data valid")
        return
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, col in enumerate(selected_lines):
        clean_data = sample_data[[x_col, col]].dropna()
        if len(clean_data) > 0:
            fig.add_trace(go.Scatter(
                x=clean_data[x_col],
                y=clean_data[col],
                mode='lines',
                name=col,
                line=dict(color=colors[i % len(colors)], width=1.5)
            ))
    
    fig.update_layout(
        title=f"Simple Multiple Lines: {len(selected_lines)} Variables",
        xaxis_title=x_col,
        yaxis_title="Nilai",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Versi ultra-ringan untuk data ekstrem
def create_ultra_fast_multiline(df, numeric_cols, non_numeric_cols):
    """Versi ultra-ringan untuk data > 500k rows"""
    st.subheader("🚀 Multiple Line Chart Ultra-Fast")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_lines = st.multiselect(
            "Pilih garis", 
            numeric_cols[:8],
            default=numeric_cols[:min(4, len(numeric_cols))],
            key="ultra_multi_lines"
        )
    with col2:
        x_col = st.selectbox(
            "Sumbu X", 
            [df.index.name if df.index.name else "index"] + non_numeric_cols[:3] + numeric_cols[:3],
            key="ultra_multi_x"
        )
    
    if selected_lines and x_col:
        # Sampling sangat agresif
        sample_data = df[[x_col] + selected_lines].dropna()
        if len(sample_data) > 2000:
            sample_data = sample_data.sample(n=2000, random_state=42)
        
        # WebGL rendering
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, col in enumerate(selected_lines):
            clean_data = sample_data[[x_col, col]].dropna()
            if len(clean_data) > 0:
                fig.add_trace(go.Scattergl(
                    x=clean_data[x_col],
                    y=clean_data[col],
                    mode='lines',
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=1)
                ))
        
        fig.update_layout(
            title=f"Ultra-Fast: {len(selected_lines)} Lines",
            height=350,
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.info(f"📊 Ultra-Fast Mode: 2,000 samples, {len(selected_lines)} lines")


# Fungsi statistik yang dioptimalkan
@st.cache_data(show_spinner=False)
def show_optimized_statistics(df):
    st.header("📊 Statistik Deskriptif")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Jumlah Baris", df.shape[0])
    with col2:
        st.metric("Jumlah Kolom", df.shape[1])
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.metric("Kolom Numerik", len(numeric_cols))
    with col4:
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        st.metric("Kolom Non-Numerik", len(non_numeric_cols))
    
    st.subheader("👀 Preview Data")
    preview_df = df.head(100) if len(df) > 100 else df
    st.dataframe(preview_df, use_container_width=True)

    # STATISTIK NUMERIK
    if numeric_cols:
        st.subheader("📈 Statistik Numerik Lengkap")
        clean_df = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        desc_stats = clean_df.describe()
        
        # Tambahkan metrik tambahan
        additional_stats = pd.DataFrame({
            'median': clean_df.median(),
            'variance': clean_df.var(),
            'skewness': clean_df.skew(),
            'kurtosis': clean_df.kurtosis(),
            'range': clean_df.max() - clean_df.min(),
            'coefficient_of_variation': (clean_df.std() / clean_df.mean()) * 100
        }).T
        
        st.write("**Statistik Deskriptif Dasar:**")
        st.dataframe(desc_stats, use_container_width=True)
        
        st.write("**Statistik Tambahan:**")
        st.dataframe(additional_stats, use_container_width=True)
        
        # Visualisasi distribusi numerik
        st.write("**📊 Distribusi Data Numerik**")
        for col in numeric_cols[:4]:  # Batasi agar tidak terlalu banyak
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    df[col].hist(bins=30, ax=ax, edgecolor='black')
                    ax.set_title(f'Distribusi {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frekuensi')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error plot distribusi {col}: {e}")
            
            with col2:
                try:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.boxplot(df[col].dropna())
                    ax.set_title(f'Box Plot {col}')
                    ax.set_ylabel(col)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error box plot {col}: {e}")

    # ANALISIS MISSING VALUES
    st.subheader("❓ Informasi Missing Values")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Kolom': missing_data.index,
        'Jumlah Missing': missing_data.values,
        'Persentase Missing': missing_percent.values
    })

    st.dataframe(missing_df, use_container_width=True)

    # Visualisasi missing values
    if missing_data.sum() > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                missing_df_sorted = missing_df[missing_df['Jumlah Missing'] > 0].sort_values('Persentase Missing', ascending=False)
                if not missing_df_sorted.empty:
                    bars = ax.bar(missing_df_sorted['Kolom'], missing_df_sorted['Persentase Missing'])
                    ax.set_title('Persentase Missing Values per Kolom')
                    ax.set_ylabel('Persentase Missing (%)')
                    ax.tick_params(axis='x', rotation=45)
                    # Tambahkan nilai di atas bar
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom')
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error visualisasi missing values: {e}")
        
        with col2:
            try:
                # Heatmap missing values
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=ax)
                ax.set_title('Pattern Missing Values (Heatmap)')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error heatmap missing values: {e}")

    # ANALISIS TANGGAL LENGKAP
    st.subheader("📅 Analisis Data Tanggal Lengkap")

    # Identifikasi kolom tanggal - lebih robust
    date_cols = []
    potential_date_cols = []

    for col in df.columns:
        # Cek jika sudah datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        else:
            # Coba identifikasi kolom potensial
            sample_size = min(100, len(df[col].dropna()))
            if sample_size > 0:
                sample = df[col].dropna().head(sample_size)
                
                # Cek berbagai format tanggal
                date_patterns = [
                    r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}',  # YYYY-MM-DD, DD/MM/YYYY, dll
                    r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                    r'\d{4}-\d{2}-\d{2}',  # ISO format
                    r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
                    r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
                ]
                
                date_like = False
                for pattern in date_patterns:
                    if sample.astype(str).str.match(pattern).any():
                        date_like = True
                        break
                
                if date_like:
                    potential_date_cols.append(col)

    if date_cols:
        st.success(f"✅ **Kolom tanggal yang terdeteksi:** {', '.join(date_cols)}")
        
        for col in date_cols:
            st.markdown(f"#### 📊 Analisis Mendalam untuk `{col}`")
            
            # Pastikan kolom dalam format datetime
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Hapus nilai NaN yang mungkin muncul dari konversi
            date_series = df[col].dropna()
            
            if len(date_series) == 0:
                st.warning(f"Tidak ada data tanggal yang valid di kolom {col}")
                continue
            
            # Container untuk statistik dasar
            col1, col2 = st.columns(2)
            
            with col1:
                # Statistik dasar tanggal
                st.write("**📋 Statistik Dasar:**")
                date_stats_data = {
                    'Metrik': ['Tanggal Terawal', 'Tanggal Terakhir', 'Rentang Waktu', 'Jumlah Hari', 'Data Valid', 'Data Invalid'],
                    'Nilai': [
                        date_series.min().strftime('%Y-%m-%d'),
                        date_series.max().strftime('%Y-%m-%d'),
                        f"{(date_series.max() - date_series.min()).days} hari",
                        (date_series.max() - date_series.min()).days,
                        len(date_series),
                        len(df[col]) - len(date_series)
                    ]
                }
                date_stats = pd.DataFrame(date_stats_data)
                st.dataframe(date_stats, use_container_width=True, hide_index=True)
            
            with col2:
                # Analisis komponen tanggal
                st.write("**🔍 Distribusi Komponen Tanggal:**")
                
                # Ekstrak komponen tanggal
                year_counts = date_series.dt.year.value_counts().sort_index()
                month_counts = date_series.dt.month.value_counts().sort_index()
                day_counts = date_series.dt.day.value_counts().sort_index()
                dow_counts = date_series.dt.dayofweek.value_counts().sort_index()
                
                # Mapping untuk nama
                month_names = {1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 
                            5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
                            9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'}
                day_names = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 
                            4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
                
                comp_data = {
                    'Komponen': ['Tahun', 'Bulan', 'Hari', 'Hari dalam Minggu'],
                    'Jumlah Unik': [
                        year_counts.shape[0],
                        month_counts.shape[0],
                        day_counts.shape[0],
                        dow_counts.shape[0]
                    ],
                    'Nilai Terbanyak': [
                        f"{year_counts.index[0]} ({year_counts.iloc[0]} data)",
                        f"{month_names.get(month_counts.index[0], month_counts.index[0])} ({month_counts.iloc[0]} data)",
                        f"{day_counts.index[0]} ({day_counts.iloc[0]} data)",
                        f"{day_names.get(dow_counts.index[0], dow_counts.index[0])} ({dow_counts.iloc[0]} data)"
                    ]
                }
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            # Visualisasi trend waktu
            st.write("**📈 Trend Data Berdasarkan Waktu:**")
            
            trend_col1, trend_col2 = st.columns(2)
            
            with trend_col1:
                # Frekuensi per bulan
                try:
                    monthly_count = date_series.dt.to_period('M').value_counts().sort_index()
                    monthly_count.index = monthly_count.index.astype(str)
                    if not monthly_count.empty:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        monthly_count.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                        ax.set_title(f'Frekuensi Data per Bulan - {col}')
                        ax.set_xlabel('Bulan-Tahun')
                        ax.set_ylabel('Jumlah Data')
                        ax.tick_params(axis='x', rotation=45)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error membuat chart bulanan: {e}")
            
            with trend_col2:
                # Frekuensi per tahun
                try:
                    yearly_count = date_series.dt.year.value_counts().sort_index()
                    if not yearly_count.empty:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        yearly_count.plot(kind='bar', ax=ax, color='lightgreen', edgecolor='black')
                        ax.set_title(f'Frekuensi Data per Tahun - {col}')
                        ax.set_xlabel('Tahun')
                        ax.set_ylabel('Jumlah Data')
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error membuat chart tahunan: {e}")
            
            # Analisis musiman/harian
            st.write("**🌍 Analisis Musiman dan Harian:**")
            
            seasonal_col1, seasonal_col2 = st.columns(2)
            
            with seasonal_col1:
                # Distribusi per bulan (Pie chart)
                try:
                    month_names_list = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 
                                    'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
                    monthly_dist = date_series.dt.month.value_counts().sort_index()
                    if len(monthly_dist) > 0:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Pie chart
                        monthly_dist_pie = monthly_dist.copy()
                        monthly_dist_pie.index = [month_names_list[i-1] for i in monthly_dist_pie.index]
                        ax1.pie(monthly_dist_pie.values, labels=monthly_dist_pie.index, autopct='%1.1f%%', startangle=90)
                        ax1.set_title(f'Distribusi per Bulan - {col}')
                        
                        # Bar chart
                        monthly_dist_pie.plot(kind='bar', ax=ax2, color='coral', edgecolor='black')
                        ax2.set_title(f'Distribusi per Bulan - {col}')
                        ax2.set_ylabel('Jumlah Data')
                        ax2.tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error analisis bulanan: {e}")
            
            with seasonal_col2:
                # Distribusi hari dalam minggu
                try:
                    day_names_list = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
                    dow_dist = date_series.dt.dayofweek.value_counts().sort_index()
                    if len(dow_dist) > 0:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Pie chart
                        dow_dist_pie = dow_dist.copy()
                        dow_dist_pie.index = [day_names_list[i] for i in dow_dist_pie.index]
                        ax1.pie(dow_dist_pie.values, labels=dow_dist_pie.index, autopct='%1.1f%%', startangle=90)
                        ax1.set_title(f'Distribusi Hari dalam Minggu - {col}')
                        
                        # Bar chart
                        dow_dist_pie.plot(kind='bar', ax=ax2, color='gold', edgecolor='black')
                        ax2.set_title(f'Distribusi Hari dalam Minggu - {col}')
                        ax2.set_ylabel('Jumlah Data')
                        ax2.tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error analisis hari: {e}")
            
            # Analisis quarter/triwulan
            st.write("**📊 Analisis per Triwulan:**")
            try:
                quarter_dist = date_series.dt.quarter.value_counts().sort_index()
                if not quarter_dist.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        quarter_names = {1: 'Q1 (Jan-Mar)', 2: 'Q2 (Apr-Jun)', 
                                    3: 'Q3 (Jul-Sep)', 4: 'Q4 (Okt-Des)'}
                        quarter_dist.index = [quarter_names[i] for i in quarter_dist.index]
                        quarter_dist.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                        ax.set_ylabel('')  # Remove ylabel for pie chart
                        ax.set_title(f'Distribusi per Triwulan - {col}')
                        st.pyplot(fig)
                    
                    with col2:
                        st.dataframe(pd.DataFrame({
                            'Triwulan': quarter_dist.index,
                            'Jumlah Data': quarter_dist.values,
                            'Persentase': (quarter_dist.values / len(date_series) * 100).round(2)
                        }), use_container_width=True)
            except Exception as e:
                st.error(f"Error analisis triwulan: {e}")
            
            # Deteksi missing dates
            st.write("**🔎 Analisis Kelengkapan Tanggal:**")
            try:
                if len(date_series) > 1:
                    date_range = pd.date_range(start=date_series.min(), end=date_series.max(), freq='D')
                    missing_dates = date_range.difference(date_series)
                    
                    completeness_info = pd.DataFrame({
                        'Metrik': ['Total Hari dalam Rentang', 'Hari dengan Data', 'Hari Tanpa Data', 'Persentase Kelengkapan'],
                        'Nilai': [
                            len(date_range),
                            len(date_range) - len(missing_dates),
                            len(missing_dates),
                            f"{((len(date_range) - len(missing_dates)) / len(date_range) * 100):.2f}%"
                        ]
                    })
                    st.dataframe(completeness_info, use_container_width=True, hide_index=True)
                    
                    if len(missing_dates) > 0:
                        st.warning(f"⚠️ Terdapat {len(missing_dates)} hari tanpa data")
                        if len(missing_dates) <= 20:
                            st.write("**Tanggal yang hilang:**", missing_dates.strftime('%Y-%m-%d').tolist())
                        else:
                            st.write(f"**Contoh 20 tanggal yang hilang:**", missing_dates[:20].strftime('%Y-%m-%d').tolist())
                    
                    # Visualisasi kelengkapan
                    if len(date_range) > 0:
                        completeness_ratio = (len(date_range) - len(missing_dates)) / len(date_range)
                        
                        fig, ax = plt.subplots(figsize=(8, 2))
                        ax.barh(['Kelengkapan'], [completeness_ratio * 100], color='lightblue', height=0.5)
                        ax.barh(['Kelengkapan'], [100 - completeness_ratio * 100], 
                            left=[completeness_ratio * 100], color='lightcoral', height=0.5)
                        ax.set_xlim(0, 100)
                        ax.set_xlabel('Persentase (%)')
                        ax.set_title(f'Kelengkapan Data Tanggal - {col}')
                        ax.text(completeness_ratio * 100 / 2, 0, f'{completeness_ratio*100:.1f}% Terisi', 
                            ha='center', va='center', color='black', fontweight='bold')
                        ax.text(completeness_ratio * 100 + (100 - completeness_ratio * 100) / 2, 0, 
                            f'{(100 - completeness_ratio*100):.1f}% Kosong', 
                            ha='center', va='center', color='black', fontweight='bold')
                        st.pyplot(fig)
                else:
                    st.info("Data tanggal terlalu sedikit untuk analisis kelengkapan")
            except Exception as e:
                st.error(f"Error analisis kelengkapan: {e}")
            
            st.markdown("---")
            
    else:
        st.info("❌ Tidak ada kolom tanggal yang terdeteksi dalam dataset.")
        
        # Analisis kolom potensial
        if potential_date_cols:
            st.write("**🔍 Kolom yang mungkin berisi tanggal:**")
            potential_info = []
            
            for col in potential_date_cols:
                sample = df[col].dropna().head(5)
                unique_count = df[col].nunique()
                null_count = df[col].isnull().sum()
                
                potential_info.append({
                    'Kolom': col,
                    'Tipe Data': str(df[col].dtype),
                    'Contoh Nilai': sample.iloc[0] if len(sample) > 0 else 'N/A',
                    'Nilai Unik': unique_count,
                    'Null Values': null_count,
                    'Saran': 'Coba konversi ke datetime'
                })
            
            if potential_info:
                potential_df = pd.DataFrame(potential_info)
                st.dataframe(potential_df, use_container_width=True)
                
                st.write("**💡 Tips Konversi:**")
                st.code("""
    # Untuk konversi manual:
    df['nama_kolom'] = pd.to_datetime(df['nama_kolom'], errors='coerce')

    # Dengan format spesifik:
    df['nama_kolom'] = pd.to_datetime(df['nama_kolom'], format='%Y-%m-%d', errors='coerce')
                """)
        
        # Analisis data kategorikal jika tidak ada tanggal
        st.write("**📋 Analisis Data Kategorikal sebagai Alternatif:**")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            for col in categorical_cols[:3]:  # Batasi 3 kolom pertama
                st.write(f"**Analisis untuk `{col}`**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Value counts
                    value_counts = df[col].value_counts()
                    st.dataframe(value_counts.head(10), use_container_width=True)
                
                with col2:
                    # Pie chart untuk top categories
                    try:
                        top_categories = value_counts.head(5)
                        if len(top_categories) > 0:
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.pie(top_categories.values, labels=top_categories.index, autopct='%1.1f%%')
                            ax.set_title(f'Top 5 Kategori - {col}')
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error pie chart {col}: {e}")
                
                st.markdown("---")
        else:
            st.info("Tidak ada kolom kategorikal yang tersedia untuk analisis alternatif.")

    # Tambahan: Summary statistics
    st.subheader("📊 Summary Dataset")
    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Total Baris", len(df))
        st.metric("Total Kolom", len(df.columns))

    with summary_col2:
        numeric_count = len(numeric_cols) if 'numeric_cols' in locals() else len(df.select_dtypes(include=[np.number]).columns)
        categorical_count = len(df.select_dtypes(include=['object']).columns)
        st.metric("Kolom Numerik", numeric_count)
        st.metric("Kolom Kategorikal", categorical_count)

    with summary_col3:
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        completeness = ((total_cells - total_missing) / total_cells * 100)
        st.metric("Total Missing Values", total_missing)
        st.metric("Kelengkapan Dataset", f"{completeness:.1f}%")

# Cache untuk file contoh
@st.cache_data(show_spinner=False)
def create_sample_file():
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    np.random.seed(42)
    price_changes = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * (1 + price_changes).cumprod()
    
    volumes = np.random.randint(1000, 10000, len(dates))
    
    example_data = pd.DataFrame({
        'Tanggal': dates,
        'Open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
        'High': prices * (1 + np.abs(np.random.normal(0.005, 0.01, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.normal(0.005, 0.01, len(dates)))),
        'Close': prices,
        'Volume': volumes,
        'Target_Sales': np.random.randint(5000, 15000, len(dates)),
        'Actual_Sales': np.random.randint(4000, 16000, len(dates)),
        'Perusahaan': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN'], len(dates)),
        'Sektor': np.random.choice(['Teknologi', 'Kesehatan', 'Finansial', 'Konsumsi'], len(dates)),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
        'Kategori_Produk': np.random.choice(['Laptop', 'Smartphone', 'Tablet', 'Aksesori'], len(dates))
    })
    
    for i in range(len(example_data)):
        example_data.loc[i, 'High'] = max(example_data.loc[i, 'Open'], example_data.loc[i, 'Close'], example_data.loc[i, 'High'])
        example_data.loc[i, 'Low'] = min(example_data.loc[i, 'Open'], example_data.loc[i, 'Close'], example_data.loc[i, 'Low'])
    
    return example_data

# UI utama
st.markdown("Unggah file CSV atau Excel untuk melihat visualisasi dan statistik data.")

# Sidebar
st.sidebar.header("🎛️ Kontrol Aplikasi")

if st.sidebar.button("📁 Buat File Contoh"):
    example_data = create_sample_file()
    csv = example_data.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Unduh Contoh CSV",
        data=csv,
        file_name="contoh_data_saham.csv",
        mime="text/csv"
    )

# Upload file
st.sidebar.header("📤 Unggah & Gabungkan Beberapa File")
uploaded_files = st.sidebar.file_uploader(
    "Pilih file CSV atau Excel (bisa multiple)",
    type=['csv', 'xlsx', 'xls'],
    accept_multiple_files=True
)

merge_method = "concat"
if uploaded_files and len(uploaded_files) > 1:
    merge_method = st.sidebar.selectbox(
        "Metode Penggabungan Data",
        ["concat", "inner", "outer", "left", "right"],
        key="merge_method_select"
    )

# Proses file
df = None
if uploaded_files:
    datasets = []
    for uploaded_file in uploaded_files:
        dataset = process_uploaded_file(uploaded_file)
        if dataset is not None:
            datasets.append(dataset)
    
    if datasets:
        if len(datasets) == 1:
            df = datasets[0]
        else:
            df = merge_datasets(datasets, merge_method)

# Tampilkan konten berdasarkan ketersediaan data
if df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistik", "📈 Visualisasi", "💾 Data", "ℹ️ Informasi"])
    
    with tab4:
        st.subheader("📋 Informasi Penggunaan")
        st.markdown("""
        ### Cara Penggunaan Aplikasi
        
        1. **Unggah Data**: Gunakan menu sidebar untuk mengunggah file data dalam format CSV atau Excel
        2. **Statistik**: Lihat ringkasan statistik deskriptif di tab Statistik
        3. **Visualisasi**: Eksplorasi data melalui berbagai chart dan grafik
        4. **Data Mentah**: Periksa data asli dalam format tabel
        
        ### Fitur Utama
        - Analisis statistik deskriptif
        - Visualisasi data interaktif
        - Ekspor hasil analisis
        - Pembersihan data otomatis
        """)
        
        st.subheader("🧮 Rumus Matematika dan Metode Perhitungan")
        
        # Statistical Formulas
        st.markdown("""
        ### Statistik Deskriptif - Rumus dan Perhitungan
        
        #### 1. Mean (Rata-rata)
        **Rumus:**
        ```
        μ = (Σxᵢ) / n
        ```
        **Keterangan:**
        - μ = mean populasi
        - xᵢ = nilai data ke-i
        - n = jumlah data
        - Σ = simbol penjumlahan
        
        **Contoh Perhitungan:**
        Data: [2, 4, 6, 8, 10]
        μ = (2 + 4 + 6 + 8 + 10) / 5 = 30 / 5 = 6
        
        #### 2. Median
        **Langkah Perhitungan:**
        1. Urutkan data dari terkecil ke terbesar
        2. Jika n ganjil: Median = nilai tengah
        3. Jika n genap: Median = rata-rata dua nilai tengah
        
        **Rumus:**
        - n ganjil: Median = x₍ₙ₊₁₎/₂
        - n genap: Median = (x₍ₙ/₂₎ + x₍ₙ/₂₊₁₎) / 2
        
        **Contoh:**
        Data ganjil: [1, 3, 3, 6, 7, 8, 9] → Median = 6
        Data genap: [1, 2, 3, 4, 5, 6, 8, 9] → Median = (4 + 5)/2 = 4.5
        
        #### 3. Modus
        Nilai yang paling sering muncul dalam dataset
        
        #### 4. Standar Deviasi
        **Rumus Populasi:**
        ```
        σ = √[Σ(xᵢ - μ)² / n]
        ```
        
        **Rumus Sampel:**
        ```
        s = √[Σ(xᵢ - x̄)² / (n-1)]
        ```
        
        **Langkah Perhitungan:**
        1. Hitung mean (x̄)
        2. Hitung selisih setiap data dengan mean: (xᵢ - x̄)
        3. Kuadratkan selisih: (xᵢ - x̄)²
        4. Jumlahkan semua kuadrat selisih: Σ(xᵢ - x̄)²
        5. Bagi dengan (n-1) untuk sampel
        6. Akar kuadrat hasilnya
        
        **Contoh:**
        Data: [2, 4, 4, 4, 5, 5, 7, 9]
        Mean = 5
        Variance = [(2-5)² + (4-5)² + ... + (9-5)²] / (8-1) = 32/7 ≈ 4.57
        Standar Deviasi = √4.57 ≈ 2.14
        """)
        
        # Additional Statistical Measures
        st.markdown("""
        #### 5. Variance (Ragam)
        **Rumus Sampel:**
        ```
        s² = Σ(xᵢ - x̄)² / (n-1)
        ```
        
        #### 6. Koefisien Korelasi Pearson
        **Rumus:**
        ```
        r = [Σ(xᵢ - x̄)(yᵢ - ȳ)] / [√Σ(xᵢ - x̄)² × √Σ(yᵢ - ȳ)²]
        ```
        
        **Interpretasi:**
        - r = +1: Korelasi positif sempurna
        - r = -1: Korelasi negatif sempurna
        - r = 0: Tidak ada korelasi
        
        #### 7. Regresi Linear Sederhana
        **Persamaan:**
        ```
        y = a + bx
        ```
        
        **Koefisien:**
        ```
        b = [Σ(xᵢ - x̄)(yᵢ - ȳ)] / Σ(xᵢ - x̄)²
        a = ȳ - bx̄
        ```
        """)
        
        # Probability Distributions
        st.markdown("""
        ### Distribusi Probabilitas
        
        #### Distribusi Normal
        **Fungsi Kepadatan Probabilitas:**
        ```
        f(x) = (1 / (σ√(2π))) × e^(-(x-μ)² / (2σ²))
        ```
        
        #### Distribusi Binomial
        **Rumus:**
        ```
        P(X=k) = C(n,k) × pᵏ × (1-p)ⁿ⁻ᵏ
        ```
        **Dimana:**
        - C(n,k) = n! / (k!(n-k)!)
        - p = probabilitas sukses
        - n = jumlah percobaan
        - k = jumlah sukses
        """)
        
        st.subheader("📄 Hak Lisensi")
        st.markdown("""
STRICT PROPRIETARY SOFTWARE LICENSE — ALL RIGHTS RESERVED (INDONESIA)
==================================================================

Hak Cipta © 2025 Dwi Bakti N Dev. Seluruh hak dilindungi undang-undang.

DEFINISI
--------
1. "Pemilik" berarti Dwi Bakti N Dev, pemilik semua hak atas Perangkat Lunak.
2. "Perangkat Lunak" berarti seluruh kode sumber, kode objek, dokumentasi, gambar, aset, skrip, build, dan materi lain yang disediakan di bawah lisensi ini, termasuk setiap pembaruan, patch, atau modifikasi apa pun.
3. "Pihak Ketiga" berarti setiap orang, badan, entitas, atau organisasi selain Pemilik.
4. "Pengguna" berarti pihak yang mendapatkan salinan Perangkat Lunak berdasarkan ketentuan lisensi ini.

PENDAHULUAN
----------
Perangkat Lunak ini diberikan oleh Pemilik hanya berdasarkan ketentuan-ketentuan di bawah ini. Jika Pengguna tidak menyetujui semua ketentuan ini, Pengguna tidak berhak menggunakan, menyalin, atau memiliki akses ke Perangkat Lunak.

HAK CIPTA & HAK MILIK
----------------------
Pemilik memegang seluruh hak kekayaan intelektual atas Perangkat Lunak, termasuk namun tidak terbatas pada hak cipta, hak basis data, dan hak terkait. Perangkat Lunak dilindungi oleh hukum hak cipta Indonesia dan perjanjian internasional yang berlaku. **Tidak ada hak kepemilikan atau hak eksplisit/implisit yang diberikan kepada Pengguna**, kecuali hak terbatas yang diberikan secara tegas di bawah ini.

PEMBERIAN LISENSI (SANGAT TERBATAS)
----------------------------------
Dengan lisensi ini, Pemilik memberikan hak non-eksklusif, tidak dapat dipindahtangankan, dan tidak dapat disublicense kepada Pengguna untuk **mengakses dan menggunakan Perangkat Lunak hanya untuk tujuan internal evaluasi** (atau tujuan lain yang ditetapkan secara tertulis oleh Pemilik), selama dan sejauh Pengguna mematuhi ketentuan berikut. Hak yang tidak disebutkan secara tegas di sini **dilarang**.

PEMBATASAN PENGGUNAAN (PROHIBITED USES)
----------------------------------------
Tanpa izin tertulis terpisah dari Pemilik, Pengguna dilarang:
a. Menyalin, mereproduksi, atau membuat karya turunan dari Perangkat Lunak (seluruh atau sebagian).  
b. Menyebarluaskan, menjual, menyewakan, meminjami, memberi lisensi ulang, atau membuat tersedia bagi publik Perangkat Lunak atau bagian darinya.  
c. Menggunakan Perangkat Lunak untuk menyediakan jasa komersial (SaaS, hosting, layanan pihak ketiga) yang menyalurkan fungsionalitas Perangkat Lunak kepada pihak lain.  
d. Membongkar (reverse engineer), mendekompilasi, atau mencoba memperoleh kode sumber (kecuali diizinkan secara tegas oleh hukum yang berlaku dan hanya sejauh hukum tersebut memaksakan hak tersebut).  
e. Menghapus, mengubah, atau menutupi pemberitahuan hak cipta, tanda tanggung jawab (attribution), atau label kepemilikan lainnya.  
f. Mengalihkan, memindahkan, atau mentransfer Perangkat Lunak atau hak atasnya kepada pihak lain tanpa izin tertulis dari Pemilik.

KEWAJIBAN PENGGUNA
-------------------
Pengguna harus:
1. Menjaga kerahasiaan Perangkat Lunak dan tidak mengungkapkannya kepada pihak ketiga tanpa persetujuan tertulis dari Pemilik.  
2. Segera melaporkan kepada Pemilik setiap penyalahgunaan, pelanggaran keamanan, atau indikasi pelanggaran hak cipta oleh pihak ketiga.  
3. Mematuhi semua peraturan ekspor/impor dan pembatasan hukum terkait kriptografi, enkripsi, atau teknologi lainnya.

PEMBEBASAN JAMINAN
------------------
Perangkat Lunak disediakan "SEBAGAIMANA ADANYA" tanpa jaminan apa pun, baik tersurat ataupun tersirat, termasuk namun tidak terbatas pada jaminan kelayakan untuk tujuan tertentu, non-pelanggaran, atau jaminan terkait keamanan. Pemilik tidak menjamin bahwa Perangkat Lunak akan memenuhi kebutuhan Pengguna atau bebas dari bug.

BATASAN TANGGUNG JAWAB
----------------------
Dalam keadaan apa pun, Pemilik tidak bertanggung jawab atas kerugian langsung, tidak langsung, insidental, khusus, emergen, atau konsekuensial yang timbul dari penggunaan atau ketidakmampuan untuk menggunakan Perangkat Lunak, bahkan jika Pemilik telah diberitahu mengenai kemungkinan kerugian tersebut.

PENGGANTI RUGI
--------------
Pengguna setuju untuk mengganti rugi, membela, dan membebaskan Pemilik dari klaim, kerugian, tanggung jawab, biaya, atau pengeluaran (termasuk biaya pengacara) yang timbul akibat pelanggaran ketentuan lisensi ini oleh Pengguna atau pelanggaran hukum yang dilakukan oleh Pengguna.

PENEGAKAN & SANKSI
------------------
Pelanggaran terhadap lisensi ini akan menyebabkan pengakhiran hak penggunaan secara otomatis dan/atau tindakan hukum sesuai undang-undang. Pemilik berhak menuntut ganti rugi, perintah pengadilan (injunctive relief), dan pemulihan kerugian yang timbul akibat pelanggaran.

PEMBATALAN LISENSI
------------------
Pemilik dapat mengakhiri lisensi ini jika Pengguna melanggar ketentuan apa pun. Setelah berakhirnya lisensi, Pengguna harus menghentikan semua penggunaan Perangkat Lunak dan menghancurkan semua salinan (termasuk salinan cadangan), serta memberikan pernyataan tertulis kepada Pemilik yang mengonfirmasi penghancuran tersebut.

PEMBATASAN TANGGUNG JAWAB HUKUM
-------------------------------
Jika undang-undang yang berlaku tidak mengizinkan pembatasan tertentu yang tercantum di atas, ketentuan tersebut akan diberlakukan sejauh diizinkan oleh hukum, sedangkan ketentuan lainnya tetap berlaku penuh dan mengikat.

HUKUM YANG MENGATUR & YURISDIKSI
--------------------------------
Lisensi ini diatur oleh dan ditafsirkan sesuai dengan hukum Republik Indonesia. Sengketa yang timbul sehubungan dengan lisensi ini akan diselesaikan terlebih dahulu melalui mediasi; jika gagal, sengketa akan diajukan ke pengadilan yang berwenang di [KOTA/DAERAH] (misalnya: Jakarta).

PEMBERITAHUAN & KONTAK
----------------------
Permintaan izin, pemberitahuan pelanggaran, atau komunikasi lain terkait lisensi ini harus diarahkan ke:
Nama Pemilik : Dwi Bakti N Dev
Nama Pengembang : Dwi Bakti N Dev
Pencipta : Dwi Bakti N Dev
Website : https://portofolio-dwi-bakti-n-dev-liard.vercel.app


KESELURUHAN PERJANJIAN
----------------------
Ketentuan lisensi ini merupakan keseluruhan perjanjian antara Pemilik dan Pengguna terkait Perangkat Lunak dan menggantikan semua perjanjian sebelumnya.

TANGGAL EFEKTIF
---------------
Tanggal efektif: 14/10/2025

        """)
        
        st.subheader("🔬 Informasi Penelitian")
        st.markdown("""
        ### Metodologi Penelitian
        
        **Sumber Data**
        - Data primer dari survei lapangan
        - Data sekunder dari publikasi resmi
        - Dataset open source terpercaya
        
        **Metode Analisis**
        - Statistik deskriptif (mean, median, modus, standar deviasi)
        - Analisis eksploratori data (EDA)
        - Visualisasi data untuk identifikasi pola
        - Validasi data dengan multiple methods
        
        **Metode Statistik yang Digunakan:**
        
        1. **Uji Normalitas** - Shapiro-Wilk test
        ```
        W = (Σaᵢ × x₍ᵢ₎)² / Σ(xᵢ - x̄)²
        ```
        
        2. **Uji Hipotesis** - t-test
        ```
        t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
        ```
        
        3. **Analisis Varians (ANOVA)**
        ```
        F = (Varians antar grup) / (Varians dalam grup)
        ```
        
        """)
        
        # Tambahan informasi kontak
        st.subheader("📞 Kontak")
        st.markdown("""
        Untuk pertanyaan lebih lanjut mengenai penelitian atau penggunaan aplikasi:
          
        **Website**: https://portofolio-dwi-bakti-n-dev-liard.vercel.app
        **Nama Developer**: Dwi Bakti N Dev
        """)
        
        # Additional mathematical resources
        st.subheader("📚 Sumber Belajar Statistik")
        st.markdown("""
        ### Buku Referensi Recommended:
        
        1. **"Statistics for Data Science"** - James et al.
        - Bab 2: Descriptive Statistics
        - Bab 3: Probability Distributions
        - Bab 4: Statistical Inference
        
        2. **"Introduction to Probability"** - Bertsekas
        - Fundamental probability theory
        - Random variables and distributions
        
        3. **"The Elements of Statistical Learning"** - Hastie et al.
        - Advanced statistical methods
        - Machine learning applications
        
        ### Online Resources:
        - Khan Academy - Statistics and Probability
        - MIT OpenCourseWare - Introduction to Probability and Statistics
        - Coursera - Data Science and Statistical Analysis
        """)
    
    with tab1:
        show_optimized_statistics(df)
    
    with tab2:
        st.header("🎨 Visualisasi Data")
        create_all_visualizations(df)
    
    with tab3:
        st.header("📋 Data Lengkap")
        
        # Tampilkan dataframe
        st.dataframe(df, use_container_width=True)
        
        # Section untuk statistik deskriptif
        st.subheader("📊 Statistik Deskriptif")
        
        # Pilih kolom numerik
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_columns:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Jumlah Data", len(df))
                st.metric("Kolom Numerik", len(numeric_columns))
            
            with col2:
                if numeric_columns:
                    total_values = df[numeric_columns].sum().sum()
                    st.metric("Total Semua Nilai", f"{total_values:,.2f}")
            
            with col3:
                if numeric_columns:
                    avg_value = df[numeric_columns].mean().mean()
                    st.metric("Rata-rata Keseluruhan", f"{avg_value:,.2f}")
            
            with col4:
                if numeric_columns:
                    std_value = df[numeric_columns].std().mean()
                    st.metric("Std Dev Rata-rata", f"{std_value:,.2f}")
        
        # Rumus Matematika yang Digunakan
        st.subheader("🧮 Rumus Matematika yang Digunakan")
        
        rumus_col1, rumus_col2 = st.columns(2)
        
        with rumus_col1:
            st.markdown("""
            **Statistik Dasar:**
            - **Mean (Rata-rata):** 
            ```
            μ = (Σx) / n
            ```
            - **Standar Deviasi:**
            ```
            σ = √[Σ(x - μ)² / (n-1)]
            ```
            - **Varians:**
            ```
            σ² = Σ(x - μ)² / (n-1)
            ```
            """)
        
        with rumus_col2:
            st.markdown("""
            **Akumulasi & Aggregasi:**
            - **Total Kumulatif:**
            ```
            S = Σx_i
            ```
            - **Rata-rata Bergerak:**
            ```
            MA = (x_t + x_{t-1} + ... + x_{t-n+1}) / n
            ```
            - **Pertumbuhan:**
            ```
            Growth = (Akhir - Awal) / Awal × 100%
            ```
            """)
        
        # Visualisasi Data
        st.subheader("📈 Visualisasi Data")
        
        if numeric_columns:
            # Pilih kolom untuk visualisasi
            selected_column = st.selectbox("Pilih Kolom untuk Visualisasi:", numeric_columns)
            
            if selected_column:
                # Chart untuk distribusi data
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(df.index, df[selected_column], marker='o', linewidth=2, markersize=4)
                ax1.set_title(f'Trend Data - {selected_column}')
                ax1.set_xlabel('Index')
                ax1.set_ylabel(selected_column)
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                
                # Histogram
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.hist(df[selected_column], bins=20, alpha=0.7, edgecolor='black')
                ax2.set_title(f'Distribusi Data - {selected_column}')
                ax2.set_xlabel(selected_column)
                ax2.set_ylabel('Frekuensi')
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
        
        # Analisis Akumulasi
        st.subheader("📊 Analisis Akumulasi")
        
        if numeric_columns:
            # Pilih kolom untuk analisis akumulasi
            accum_column = st.selectbox("Pilih Kolom untuk Analisis Akumulasi:", numeric_columns, key="accum")
            
            if accum_column:
                # Hitung nilai kumulatif
                df_cumulative = df.copy()
                df_cumulative['Kumulatif'] = df_cumulative[accum_column].cumsum()
                df_cumulative['Rata-rata Bergerak (3)'] = df_cumulative[accum_column].rolling(window=3).mean()
                df_cumulative['Rata-rata Bergerak (5)'] = df_cumulative[accum_column].rolling(window=5).mean()
                
                # Tampilkan data akumulasi
                st.write("**Data dengan Akumulasi:**")
                st.dataframe(df_cumulative[[accum_column, 'Kumulatif', 'Rata-rata Bergerak (3)', 'Rata-rata Bergerak (5)']].tail(10), 
                            use_container_width=True)
                
                # Chart akumulasi
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                ax3.plot(df_cumulative.index, df_cumulative['Kumulatif'], 
                        label='Nilai Kumulatif', linewidth=2, color='blue')
                ax3.plot(df_cumulative.index, df_cumulative[accum_column], 
                        label=f'Nilai {accum_column}', linewidth=1, color='red', alpha=0.7)
                ax3.set_title(f'Akumulasi Data - {accum_column}')
                ax3.set_xlabel('Index')
                ax3.set_ylabel('Nilai')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)
        
        # Analisis Korelasi (jika ada multiple numeric columns)
        if len(numeric_columns) > 1:
            st.subheader("🔗 Analisis Korelasi")
            
            # Heatmap korelasi
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            correlation_matrix = df[numeric_columns].corr()
            im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax4.set_xticks(range(len(numeric_columns)))
            ax4.set_yticks(range(len(numeric_columns)))
            ax4.set_xticklabels(numeric_columns, rotation=45)
            ax4.set_yticklabels(numeric_columns)
            
            # Tambahkan nilai di setiap cell
            for i in range(len(numeric_columns)):
                for j in range(len(numeric_columns)):
                    ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                            ha='center', va='center', color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')
            
            ax4.set_title('Matriks Korelasi')
            plt.colorbar(im, ax=ax4)
            st.pyplot(fig4)
        
        # Summary Statistics Detail
        st.subheader("📋 Detail Statistik per Kolom")
        
        if numeric_columns:
            for col in numeric_columns:
                with st.expander(f"Statistik untuk {col}"):
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                        
                        with stats_col1:
                            st.metric("Mean", f"{col_data.mean():.2f}")
                            st.metric("Median", f"{col_data.median():.2f}")
                        
                        with stats_col2:
                            st.metric("Std Dev", f"{col_data.std():.2f}")
                            st.metric("Varians", f"{col_data.var():.2f}")
                        
                        with stats_col3:
                            st.metric("Min", f"{col_data.min():.2f}")
                            st.metric("Max", f"{col_data.max():.2f}")
                        
                        with stats_col4:
                            st.metric("Range", f"{col_data.max() - col_data.min():.2f}")
                            st.metric("Jumlah Data", len(col_data))
        
    st.sidebar.header("📥 Unduh Data")
    if st.sidebar.button("💾 Unduh Data yang Telah Diproses"):
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="📥 Unduh sebagai CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
        
else:
    st.info("Silakan unggah file CSV atau Excel melalui sidebar di sebelah kiri.")
    
    st.subheader("📋 Contoh Data")
    example_data = create_sample_file()
    st.dataframe(example_data.head())
    
    st.subheader("📊 Contoh Visualisasi Baru")
    col1, col2 = st.columns(2)
    
    with col1:
        # Contoh Histogram Responsif
        fig_hist = px.histogram(example_data, x='Close', title='Contoh Histogram Responsif',
                               color_discrete_sequence=['#636EFA'], opacity=0.8)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Contoh KPI Scorecard
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Rata-rata Close Price</div>
            <div class="kpi-value">105.42</div>
            <div class="kpi-title">Contoh KPI</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Contoh Treemap
        tree_data = example_data.groupby(['Sektor', 'Kategori_Produk'])['Volume'].sum().reset_index()
        fig_tree = px.treemap(tree_data, path=['Sektor', 'Kategori_Produk'], values='Volume',
                             title='Contoh Treemap')
        st.plotly_chart(fig_tree, use_container_width=True)

st.markdown("---")
st.markdown("Dashboard Statistik © 2025 - Dibuat oleh Dwi Bakti N Dev")
