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
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) == 1:
                df = pd.read_excel(uploaded_file)
                st.sidebar.success(f"Excel berhasil dibaca: {uploaded_file.name} (Sheet: {sheet_names[0]})")
            else:
                selected_sheet = st.sidebar.selectbox(
                    f"Pilih sheet dari {uploaded_file.name}",
                    sheet_names,
                    key=f"sheet_{uploaded_file.name}"
                )
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                st.sidebar.success(f"Excel berhasil dibaca: {uploaded_file.name} (Sheet: {selected_sheet})")
                
        else:
            st.error("Format file tidak didukung. Harap unggah file CSV atau Excel.")
            return None
        
        df.columns = df.columns.str.strip()
        df = auto_convert_dates(df)
        return df
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {str(e)}")
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
    
    # Pilihan jenis chart yang diperluas - TAMBAH PIE CHART DI SINI
    chart_types = [
        "📈 Grafik Garis (Line Chart)",
        "📊 Grafik Batang (Bar Chart)", 
        "📋 Histogram",
        "🔄 Kombinasi Grafik Garis & Batang",
        "🔵 Scatter Plot",
        "🫧 Grafik Gelembung (Bubble Chart)",
        "🎯 Grafik Gauge (Speedometer)",
        "🕷️ Grafik Radar (Spider Chart)",
        "📦 Diagram Bingkai (Box Plot)",
        "🍾 Grafik Corong (Funnel Chart)",
        "🥧 Pie Chart dengan Slider",  # TAMBAHAN BARU
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
                
            elif chart_type == "📋 Histogram":
                create_histogram(df, numeric_cols)
                
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
                
            elif chart_type == "🥧 Pie Chart dengan Slider":  # FUNGSI BARU
                create_pie_chart_with_slider(df, numeric_cols, non_numeric_cols)
                
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

# FUNGSI BARU: Pie Chart dengan Slider
def create_pie_chart_with_slider(df, numeric_cols, non_numeric_cols):
    st.subheader("🥧 Pie Chart dengan Slider Persentase")
    
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

# Fungsi-fungsi chart yang sudah ada (tetap sama seperti sebelumnya)
def create_line_chart(df, numeric_cols, non_numeric_cols):
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("Pilih kolom untuk sumbu X", 
                           [df.index.name if df.index.name else "index"] + non_numeric_cols + numeric_cols, 
                           key="line_x_col")
    with col2:
        y_col = st.selectbox("Pilih kolom untuk sumbu Y", numeric_cols, key="line_y_col")
    
    if x_col and y_col:
        fig = px.line(df, x=x_col, y=y_col, title=f"Grafik Garis: {y_col} over {x_col}")
        fig.update_traces(hovertemplate=f'<b>{y_col}</b><br>{x_col}: %{{x}}<br>Nilai: %{{y}}<extra></extra>')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Keterangan Grafik Garis"):
            st.markdown("""
            **Grafik Garis (Line Chart)** digunakan untuk menampilkan tren data over waktu atau urutan lainnya.
            - **Kelebihan**: Mudah melihat tren, pola, dan perubahan over time
            - **Kekurangan**: Tidak ideal untuk data kategorikal yang tidak berurutan
            - **Penggunaan**: Analisis time series, monitoring performa, trend analysis
            """)

def create_bar_chart(df, numeric_cols, non_numeric_cols):
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("Pilih kolom untuk sumbu X", non_numeric_cols if non_numeric_cols else numeric_cols, 
                           key="bar_x_col")
    with col2:
        y_col = st.selectbox("Pilih kolom untuk sumbu Y", numeric_cols, key="bar_y_col")
    
    if x_col and y_col:
        bar_data = df.groupby(x_col)[y_col].mean().reset_index()
        fig = px.bar(bar_data, x=x_col, y=y_col, title=f"Grafik Batang: Rata-rata {y_col} per {x_col}")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Keterangan Grafik Batang"):
            st.markdown("""
            **Grafik Batang (Bar Chart)** digunakan untuk membandingkan nilai antar kategori.
            - **Kelebihan**: Mudah membandingkan nilai antar kategori
            - **Kekurangan**: Tidak efektif untuk data dengan banyak kategori
            - **Penggunaan**: Perbandingan kategori, ranking, distribusi kategorikal
            """)

def create_histogram(df, numeric_cols):
    col = st.selectbox("Pilih kolom untuk histogram", numeric_cols, key="hist_col")
    
    if col:
        fig = px.histogram(df, x=col, title=f"Distribusi {col}", nbins=30)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Keterangan Histogram"):
            st.markdown("""
            **Histogram** menampilkan distribusi frekuensi data numerik.
            - **Kelebihan**: Menunjukkan bentuk distribusi, central tendency, dan spread
            - **Kekurangan**: Sensitif terhadap jumlah bins yang dipilih
            - **Penggunaan**: Analisis distribusi, identifikasi outlier, understanding data shape
            """)

# ... (fungsi-fungsi chart lainnya tetap sama)

def create_combined_chart(df, numeric_cols, non_numeric_cols):
    st.subheader("Kombinasi Grafik Garis dan Batang")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_col = st.selectbox("Pilih kolom untuk sumbu X", 
                           [df.index.name if df.index.name else "index"] + non_numeric_cols + numeric_cols, 
                           key="comb_x_col")
    with col2:
        line_col = st.selectbox("Pilih kolom untuk garis", numeric_cols, key="line_col")
    with col3:
        bar_col = st.selectbox("Pilih kolom untuk batang", numeric_cols, key="bar_col")
    
    if x_col and line_col and bar_col:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[line_col],
            mode='lines',
            name=line_col,
            yaxis='y1',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Bar(
            x=df[x_col],
            y=df[bar_col],
            name=bar_col,
            yaxis='y2',
            marker=dict(color='orange'),
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f"Kombinasi: {line_col} (Garis) dan {bar_col} (Batang)",
            xaxis=dict(title=x_col),
            yaxis=dict(title=line_col, side='left'),
            yaxis2=dict(title=bar_col, side='right', overlaying='y'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Keterangan Kombinasi Chart"):
            st.markdown("""
            **Kombinasi Grafik** menggabungkan dua jenis chart dalam satu visualisasi.
            - **Kelebihan**: Dapat menampilkan dua metrik dengan skala berbeda
            - **Kekurangan**: Bisa membingungkan jika tidak didesain dengan baik
            - **Penggunaan**: Analisis hubungan antara dua metrik dengan skala berbeda
            """)

def create_scatter_plot(df, numeric_cols, non_numeric_cols):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_col = st.selectbox("Pilih kolom X", numeric_cols, key="scatter_x")
    with col2:
        y_col = st.selectbox("Pilih kolom Y", numeric_cols, key="scatter_y")
    with col3:
        color_col = st.selectbox("Pilih kolom warna", [None] + non_numeric_cols, key="scatter_color")
    
    if x_col and y_col:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                        title=f"Scatter Plot: {y_col} vs {x_col}")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Keterangan Scatter Plot"):
            st.markdown("""
            **Scatter Plot** menampilkan hubungan antara dua variabel numerik.
            - **Kelebihan**: Menunjukkan korelasi dan pola hubungan
            - **Kekurangan**: Tidak efektif untuk data kategorikal
            - **Penggunaan**: Analisis korelasi, identifikasi cluster, deteksi outlier
            """)

def create_bubble_chart(df, numeric_cols, non_numeric_cols):
    st.subheader("Grafik Gelembung (Bubble Chart)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_col = st.selectbox("Pilih kolom X", numeric_cols, key="bubble_x")
    with col2:
        y_col = st.selectbox("Pilih kolom Y", numeric_cols, key="bubble_y")
    with col3:
        size_col = st.selectbox("Pilih kolom ukuran", numeric_cols, key="bubble_size")
    with col4:
        color_col = st.selectbox("Pilih kolom warna", [None] + non_numeric_cols, key="bubble_color")
    
    if x_col and y_col and size_col:
        fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col,
                        hover_name=df.index if df.index.name else None,
                        title=f"Bubble Chart: {y_col} vs {x_col}")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Keterangan Bubble Chart"):
            st.markdown("""
            **Bubble Chart** adalah scatter plot dengan dimensi ketiga (ukuran gelembung).
            - **Kelebihan**: Menampilkan tiga dimensi data sekaligus
            - **Kekurangan**: Bisa sulit dibaca jika terlalu banyak gelembung
            - **Penggunaan**: Analisis tiga variabel, comparison dengan multiple dimensions
            """)

def create_gauge_chart(df, numeric_cols):
    st.subheader("Grafik Gauge (Speedometer)")
    
    value_col = st.selectbox("Pilih kolom nilai", numeric_cols, key="gauge_value")
    max_value = st.number_input("Nilai maksimum gauge", value=100, key="gauge_max")
    
    if value_col:
        avg_value = df[value_col].mean()
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Rata-rata {value_col}"},
            delta = {'reference': df[value_col].median()},
            gauge = {
                'axis': {'range': [None, max_value]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, max_value/3], 'color': "lightgray"},
                    {'range': [max_value/3, 2*max_value/3], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9}}
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Keterangan Gauge Chart"):
            st.markdown("""
            **Gauge Chart** menampilkan nilai metrik dalam format speedometer.
            - **Kelebihan**: Visual yang intuitif untuk KPI dan progress
            - **Kekurangan**: Hanya menampilkan satu nilai utama
            - **Penggunaan**: Dashboard KPI, monitoring target, progress tracking
            """)

def create_radar_chart(df, numeric_cols, non_numeric_cols):
    st.subheader("Grafik Radar (Spider Chart)")
    
    category_col = st.selectbox("Pilih kolom kategori", non_numeric_cols, key="radar_category")
    selected_cols = st.multiselect("Pilih kolom nilai", numeric_cols, 
                                 default=numeric_cols[:min(5, len(numeric_cols))], 
                                 key="radar_values")
    
    if category_col and selected_cols and len(selected_cols) >= 3:
        radar_data = df.groupby(category_col)[selected_cols].mean().reset_index()
        
        fig = go.Figure()
        
        for idx, row in radar_data.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row[selected_cols].values.tolist() + [row[selected_cols].values[0]],
                theta=selected_cols + [selected_cols[0]],
                fill='toself',
                name=row[category_col]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Keterangan Radar Chart"):
            st.markdown("""
            **Radar Chart** menampilkan multiple variabel dalam format radial.
            - **Kelebihan**: Dapat membandingkan profil multi-variabel
            - **Kekurangan**: Sulit dibaca jika terlalu banyak variabel/kategori
            - **Penggunaan**: Profil kompetensi, comparison multi-attribute, performance analysis
            """)

def create_box_plot(df, numeric_cols):
    st.subheader("Diagram Bingkai (Box Plot)")
    
    col = st.selectbox("Pilih kolom untuk box plot", numeric_cols, key="box_col")
    
    if col:
        fig = px.box(df, y=col, title=f"Box Plot {col}")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Keterangan Box Plot"):
            st.markdown("""
            **Box Plot** menampilkan distribusi statistik data.
            - **Kelebihan**: Menunjukkan median, quartile, dan outlier
            - **Kekurangan**: Tidak menunjukkan bentuk distribusi secara detail
            - **Penggunaan**: Analisis distribusi, identifikasi outlier, comparison distribusi
            """)

def create_funnel_chart(df, numeric_cols, non_numeric_cols):
    st.subheader("Grafik Corong (Funnel Chart)")
    
    stage_col = st.selectbox("Pilih kolom stage", non_numeric_cols, key="funnel_stage")
    value_col = st.selectbox("Pilih kolom nilai", numeric_cols, key="funnel_value")
    
    if stage_col and value_col:
        funnel_data = df.groupby(stage_col)[value_col].sum().reset_index()
        funnel_data = funnel_data.sort_values(value_col, ascending=False)
        
        fig = px.funnel(funnel_data, x=value_col, y=stage_col, title=f"Funnel Chart: {value_col} per {stage_col}")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Keterangan Funnel Chart"):
            st.markdown("""
            **Funnel Chart** menampilkan proses bertahap dengan attrition.
            - **Kelebihan**: Visualisasi proses dan konversi yang jelas
            - **Kekurangan**: Hanya untuk data sequential
            - **Penggunaan**: Sales funnel, conversion analysis, process flow
            """)

def create_wordcloud(df, non_numeric_cols):
    st.subheader("Word Cloud")
    
    text_col = st.selectbox("Pilih kolom teks", non_numeric_cols, key="wordcloud_col")
    
    if text_col:
        text = ' '.join(df[text_col].astype(str).dropna())
        
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
            with st.expander("ℹ️ Keterangan Word Cloud"):
                st.markdown("""
                **Word Cloud** menampilkan frekuensi kata dengan ukuran font.
                - **Kelebihan**: Visualisasi teks yang intuitif
                - **Kekurangan**: Tidak menunjukkan hubungan antar kata
                - **Penggunaan**: Text analysis, sentiment visualization, topic identification
                """)
        else:
            st.warning("Tidak ada teks yang valid untuk ditampilkan")

def create_gantt_chart(df):
    st.subheader("Grafik Gantt")
    
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(date_cols) >= 2 and text_cols:
        start_col = st.selectbox("Pilih kolom tanggal mulai", date_cols, key="gantt_start")
        end_col = st.selectbox("Pilih kolom tanggal selesai", date_cols, key="gantt_end")
        task_col = st.selectbox("Pilih kolom task", text_cols, key="gantt_task")
        
        if start_col and end_col and task_col:
            gantt_data = []
            for idx, row in df.iterrows():
                if pd.notna(row[start_col]) and pd.notna(row[end_col]):
                    gantt_data.append(dict(
                        Task=row[task_col],
                        Start=row[start_col],
                        Finish=row[end_col],
                        Resource=f"Task {idx}"
                    ))
            
            if gantt_data:
                fig = ff.create_gantt(gantt_data, index_col='Resource', show_colorbar=True)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("ℹ️ Keterangan Gantt Chart"):
                    st.markdown("""
                    **Gantt Chart** menampilkan jadwal dan timeline tasks.
                    - **Kelebihan**: Visualisasi timeline yang jelas
                    - **Kekurangan**: Tidak menunjukkan dependencies antar tasks
                    - **Penggunaan**: Project management, timeline visualization, progress tracking
                    """)
            else:
                st.warning("Tidak ada data yang valid untuk Gantt chart")
    else:
        st.warning("Diperuhkan minimal 2 kolom tanggal dan 1 kolom teks untuk Gantt chart")

def create_map_chart(df):
    st.subheader("Grafik Peta")
    st.info("Fitur peta membutuhkan data latitude dan longitude atau nama negara/wilayah")
    
    possible_geo_cols = [col for col in df.columns if any(geo in col.lower() for geo in 
                        ['lat', 'lon', 'long', 'country', 'state', 'city', 'region'])]
    
    if possible_geo_cols:
        st.write("Kolom yang terdeteksi mungkin berisi data geografis:", possible_geo_cols)
    else:
        st.warning("Tidak terdeteksi kolom geografis. Contoh data peta tidak dapat ditampilkan.")

def create_flow_map(df):
    st.subheader("Grafik Peta Aliran")
    st.info("Fitur peta aliran membutuhkan data origin-destination dengan koordinat")
    
    st.write("Untuk flow map yang lengkap, diperlukan data dengan:")
    st.write("- Koordinat origin (lat, lon)")
    st.write("- Koordinat destination (lat, lon)")
    st.write("- Value untuk ketebalan flow")

def create_heatmap(df, numeric_cols):
    st.subheader("Heatmap")
    
    selected_cols = st.multiselect("Pilih kolom untuk heatmap", numeric_cols, 
                                 default=numeric_cols[:min(8, len(numeric_cols))], 
                                 key="heatmap_cols")
    
    if len(selected_cols) >= 2:
        clean_df = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
        corr_matrix = clean_df.corr()
        
        fig = px.imshow(corr_matrix, 
                      text_auto=True, 
                      aspect="auto", 
                      title="Heatmap Korelasi",
                      color_continuous_scale='RdBu_r')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Keterangan Heatmap"):
            st.markdown("""
            **Heatmap** menampilkan matriks korelasi antar variabel.
            - **Kelebihan**: Visualisasi hubungan yang komprehensif
            - **Kekurangan**: Hanya menunjukkan linear correlation
            - **Penggunaan**: Correlation analysis, pattern recognition, multivariate analysis
            """)
    else:
        st.warning("Pilih minimal 2 kolom untuk heatmap")

def create_multiple_line_chart(df, numeric_cols, non_numeric_cols):
    st.subheader("Multiple Line Chart")
    
    selected_lines = st.multiselect("Pilih kolom untuk garis", numeric_cols, 
                                  default=numeric_cols[:min(5, len(numeric_cols))],
                                  key="multi_line_select")
    
    x_col = st.selectbox("Pilih kolom untuk sumbu X", 
                       [df.index.name if df.index.name else "index"] + non_numeric_cols + numeric_cols, 
                       key="multi_x_col")
    
    if selected_lines and x_col:
        fig = go.Figure()
        
        display_df = df
        if len(df) > 1000:
            display_df = df.sample(1000, random_state=42)
            st.warning(f"Data ditampilkan 1000 sampel acak dari {len(df)} baris untuk performa")
        
        for col in selected_lines:
            clean_data = display_df[[x_col, col]].dropna()
            if len(clean_data) > 0:
                fig.add_trace(go.Scatter(
                    x=clean_data[x_col],
                    y=clean_data[col],
                    mode='lines',
                    name=col,
                    hovertemplate=f'<b>{col}</b><br>{x_col}: %{{x}}<br>Nilai: %{{y}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=f"Multiple Line Chart: {', '.join(selected_lines)}",
            xaxis_title=x_col,
            yaxis_title="Nilai",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ℹ️ Keterangan Multiple Line Chart"):
            st.markdown("""
            **Multiple Line Chart** menampilkan beberapa garis dalam satu chart.
            - **Kelebihan**: Perbandingan multiple trends secara bersamaan
            - **Kekurangan**: Bisa crowded jika terlalu banyak garis
            - **Penggunaan**: Comparative trend analysis, multi-metric monitoring
            """)

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
    
    if numeric_cols:
        st.subheader("📈 Statistik Numerik Lengkap")
        clean_df = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        desc_stats = clean_df.describe()
        st.dataframe(desc_stats, use_container_width=True)
    
    st.subheader("❓ Informasi Missing Values")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Kolom': missing_data.index,
        'Jumlah Missing': missing_data.values,
        'Persentase Missing': missing_percent.values
    })
    st.dataframe(missing_df, use_container_width=True)

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
    tab1, tab2, tab3 = st.tabs(["📊 Statistik", "📈 Visualisasi", "💾 Data"])
    
    with tab1:
        show_optimized_statistics(df)
    
    with tab2:
        st.header("🎨 Visualisasi Data")
        create_all_visualizations(df)
    
    with tab3:
        st.header("📋 Data Lengkap")
        st.dataframe(df, use_container_width=True)
        
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
    
    st.subheader("📊 Contoh Visualisasi")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_example = px.line(example_data, x='Tanggal', y='Close', title='Contoh Trend Harga Saham')
        st.plotly_chart(fig_example, use_container_width=True)
    
    with col2:
        # Contoh Pie Chart dengan slider
        sector_data = example_data.groupby('Sektor')['Volume'].sum().reset_index()
        sector_data['percentage'] = (sector_data['Volume'] / sector_data['Volume'].sum() * 100).round(2)
        
        fig_pie = px.pie(sector_data, names='Sektor', values='Volume', 
                        title='Volume Perdagangan per Sektor',
                        hover_data=['percentage'])
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")
st.markdown("Dashboard Statistik © 2025 - Dibuat oleh Dwi Bakti N Dev")
