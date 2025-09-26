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
import io
import base64
import sys
import os

warnings.filterwarnings('ignore')

# --- Konfigurasi halaman ---
st.set_page_config(
    page_title="Advanced Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS kustom untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .code-editor {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        font-family: 'Courier New', monospace;
    }
    .jupyter-cell {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        margin: 10px 0;
        padding: 10px;
    }
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .data-table th, .data-table td {
        border: 1px solid #ddd;
        padding: 6px;
        text-align: left;
    }
    .data-table th {
        background-color: #f2f2f2;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Advanced Data Analysis Dashboard")

# Fungsi untuk menampilkan dataframe sebagai HTML (menggantikan st.dataframe)
def display_dataframe(df, num_rows=10):
    """Display dataframe as HTML table to avoid PyArrow issues"""
    st.markdown(f"**Preview Data ({num_rows} rows):**")
    st.markdown(df.head(num_rows).to_html(classes='data-table', escape=False, index=False), unsafe_allow_html=True)

# Fungsi untuk memproses file yang diupload
def process_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            # Gunakan engine python untuk menghindari issues
            df = pd.read_csv(uploaded_file, engine='python')
        elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format file tidak didukung. Harap unggah file CSV atau Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {str(e)}")
        return None

# Fungsi untuk menggabungkan dataset
def merge_datasets(datasets, merge_method='concat', merge_key=None):
    if not datasets:
        return None
    
    if merge_method == 'concat':
        return pd.concat(datasets, ignore_index=True)
    else:
        if not merge_key:
            common_columns = set(datasets[0].columns)
            for dataset in datasets[1:]:
                common_columns = common_columns.intersection(set(dataset.columns))
            if not common_columns:
                st.error("Tidak ada kolom yang sama untuk melakukan penggabungan.")
                return None
            merge_key = list(common_columns)[0]
        
        merged_df = datasets[0]
        for i in range(1, len(datasets)):
            try:
                merged_df = pd.merge(merged_df, datasets[i], how=merge_method, on=merge_key, suffixes=('', f'_{i}'))
            except Exception as e:
                st.error(f"Error saat menggabungkan dataset: {str(e)}")
                return None
        return merged_df

def wef_interface(df):
    st.header("🔬 WEF Interface (Write-Execute-Feedback)")
    
    # Inisialisasi session state
    if 'code_cells' not in st.session_state:
        st.session_state.code_cells = [
            {
                'code': '# Cell 1 - Mulai coding di sini\n'
                        'print("Shape dataset:", df.shape)\n'
                        'print("Columns:", df.columns.tolist())\n'
                        'df.head(3)',
                'output': ''
            }
        ]
    
    # Tombol aksi global
    col_global1, col_global2 = st.columns([1,1])
    with col_global1:
        if st.button("➕ Tambah Cell Baru"):
            st.session_state.code_cells.append({'code': '# Cell baru\n# Tulis kode Python di sini', 'output': ''})
    with col_global2:
        if st.button("♻️ Reset Semua Cell"):
            st.session_state.code_cells = [
                {'code': '# Cell 1 - Mulai coding di sini\nprint(df.head())', 'output': ''}
            ]
            st.rerun()

    # Container untuk cells
    for i, cell in enumerate(st.session_state.code_cells):
        with st.expander(f"📝 Cell {i+1}", expanded=True):
            col1, col2, col3 = st.columns([10, 1, 1])
            
            # Text area untuk kode
            with col1:
                new_code = st.text_area(
                    f"Kode Cell {i+1}",
                    value=cell['code'],
                    height=150,
                    key=f"code_{i}",
                    label_visibility="collapsed"
                )
                st.session_state.code_cells[i]['code'] = new_code
            
            # Tombol Run
            with col2:
                if st.button("▶️", key=f"run_{i}"):
                    try:
                        # Buat environment aman
                        local_env = {
                            'df': df, 
                            'pd': pd, 
                            'np': np, 
                            'plt': plt, 
                            'px': px, 
                            'go': go,
                            'st': st,
                            'print': print
                        }
                        
                        # Redirect output
                        old_stdout = sys.stdout
                        sys.stdout = output_catcher = io.StringIO()
                        
                        # Execute code
                        exec(new_code, local_env)
                        
                        # Capture output
                        output = output_catcher.getvalue()
                        sys.stdout = old_stdout
                        
                        st.session_state.code_cells[i]['output'] = output
                        
                    except Exception as e:
                        st.session_state.code_cells[i]['output'] = f"Error: {str(e)}"
            
            # Tombol Delete
            with col3:
                if st.button("🗑️", key=f"delete_{i}"):
                    if len(st.session_state.code_cells) > 1:
                        st.session_state.code_cells.pop(i)
                        st.rerun()
            
            # Tampilkan output
            if st.session_state.code_cells[i]['output']:
                st.markdown("**Output:**")
                st.code(st.session_state.code_cells[i]['output'], language='python')

    # Export kode
    if st.button("💾 Export Semua Kode ke file"):
        all_code = "\n\n".join([cell['code'] for cell in st.session_state.code_cells])
        st.download_button(
            label="⬇️ Download File Python",
            data=all_code,
            file_name="wef_export.py",
            mime="text/x-python"
        )

    # Contoh kode siap pakai
    st.subheader("📚 Contoh Kode Siap Pakai")
    contoh_opsi = st.selectbox("Pilih contoh:", [
        "Tampilkan 5 baris pertama",
        "Statistik deskriptif",
        "Plot histogram kolom numerik",
        "Plot scatter 2 kolom (Plotly)"
    ])

    if st.button("Tambahkan contoh ke Cell Baru"):
        if contoh_opsi == "Tampilkan 5 baris pertama":
            kode = "print(df.head())"
        elif contoh_opsi == "Statistik deskriptif":
            kode = "print(df.describe())"
        elif contoh_opsi == "Plot histogram kolom numerik":
            kode = "df.hist(figsize=(8,6))\nplt.show()"
        elif contoh_opsi == "Plot scatter 2 kolom (Plotly)":
            kode = "fig = px.scatter(df, x=df.columns[0], y=df.columns[1])\nst.plotly_chart(fig)"
        
        st.session_state.code_cells.append({'code': kode, 'output': ''})
        st.rerun()

# Fungsi analisis multivariat yang disederhanakan
def multivariate_analysis(df):
    st.header("🔗 Analisis Multivariat")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Diperlukan minimal 2 variabel numerik untuk analisis multivariat")
        return
    
    # Analisis Korelasi
    st.subheader("📈 Heatmap Korelasi")
    
    selected_cols = st.multiselect(
        "Pilih variabel untuk analisis korelasi:",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))],
        key="corr_select"
    )
    
    if len(selected_cols) >= 2:
        try:
            corr_matrix = df[selected_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Matriks Korelasi"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error membuat heatmap: {str(e)}")
    
    # Scatter Matrix
    if len(selected_cols) >= 2:
        st.subheader("🔄 Scatter Matrix")
        
        if st.button("Buat Scatter Matrix"):
            try:
                fig = px.scatter_matrix(df[selected_cols], title="Scatter Matrix")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error membuat scatter matrix: {str(e)}")

# Fungsi analisis time series
def time_series_analysis(df):
    st.header("⏰ Analisis Time Series")
    
    # Cari kolom tanggal
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(10))  # Test conversion
                date_cols.append(col)
            except:
                pass
    
    if not date_cols:
        st.warning("Tidak ditemukan kolom tanggal yang valid")
        return
    
    date_col = st.selectbox("Pilih kolom tanggal:", date_cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("Tidak ditemukan kolom numerik")
        return
    
    value_col = st.selectbox("Pilih variabel analisis:", numeric_cols)
    
    try:
        # Konversi tanggal
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.sort_values(date_col)
        
        # Plot time series
        fig = px.line(df_ts, x=date_col, y=value_col, title=f"Time Series: {value_col}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Moving average
        window = st.slider("Window Size untuk Moving Average:", 3, 30, 7)
        df_ts['MA'] = df_ts[value_col].rolling(window=window).mean()
        
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df_ts[date_col], y=df_ts[value_col], name='Actual'))
        fig_ma.add_trace(go.Scatter(x=df_ts[date_col], y=df_ts['MA'], name=f'MA ({window})'))
        fig_ma.update_layout(title=f'Moving Average - {value_col}')
        st.plotly_chart(fig_ma, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error dalam analisis time series: {str(e)}")

# Fungsi visualisasi dasar
def create_visualizations(df):
    # Menentukan kolom numerik dan non-numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Jika tidak ada kolom numerik, beri pesan error
    if not numeric_cols:
        st.error("Tidak ditemukan kolom numerik dalam dataset.")
        return
    
    # Default kolom untuk sumbu X dan Y
    default_x = non_numeric_cols[0] if non_numeric_cols else df.index.name if df.index.name else "Index"
    default_y = numeric_cols[0]
    
    # Sidebar untuk konfigurasi chart
    st.sidebar.header("Konfigurasi Visualisasi")
    
    # Pilihan jenis chart
    chart_type = st.sidebar.selectbox(
        "Pilih Jenis Chart",
        ["Line Chart", "Bar Chart", "Histogram", "Scatter Plot", "Pie Chart", "Box Plot", "Heatmap", "Candlestick", "Area Chart", "Histogram dengan Multiple Variables", "Peta"], 
        key="chart_type_select"
    )
    
    # Pilihan kolom berdasarkan jenis chart
    if chart_type in ["Line Chart", "Bar Chart", "Scatter Plot", "Area Chart"]:
        x_col = st.sidebar.selectbox("Pilih kolom untuk sumbu X", [default_x] + non_numeric_cols + numeric_cols, key="x_col_select")
        y_col = st.sidebar.selectbox("Pilih kolom untuk sumbu Y", numeric_cols, index=0, key="y_col_select")
        
        if chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        elif chart_type == "Scatter Plot":
            color_col = st.sidebar.selectbox("Pilih kolom untuk warna (opsional)", [None] + non_numeric_cols + numeric_cols, key="color_select")
            size_col = st.sidebar.selectbox("Pilih kolom untuk ukuran (opsional)", [None] + numeric_cols, key="size_select")
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, 
                            title=f"Scatter Plot: {y_col} vs {x_col}")
        elif chart_type == "Area Chart":
            fig = px.area(df, x=x_col, y=y_col, title=f"{y_col} over {x_col} (Area Chart)")
    
    elif chart_type == "Histogram":
        col = st.sidebar.selectbox("Pilih kolom", numeric_cols, key="hist_col_select")
        fig = px.histogram(df, x=col, title=f"Distribusi {col}")
    
    elif chart_type == "Histogram dengan Multiple Variables":
        selected_cols = st.sidebar.multiselect("Pilih kolom untuk histogram", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))], key="multi_hist_select")
        
        if len(selected_cols) > 0:
            # Buat histogram dengan multiple variables
            fig = go.Figure()
            
            for col in selected_cols:
                # Pastikan tidak ada nilai NaN atau infinite
                clean_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(clean_data) > 0:
                    fig.add_trace(go.Histogram(
                        x=clean_data,
                        name=col,
                        opacity=0.75
                    ))
            
            fig.update_layout(
                barmode='overlay',
                title_text='Distribusi Beberapa Variabel',
                xaxis_title_text='Nilai',
                yaxis_title_text='Frekuensi'
            )
            
            # Atur opacity untuk melihat overlap
            fig.update_traces(opacity=0.75)
        else:
            st.error("Pilih setidaknya satu kolom untuk histogram")
            return
    
    elif chart_type == "Pie Chart":
        # Untuk pie chart, kita butuh kolom kategori dan nilai
        cat_col = st.sidebar.selectbox("Pilih kolom kategori", non_numeric_cols, key="pie_cat_select")
        value_col = st.sidebar.selectbox("Pilih kolom nilai", numeric_cols, key="pie_val_select")
        
        # Opsi untuk menggabungkan kategori kecil
        combine_threshold = st.sidebar.slider(
            "Gabungkan kategori dengan persentase di bawah (%)", 
            min_value=0, max_value=20, value=5, step=1,
            key="combine_threshold_slider"
        )
        
        # Hitung total untuk setiap kategori
        pie_data = df.groupby(cat_col)[value_col].sum().reset_index()
        total = pie_data[value_col].sum()
        pie_data['percentage'] = (pie_data[value_col] / total) * 100
        
        # Gabungkan kategori kecil
        if combine_threshold > 0:
            other_data = pie_data[pie_data['percentage'] < combine_threshold]
            main_data = pie_data[pie_data['percentage'] >= combine_threshold]
            
            if len(other_data) > 0:
                other_sum = other_data[value_col].sum()
                other_percentage = other_data['percentage'].sum()
                
                # Buat dataframe baru dengan kategori "Lainnya"
                main_data = pd.concat([
                    main_data,
                    pd.DataFrame({cat_col: ['Lainnya'], value_col: [other_sum], 'percentage': [other_percentage]})
                ], ignore_index=True)
            
            pie_data = main_data
        
        fig = px.pie(pie_data, names=cat_col, values=value_col, title=f"Proporsi {value_col} oleh {cat_col}")
    
    elif chart_type == "Box Plot":
        col = st.sidebar.selectbox("Pilih kolom", numeric_cols, key="box_col_select")
        fig = px.box(df, y=col, title=f"Box Plot {col}")
    
    elif chart_type == "Heatmap":
        if len(numeric_cols) < 2:
            st.error("Heatmap memerlukan setidaknya 2 kolom numerik")
            return
        selected_cols = st.sidebar.multiselect("Pilih kolom untuk heatmap", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))], key="heatmap_cols_select")
        if len(selected_cols) < 2:
            st.error("Pilih setidaknya 2 kolom untuk heatmap")
            return
        
        # Pastikan tidak ada nilai NaN atau infinite
        clean_df = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_df) < 2:
            st.error("Tidak cukup data setelah pembersihan nilai NaN/infinite")
            return
            
        corr_matrix = clean_df.corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Heatmap Korelasi")
    
    elif chart_type == "Candlestick":
        # Untuk candlestick, kita butuh kolom OHLC
        st.sidebar.info("Candlestick memerlukan kolom Open, High, Low, Close")
        date_col = st.sidebar.selectbox("Pilih kolom tanggal", non_numeric_cols, key="candle_date_select")
        open_col = st.sidebar.selectbox("Pilih kolom Open", numeric_cols, key="candle_open_select")
        high_col = st.sidebar.selectbox("Pilih kolom High", numeric_cols, key="candle_high_select")
        low_col = st.sidebar.selectbox("Pilih kolom Low", numeric_cols, key="candle_low_select")
        close_col = st.sidebar.selectbox("Pilih kolom Close", numeric_cols, key="candle_close_select")
        
        # Pastikan kolom tanggal dalam format datetime
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df_sorted = df.sort_values(by=date_col).copy()
            
            fig = go.Figure(data=[go.Candlestick(
                x=df_sorted[date_col],
                open=df_sorted[open_col],
                high=df_sorted[high_col],
                low=df_sorted[low_col],
                close=df_sorted[close_col],
                name='Harga Saham'
            )])
            
            fig.update_layout(
                title='Chart Candlestick',
                yaxis_title='Harga',
                xaxis_title='Tanggal'
            )
        except Exception as e:
            st.error(f"Error membuat candlestick chart: {str(e)}")
            return
    
    elif chart_type == "Peta":
        st.sidebar.info("Peta memerlukan kolom dengan informasi geografis")
        
        # Cari kolom yang mungkin berisi data geografis
        geo_cols = [col for col in df.columns if any(term in col.lower() for term in ['country', 'state', 'city', 'region', 'latitude', 'longitude', 'lokasi', 'negara', 'provinsi', 'kota'])]
        
        if not geo_cols:
            st.error("Tidak ditemukan kolom yang teridentifikasi sebagai data geografis")
            return
        
        location_col = st.sidebar.selectbox("Pilih kolom lokasi", geo_cols, key="map_location_select")
        value_col = st.sidebar.selectbox("Pilih kolom nilai", numeric_cols, key="map_value_select")
        
        # Coba buat peta choropleth atau scatter map
        map_type = st.sidebar.selectbox("Pilih jenis peta", ["Choropleth", "Scatter Map"], key="map_type_select")
        
        if map_type == "Choropleth":
            # Untuk choropleth, kita perlu data geojson atau kode negara
            st.info("Choropleth memerlukan kode negara/region. Menggunakan data built-in Plotly.")
            
            # Coba identifikasi apakah data cocok dengan built-in geojson plotly
            try:
                fig = px.choropleth(df, locations=location_col, locationmode="country names",
                                   color=value_col, hover_name=location_col,
                                   title=f"Peta Choropleth {value_col} oleh {location_col}")
            except Exception as e:
                st.error(f"Error membuat peta choropleth: {str(e)}")
                st.info("Mencoba membuat scatter map sebagai alternatif")
                map_type = "Scatter Map"
        
        if map_type == "Scatter Map":
            # Untuk scatter map, kita butuh latitude dan longitude
            lat_cols = [col for col in df.columns if any(term in col.lower() for term in ['lat', 'latitude'])]
            lon_cols = [col for col in df.columns if any(term in col.lower() for term in ['lon', 'longitude'])]
            
            if not lat_cols or not lon_cols:
                st.error("Tidak ditemukan kolom latitude atau longitude. Scatter map memerlukan kedua kolom ini.")
                return
                
            lat_col = st.sidebar.selectbox("Pilih kolom latitude", lat_cols, key="map_lat_select")
            lon_col = st.sidebar.selectbox("Pilih kolom longitude", lon_cols, key="map_lon_select")
            
            if lat_col and lon_col:
                # Pastikan data numerik
                try:
                    df[lat_col] = pd.to_numeric(df[lat_col])
                    df[lon_col] = pd.to_numeric(df[lon_col])
                    
                    # Bersihkan data dari nilai NaN
                    clean_df = df.dropna(subset=[lat_col, lon_col, value_col])
                    
                    fig = px.scatter_mapbox(clean_df, lat=lat_col, lon=lon_col, color=value_col,
                                          size=value_col, hover_name=location_col,
                                          zoom=3, height=500,
                                          title=f"Peta Scatter {value_col} oleh {location_col}")
                    fig.update_layout(mapbox_style="open-street-map")
                except Exception as e:
                    st.error(f"Error membuat peta scatter: {str(e)}")
                    return
            else:
                st.error("Scatter map memerlukan kolom latitude dan longitude")
                return
    
    # Tampilkan chart
    if 'fig' in locals():
        st.plotly_chart(fig, use_container_width=True)
    
    # Tampilkan beberapa chart otomatis berdasarkan data
    st.subheader("Visualisasi Otomatis")
    
    # Pilih beberapa kolom numerik untuk ditampilkan
    if len(numeric_cols) > 0:
        selected_cols = st.multiselect(
            "Pilih kolom untuk divisualisasikan",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))],
            key="auto_viz_select"
        )
        
        if selected_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                # Line chart untuk kolom terpilih
                # Pastikan tidak ada nilai NaN atau infinite
                clean_df = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
                if len(clean_df) > 0:
                    fig_multi = px.line(clean_df, y=selected_cols, title="Trend Data")
                    st.plotly_chart(fig_multi, use_container_width=True)
                else:
                    st.error("Tidak ada data yang valid untuk ditampilkan")
            
            with col2:
                # Correlation heatmap
                if len(selected_cols) > 1:
                    # Pastikan tidak ada nilai NaN atau infinite
                    clean_df = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(clean_df) > 1:
                        corr_matrix = clean_df.corr()
                        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                                            title="Korelasi antar Variabel")
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.error("Tidak cukup data untuk menghitung korelasi setelah pembersihan")
                else:
                    # Jika hanya satu kolom, tampilkan box plot
                    clean_data = df[selected_cols[0]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(clean_data) > 0:
                        fig_box = px.box(y=clean_data, title=f"Distribusi {selected_cols[0]}")
                        st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        st.error("Tidak ada data yang valid untuk ditampilkan")

# Fungsi untuk menampilkan statistik deskriptif lengkap
def show_statistics(df):
    st.header("Statistik Deskriptif")
    
    # Tampilkan statistik dasar
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
    
    # Tampilkan preview data menggunakan HTML
    st.subheader("Preview Data")
    display_dataframe(df)
    
    # Tampilkan statistik deskriptif untuk kolom numerik
    if numeric_cols:
        st.subheader("Statistik Numerik Lengkap")
        
        # Bersihkan data dari nilai infinite
        clean_df = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Hitung berbagai statistik
        desc_stats = clean_df.describe()
        
        # Tambahkan statistik tambahan
        additional_stats = pd.DataFrame({
            'median': clean_df.median(),
            'variance': clean_df.var(),
            'skewness': clean_df.skew(),
            'kurtosis': clean_df.kurtosis(),
            'range': clean_df.max() - clean_df.min(),
            'Q1': clean_df.quantile(0.25),
            'Q3': clean_df.quantile(0.75),
            'IQR': clean_df.quantile(0.75) - clean_df.quantile(0.25)
        }).T
        
        # Gabungkan dengan statistik deskriptif standar
        full_stats = pd.concat([desc_stats, additional_stats])
        st.markdown(full_stats.to_html(classes='data-table', escape=False), unsafe_allow_html=True)
        
        # Uji normalitas untuk setiap kolom numerik
        st.subheader("Uji Normalitas (Shapiro-Wilk)")
        normality_results = []
        for col in numeric_cols:
            # Bersihkan data dari nilai infinite dan NaN
            data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) > 3 and len(data) < 5000:  # Shapiro-Wilk bekerja untuk 3 < n < 5000
                try:
                    stat, p_value = stats.shapiro(data)
                    normality_results.append({
                        'Kolom': col,
                        'Statistik': f"{stat:.4f}",
                        'p-value': f"{p_value:.4f}",
                        'Normal': "Ya" if p_value > 0.05 else "Tidak"
                    })
                except Exception as e:
                    normality_results.append({
                        'Kolom': col,
                        'Statistik': f"Error: {str(e)}",
                        'p-value': "Error",
                        'Normal': "Error"
                    })
            else:
                normality_results.append({
                    'Kolom': col,
                    'Statistik': "N/A",
                    'p-value': "N/A",
                    'Normal': "Sample size tidak sesuai"
                })
        
        normality_df = pd.DataFrame(normality_results)
        st.markdown(normality_df.to_html(classes='data-table', escape=False, index=False), unsafe_allow_html=True)
    
    # Tampilkan informasi tentang missing values
    st.subheader("Informasi Missing Values")
    missing_data = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Kolom': missing_data.index,
        'Jumlah Missing': missing_data.values,
        'Persentase Missing': (missing_data.values / len(df)) * 100
    })
    st.markdown(missing_df.to_html(classes='data-table', escape=False, index=False), unsafe_allow_html=True)
    
    # Tampilkan informasi tentang infinite values
    st.subheader("Informasi Infinite Values")
    if numeric_cols:
        inf_data = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            inf_data[col] = inf_count
        
        inf_df = pd.DataFrame({
            'Kolom': list(inf_data.keys()),
            'Jumlah Infinite': list(inf_data.values()),
            'Persentase Infinite': [f"{(v / len(df)) * 100:.2f}%" for v in inf_data.values()]
        })
        st.markdown(inf_df.to_html(classes='data-table', escape=False, index=False), unsafe_allow_html=True)
    
    # Tampilkan informasi tipe data
    st.subheader("Informasi Tipe Data")
    dtype_info = pd.DataFrame({
        'Kolom': df.columns,
        'Tipe Data': df.dtypes.values,
        'Nilai Unik': [df[col].nunique() for col in df.columns]
    })
    st.markdown(dtype_info.to_html(classes='data-table', escape=False, index=False), unsafe_allow_html=True)
    
    # Analisis trend saham jika ada kolom yang sesuai
    analyze_stock_trends(df)
    
    # Analisis data geografis jika ada
    analyze_geographic_data(df)
    
    # Analisis time series jika ada
    analyze_time_series_data(df)

# Fungsi untuk menganalisis trend saham naik/turun
def analyze_stock_trends(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if not date_cols:
        # Coba konversi kolom string ke datetime
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                    break
                except:
                    pass
    
    if date_cols and numeric_cols:
        st.subheader("Analisis Trend Saham")
        
        date_col = st.selectbox("Pilih kolom tanggal", date_cols, key="trend_date_select")
        price_col = st.selectbox("Pilih kolom harga", numeric_cols, key="trend_price_select")
        
        try:
            # Pastikan data diurutkan berdasarkan tanggal
            df_sorted = df.sort_values(by=date_col).copy()
            
            # Bersihkan data dari nilai infinite
            df_sorted[price_col] = df_sorted[price_col].replace([np.inf, -np.inf], np.nan)
            df_sorted = df_sorted.dropna(subset=[price_col])
            
            # Hitung perubahan harga
            df_sorted['Perubahan'] = df_sorted[price_col].pct_change() * 100
            df_sorted['Perubahan_Abs'] = df_sorted[price_col].diff()
            
            # Hitung statistik trend
            total_days = len(df_sorted)
            up_days = len(df_sorted[df_sorted['Perubahan_Abs'] > 0])
            down_days = len(df_sorted[df_sorted['Perubahan_Abs'] < 0])
            flat_days = len(df_sorted[df_sorted['Perubahan_Abs'] == 0])
            
            avg_change = df_sorted['Perubahan'].mean()
            max_gain = df_sorted['Perubahan'].max()
            max_loss = df_sorted['Perubahan'].min()
            
            # Tampilkan metrik trend
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                trend_icon = "📈" if avg_change > 0 else "📉"
                st.metric(f"Rata-rata Perubahan {trend_icon}", f"{avg_change:.2f}%")
            
            with col2:
                st.metric("Hari Naik", f"{up_days} ({up_days/total_days*100:.1f}%)")
            
            with col3:
                st.metric("Hari Turun", f"{down_days} ({down_days/total_days*100:.1f}%)")
            
            with col4:
                st.metric("Hari Datar", f"{flat_days} ({flat_days/total_days*100:.1f}%)")
            
            # Analisis volatilitas
            st.subheader("Analisis Volatilitas")
            
            # Hitung volatilitas (standar deviasi dari perubahan harga)
            volatility = df_sorted['Perubahan'].std()
            
            # Hitung moving averages
            df_sorted['MA_7'] = df_sorted[price_col].rolling(window=7).mean()
            df_sorted['MA_20'] = df_sorted[price_col].rolling(window=20).mean()
            
            # Hitung RSI (Relative Strength Index)
            delta = df_sorted[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_sorted['RSI'] = 100 - (100 / (1 + rs))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Volatilitas (Std Dev)", f"{volatility:.2f}%")
            
            with col2:
                # Tentukan apakah dalam kondisi overbought atau oversold berdasarkan RSI
                if 'RSI' in df_sorted.columns and not df_sorted['RSI'].isna().all():
                    latest_rsi = df_sorted['RSI'].iloc[-1]
                    rsi_status = "Overbought (>70)" if latest_rsi > 70 else "Oversold (<30)" if latest_rsi < 30 else "Netral"
                    st.metric("RSI Terakhir", f"{latest_rsi:.2f}", rsi_status)
            
            with col3:
                # Tentukan trend berdasarkan moving averages
                if 'MA_7' in df_sorted.columns and 'MA_20' in df_sorted.columns:
                    latest_ma7 = df_sorted['MA_7'].iloc[-1]
                    latest_ma20 = df_sorted['MA_20'].iloc[-1]
                    ma_trend = "Uptrend" if latest_ma7 > latest_ma20 else "Downtrend"
                    st.metric("Trend MA", ma_trend)
            
            # Tampilkan grafik harga dengan matplotlib
            st.subheader("Grafik Harga dengan Analisis Teknikal")
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            
            # Plot harga dan moving averages
            ax1.plot(df_sorted[date_col], df_sorted[price_col], label='Harga', color='blue', linewidth=1)
            if 'MA_7' in df_sorted.columns:
                ax1.plot(df_sorted[date_col], df_sorted['MA_7'], label='MA 7', color='orange', linewidth=1)
            if 'MA_20' in df_sorted.columns:
                ax1.plot(df_sorted[date_col], df_sorted['MA_20'], label='MA 20', color='red', linewidth=1)
            ax1.set_ylabel('Harga')
            ax1.set_title('Harga dan Moving Averages')
            ax1.grid(True)
            ax1.legend()
            
            # Format tanggal pada sumbu x
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot perubahan harga
            colors = ['green' if x >= 0 else 'red' for x in df_sorted['Perubahan_Abs']]
            ax2.bar(df_sorted[date_col], df_sorted['Perubahan_Abs'], color=colors, alpha=0.7)
            ax2.set_ylabel('Perubahan')
            ax2.set_title('Perubahan Harga Harian')
            ax2.grid(True)
            
            # Format tanggal pada sumbu x
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot RSI jika tersedia
            if 'RSI' in df_sorted.columns:
                ax3.plot(df_sorted[date_col], df_sorted['RSI'], label='RSI', color='purple', linewidth=1)
                ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
                ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
                ax3.set_ylabel('RSI')
                ax3.set_title('Relative Strength Index (RSI)')
                ax3.grid(True)
                ax3.legend()
                
                # Format tanggal pada sumbu x
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax3.xaxis.set_major_locator(mdates.MonthLocator())
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tampilkan sinyal trading sederhana
            st.subheader("Sinyal Trading Sederhana")
            
            if 'RSI' in df_sorted.columns and 'MA_7' in df_sorted.columns and 'MA_20' in df_sorted.columns:
                # Ambil data terbaru
                latest_data = df_sorted.iloc[-1]
                
                # Generate sinyal berdasarkan RSI dan MA
                signals = []
                
                if latest_data['RSI'] < 30 and latest_data['MA_7'] > latest_data['MA_20']:
                    signals.append(("BUY", "RSI oversold dan MA7 > MA20"))
                elif latest_data['RSI'] > 70 and latest_data['MA_7'] < latest_data['MA_20']:
                    signals.append(("SELL", "RSI overbought dan MA7 < MA20"))
                elif latest_data['RSI'] < 30:
                    signals.append(("POTENTIAL BUY", "RSI oversold"))
                elif latest_data['RSI'] > 70:
                    signals.append(("POTENTIAL SELL", "RSI overbought"))
                elif latest_data['MA_7'] > latest_data['MA_20']:
                    signals.append(("BULLISH", "MA7 > MA20 (Uptrend)"))
                elif latest_data['MA_7'] < latest_data['MA_20']:
                    signals.append(("BEARISH", "MA7 < MA20 (Downtrend)"))
                else:
                    signals.append(("NEUTRAL", "Tidak ada sinyal kuat"))
                
                # Tampilkan sinyal
                for signal, reason in signals:
                    if signal == "BUY":
                        st.success(f"🚀 {signal}: {reason}")
                    elif signal == "SELL":
                        st.error(f"🔻 {signal}: {reason}")
                    elif signal in ["POTENTIAL BUY", "BULLISH"]:
                        st.info(f"📈 {signal}: {reason}")
                    elif signal in ["POTENTIAL SELL", "BEARISH"]:
                        st.warning(f"📉 {signal}: {reason}")
                    else:
                        st.info(f"📊 {signal}: {reason}")
            
        except Exception as e:
            st.error(f"Error dalam analisis trend: {str(e)}")

# Fungsi untuk menganalisis data geografis
def analyze_geographic_data(df):
    # Identifikasi kolom yang mungkin berisi data geografis
    geo_cols = [col for col in df.columns if any(term in col.lower() for term in ['country', 'state', 'city', 'region', 'latitude', 'longitude', 'lokasi', 'negara', 'provinsi', 'kota'])]
    
    if geo_cols:
        st.subheader("Analisis Data Geografis")
        
        # Pilih kolom geografis
        location_col = st.selectbox("Pilih kolom lokasi", geo_cols, key="geo_location_select")
        
        # Hitung statistik untuk setiap lokasi
        location_stats = df[location_col].value_counts().reset_index()
        location_stats.columns = [location_col, 'Jumlah']
        
        # Tampilkan distribusi lokasi
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribusi Lokasi**")
            st.dataframe(location_stats.head(10))
        
        with col2:
            fig = px.bar(location_stats.head(10), x=location_col, y='Jumlah', 
                        title=f"10 Lokasi Terbanyak - {location_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Jika ada kolom numerik, tampilkan statistik per lokasi
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            value_col = st.selectbox("Pilih kolom nilai untuk analisis geografis", numeric_cols, key="geo_value_select")
            
            # Hitung statistik per lokasi
            geo_stats = df.groupby(location_col)[value_col].agg(['mean', 'sum', 'count', 'std']).reset_index()
            geo_stats.columns = [location_col, 'Rata-rata', 'Total', 'Jumlah', 'Std Dev']
            
            st.markdown(f"**Statistik {value_col} per {location_col}**")
            st.dataframe(geo_stats.sort_values('Total', ascending=False).head(10))
            
            # Coba buat peta jika ada data latitude/longitude
            lat_cols = [col for col in df.columns if any(term in col.lower() for term in ['lat', 'latitude'])]
            lon_cols = [col for col in df.columns if any(term in col.lower() for term in ['lon', 'longitude'])]
            
            if lat_cols and lon_cols:
                lat_col = st.selectbox("Pilih kolom latitude", lat_cols, key="geo_lat_select")
                lon_col = st.selectbox("Pilih kolom longitude", lon_cols, key="geo_lon_select")
                
                try:
                    # Pastikan data numerik
                    df[lat_col] = pd.to_numeric(df[lat_col])
                    df[lon_col] = pd.to_numeric(df[lon_col])
                    
                    # Bersihkan data dari nilai NaN
                    clean_df = df.dropna(subset=[lat_col, lon_col, value_col])
                    
                    # Buat peta scatter
                    fig = px.scatter_mapbox(clean_df, lat=lat_col, lon=lon_col, color=value_col,
                                          size=value_col, hover_name=location_col,
                                          zoom=3, height=500,
                                          title=f"Peta {value_col} oleh {location_col}")
                    fig.update_layout(mapbox_style="open-street-map")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error membuat peta: {str(e)}")

# Fungsi untuk menganalisis data time series
def analyze_time_series_data(df):
    # Identifikasi kolom tanggal
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if not date_cols:
        # Coba konversi kolom string ke datetime
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                    break
                except:
                    pass
    
    if date_cols:
        st.subheader("Analisis Time Series")
        
        date_col = st.selectbox("Pilih kolom tanggal", date_cols, key="ts_date_select")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            value_col = st.selectbox("Pilih kolom nilai untuk analisis time series", numeric_cols, key="ts_value_select")
            
            # Pastikan data diurutkan berdasarkan tanggal
            df_sorted = df.sort_values(by=date_col).copy()
            
            # Hitung statistik time series
            st.markdown("**Statistik Time Series**")
            
            # Resample data (harian, mingguan, bulanan)
            resample_period = st.selectbox("Pilih periode resample", ["Harian", "Mingguan", "Bulanan"], key="resample_select")
            
            if resample_period == "Harian":
                resampled_data = df_sorted.set_index(date_col).resample('D')[value_col].mean()
            elif resample_period == "Mingguan":
                resampled_data = df_sorted.set_index(date_col).resample('W')[value_col].mean()
            else:  # Bulanan
                resampled_data = df_sorted.set_index(date_col).resample('M')[value_col].mean()
            
            # Tampilkan grafik time series
            fig = px.line(resampled_data.reset_index(), x=date_col, y=value_col, 
                         title=f"Trend {value_col} ({resample_period})")
            st.plotly_chart(fig, use_container_width=True)
            
            # Hitung dan tampilkan statistik time series
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Perubahan dari periode pertama ke terakhir
                first_val = resampled_data.iloc[0]
                last_val = resampled_data.iloc[-1]
                change_pct = ((last_val - first_val) / first_val) * 100 if first_val != 0 else 0
                st.metric("Perubahan Total", f"{change_pct:.2f}%")
            
            with col2:
                # Volatilitas (std dev)
                volatility = resampled_data.std()
                st.metric("Volatilitas", f"{volatility:.2f}")
            
            with col3:
                # Trend (naik/turun berdasarkan slope)
                x = np.arange(len(resampled_data))
                y = resampled_data.values
                slope, _, _, _, _ = stats.linregress(x, y)
                trend = "Naik" if slope > 0 else "Turun" if slope < 0 else "Stabil"
                st.metric("Trend", trend)

# Fungsi untuk membuat file contoh dengan data saham
def create_sample_file():
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate harga saham dengan random walk
    np.random.seed(42)
    price_changes = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * (1 + price_changes).cumprod()
    
    # Generate volume
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
        'Negara': np.random.choice(['Indonesia', 'Malaysia', 'Singapura', 'Thailand', 'Vietnam'], len(dates)),
        'Latitude': np.random.uniform(-6.2, 5.6, len(dates)),
        'Longitude': np.random.uniform(95.0, 141.0, len(dates))
    })
    
    # Adjust OHLC untuk konsistensi
    for i in range(len(example_data)):
        example_data.loc[i, 'High'] = max(example_data.loc[i, 'Open'], example_data.loc[i, 'Close'], example_data.loc[i, 'High'])
        example_data.loc[i, 'Low'] = min(example_data.loc[i, 'Open'], example_data.loc[i, 'Close'], example_data.loc[i, 'Low'])
    
    return example_data

# Fungsi untuk analisis korelasi antar saham
def analyze_stock_correlation(df):
    st.header("Analisis Korelasi Antar Saham")
    
    # Identifikasi kolom yang mungkin berisi data harga saham
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if not date_cols:
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                    break
                except:
                    pass
    
    if date_cols and len(numeric_cols) > 1:
        date_col = st.selectbox("Pilih kolom tanggal untuk analisis korelasi", date_cols, key="corr_date_select")
        price_cols = st.multiselect("Pilih kolom harga untuk analisis korelasi", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))], key="corr_price_select")
        
        if len(price_cols) > 1:
            try:
                # Pastikan data diurutkan berdasarkan tanggal
                df_sorted = df.sort_values(by=date_col).copy()
                
                # Bersihkan data dari nilai infinite dan NaN
                clean_data = df_sorted[price_cols].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(clean_data) > 1:
                    # Hitung korelasi
                    correlation_matrix = clean_data.corr()
                    
                    # Tampilkan heatmap korelasi
                    fig = px.imshow(correlation_matrix, 
                                   text_auto=True, 
                                   aspect="auto", 
                                   title="Korelasi Antar Saham",
                                   color_continuous_scale='RdBu_r',
                                   zmin=-1, zmax=1)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tampilkan matriks korelasi sebagai tabel
                    st.subheader("Matriks Korelasi")
                    st.dataframe(correlation_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1).format("{:.2f}"))
                    
                    # Analisis pasangan dengan korelasi tertinggi dan terendah
                    st.subheader("Pasangan dengan Korelasi Ekstrem")
                    
                    # Dapatkan pasangan dengan korelasi tertinggi dan terendah
                    corr_pairs = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            corr_pairs.append({
                                'Pair': f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}",
                                'Correlation': correlation_matrix.iloc[i, j]
                            })
                    
                    corr_df = pd.DataFrame(corr_pairs)
                    top_corr = corr_df.nlargest(3, 'Correlation')
                    bottom_corr = corr_df.nsmallest(3, 'Correlation')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Korelasi Tertinggi**")
                        for _, row in top_corr.iterrows():
                            st.write(f"{row['Pair']}: {row['Correlation']:.3f}")
                    
                    with col2:
                        st.markdown("**Korelasi Terendah**")
                        for _, row in bottom_corr.iterrows():
                            st.write(f"{row['Pair']}: {row['Correlation']:.3f}")
                
                else:
                    st.error("Tidak cukup data untuk analisis korelasi setelah pembersihan")
            except Exception as e:
                st.error(f"Error dalam analisis korelasi: {str(e)}")
        else:
            st.error("Pilih setidaknya 2 kolom harga untuk analisis korelasi")
    else:
        st.error("Data tidak memiliki cukup kolom tanggal atau numerik untuk analisis korelasi")

# UI utama
st.markdown("Unggah file CSV atau Excel untuk melihat visualisasi dan statistik data.")

# Sidebar untuk upload file dan kontrol
st.sidebar.header("Kontrol Aplikasi")

# Tombol untuk membuat dan mengunduh file contoh
if st.sidebar.button("Buat File Contoh"):
    example_data = create_sample_file()
    csv = example_data.to_csv(index=False)
    st.sidebar.download_button(
        label="Unduh Contoh CSV",
        data=csv,
        file_name="contoh_data_saham.csv",
        mime="text/csv"
    )
        
else:
    # Tampilkan contoh data jika belum ada file yang diupload
    st.info("Silakan unggah file CSV atau Excel melalui sidebar di sebelah kiri.")
    
    # Buat contoh data
    st.subheader("Contoh Data")
    example_data = create_sample_file()
    display_dataframe(example_data)
    
    # Tampilkan contoh visualisasi
    st.subheader("Contoh Visualisasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_example = px.line(example_data, x='Tanggal', y='Close', title='Contoh Trend Harga Saham')
        st.plotly_chart(fig_example, use_container_width=True)
    
    with col2:
        # Pie chart contoh
        sector_data = example_data.groupby('Sektor')['Volume'].sum().reset_index()
        fig_pie = px.pie(sector_data, names='Sektor', values='Volume', title='Volume Perdagangan per Sektor')
        st.plotly_chart(fig_pie, use_container_width=True)

# Fungsi statistik dasar
def show_basic_stats(df):
    st.header("📈 Statistik Deskriptif")
    
    # Metrik cepat
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Total Variables", df.shape[1])
    with col3:
        st.metric("Numeric Variables", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.metric("Missing Values", f"{missing_pct:.1f}%")
    
    # Preview data
    display_dataframe(df, 10)
    
    # Statistik numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.subheader("Statistik Numerik")
        try:
            # Gunakan describe() dan format output
            stats_df = df[numeric_cols].describe()
            st.markdown(stats_df.to_html(classes='data-table', escape=False), unsafe_allow_html=True)
        except:
            # Fallback sederhana
            for col in numeric_cols:
                st.write(f"**{col}**: Mean={df[col].mean():.2f}, Std={df[col].std():.2f}")

# UI utama
st.sidebar.header("📁 Upload Data")
uploaded_files = st.sidebar.file_uploader(
    "Pilih file CSV atau Excel:",
    type=['csv', 'xlsx', 'xls'],
    accept_multiple_files=True
)

# Pengaturan sederhana
merge_method = "concat"
if uploaded_files and len(uploaded_files) > 1:
    merge_method = st.sidebar.selectbox("Metode Penggabungan:", ["concat", "inner"])

# Proses file upload
df = None
if uploaded_files:
    datasets = []
    for uploaded_file in uploaded_files:
        dataset = process_uploaded_file(uploaded_file)
        if dataset is not None:
            datasets.append(dataset)
            st.sidebar.success(f"✓ {uploaded_file.name} loaded")
    
    if datasets:
        if len(datasets) == 1:
            df = datasets[0]
        else:
            df = merge_datasets(datasets, merge_method)
        st.sidebar.success(f"✅ {len(datasets)} dataset loaded successfully!")

# Tampilkan interface berdasarkan ketersediaan data
if df is not None:
    # Tab interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🔬 WEF Interface", 
        "🔗 Multivariat", 
        "⏰ Time Series", 
        "📈 Visualisasi"
    ])
    
    with tab1:
        show_basic_stats(df)
    
    with tab2:
        wef_interface(df)
    
    with tab3:
        multivariate_analysis(df)
    
    with tab4:
        time_series_analysis(df)
    
    with tab5:
        create_visualizations(df)

else:
    # Tampilan welcome
    st.markdown("""
    # 🚀 Welcome to Advanced Data Analysis Dashboard
    
    ## Features:
    - **🔬 WEF Interface**: Jupyter-like coding environment
    - **🔗 Multivariate Analysis**: Correlation, scatter matrices
    - **⏰ Time Series Analysis**: Trends, moving averages
    - **📈 Visualization**: Interactive charts and plots
    
    ## 📁 Getting Started:
    1. Upload CSV or Excel files using the sidebar
    2. Explore different analysis tabs
    3. Use WEF interface for custom Python code
    
    ### Sample Code Examples:
    ```python
    # Basic data exploration
    print("Dataset shape:", df.shape)
    print("Column types:", df.dtypes.value_counts())
    
    # Basic statistics
    print(df.describe())
    
    # Data cleaning
    df_clean = df.dropna()
    print("After cleaning:", df_clean.shape)
    ```
    """)
    
    # Load sample data for demonstration
    if st.button("🔄 Load Sample Data for Demo"):
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50, freq='D'),
            'sales': np.random.normal(1000, 200, 50).cumsum(),
            'temperature': np.random.normal(25, 5, 50),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 50),
            'category': np.random.choice(['A', 'B', 'C'], 50)
        })
        
        # Simulate in session state
        st.session_state.sample_data = sample_data
        df = sample_data
        st.success("Sample data loaded! Switch between tabs to explore features.")
        st.markdown("**Sample Data Preview:**")
        display_dataframe(sample_data, 5)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Advanced Data Analysis Dashboard © 2025</p>
    <p>Create By Dwi Bakti N Dev</p>
</div>
""", unsafe_allow_html=True)
