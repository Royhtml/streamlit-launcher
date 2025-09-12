# Streamlit Launcher

<img src = "treamlit.jpg" width = "100%" height= "100%">

[![PyPI version](https://badge.fury.io/py/launcher.svg)](https://badge.fury.io/py/launcher)
[![Downloads](https://pepy.tech/badge/launcher)](https://pepy.tech/project/launcher)
[![Downloads](https://pepy.tech/badge/launcher/month)](https://pepy.tech/project/launcher)
[![Downloads](https://pepy.tech/badge/launcher/week)](https://pepy.tech/project/launcher)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Statistik Penggunaan

| Metric | Value |
|--------|-------|
| Total Downloads | 15,000+ |
| Monthly Downloads | 2,500+ |
| Weekly Downloads | 600+ |
| Python Version Support | 3.7+ |
| Streamlit Version | 1.28+ |

## 📖 Overview

**Streamlit Launcher** adalah alat GUI yang sederhana dan powerful untuk menjalankan aplikasi Streamlit secara lokal. Tool ini dirancang khusus untuk Data Scientist dan Analis yang bekerja dengan Streamlit untuk membuat dashboard dan aplikasi data interaktif.

<img src = "Screenshot 2025-09-12 185806.png" width = "100%" height= "100%">
<img src = "Screenshot 2025-09-12 185806.png" width = "100%" height= "100%">
<img src = "Screenshot 2025-09-12 185806.png" width = "100%" height= "100%">

## 🎯 Untuk Data Scientist & Analis

### Keuntungan untuk Data Science:
- **Rapid Prototyping**: Memungkinkan pembuatan prototype dashboard dengan cepat
- **Visualisasi Interaktif**: Mendukung berbagai library visualisasi (Plotly, Matplotlib, Altair, dll)
- **Real-time Updates**: Perubahan kode langsung terlihat tanpa restart server
- **Deployment Mudah**: Dapat dengan mudah di-deploy ke cloud services

### Fitur Analisis Data yang Didukung:
- ✅ Eksplorasi data interaktif
- ✅ Visualisasi data real-time
- ✅ Machine Learning model deployment
- ✅ Dashboard monitoring
- ✅ Analisis statistik interaktif
- ✅ Reporting otomatis

## 🚀 Installation

### Prerequisites:
- Python 3.7 atau lebih tinggi
- pip (Python package manager)

### Installasi:

```bash
# Install menggunakan pip
pip install launcher

# Atau install dengan options tambahan
pip install launcher[dev]  # Untuk development
```

### Verifikasi Installasi:

```bash
# Cek versi yang terinstall
launcher --version

# Atau
python -m launcher --version
```

## 💻 Usage

### Cara Menjalankan:

```bash
# Jalankan launcher
launcher

# Atau dengan python module
python -m launcher

# Dengan options tertentu
launcher --port 8501 --host 0.0.0.0
```

### Options yang Tersedia:

```bash
launcher --help
# Output:
# Usage: launcher [OPTIONS]
# 
# Options:
#   --port INTEGER     Port number to run the app
#   --host TEXT        Host address to bind to
#   --debug BOOLEAN    Enable debug mode
#   --help             Show this message and exit
```

## 🖼️ Screenshot

<img src = "Screenshot 2025-09-12 185242.png" width="100%" height="100%">

*Tampilan GUI Streamlit Launcher yang user-friendly*

## 🔧 Advanced Configuration

### Konfigurasi Environment:

```bash
# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### File Konfigurasi:

Buat file `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false

[browser]
gatherUsageStats = false
```

## 📊 Contoh Aplikasi Data Science

### Contoh 1: EDA Dashboard

```python
# app_eda.py
import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

df = load_data()

# Sidebar filters
st.sidebar.header('Filters')
selected_columns = st.sidebar.multiselect('Select columns', df.columns.tolist())

# Main content
st.title('Exploratory Data Analysis')
st.dataframe(df[selected_columns] if selected_columns else df)

# Visualizations
if st.checkbox('Show correlation heatmap'):
    fig = px.imshow(df.corr())
    st.plotly_chart(fig)
```

### Contoh 2: Machine Learning Dashboard

```python
# app_ml.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.title('Machine Learning Model Trainer')

# Upload data
uploaded_file = st.file_uploader("Upload your dataset", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Model training interface
    target = st.selectbox('Select target variable', df.columns)
    features = st.multiselect('Select features', df.columns.drop(target))
    
    if st.button('Train Model'):
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        st.success(f'Model accuracy: {accuracy:.2f}')
```

## 🏗️ Project Structure

```
my_streamlit_project/
├── apps/
│   ├── app_eda.py
│   ├── app_ml.py
│   └── app_dashboard.py
├── data/
│   └── dataset.csv
├── requirements.txt
└── README.md
```

## 📋 Dependencies

Package ini membutuhkan:

- **streamlit** >= 1.28.0
- **python** >= 3.7.0
- **click** >= 8.0.0
- **typing-extensions** >= 4.0.0

## 🐛 Troubleshooting

### Common Issues:

1. **Port already in use:**
   ```bash
   launcher --port 8502
   ```

2. **Module not found:**
   ```bash
   pip install --upgrade launcher
   ```

3. **Permission issues:**
   ```bash
   pip install --user launcher
   ```

### Debug Mode:

```bash
# Enable debug mode
launcher --debug true

# Atau set environment variable
export STREAMLIT_DEBUG=true
```

## 🤝 Contributing

Kontribusi dipersilakan! Silakan:

1. Fork repository
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🔗 Links

- [PyPI Package](https://pypi.org/project/launcher/)
- [GitHub Repository](https://github.com/royhtml/streamlit-launcher)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Issue Tracker](https://github.com/royhtml/streamlit-launcher/issues)

## 📞 Support

Jika ada pertanyaan atau issues, silakan:

1. Check [documentation](https://github.com/royhtml/streamlit-launcher)
2. Search [existing issues](https://github.com/royhtml/streamlit-launcher/issues)
3. Create [new issue](https://github.com/royhtml/streamlit-launcher/issues/new)

---

**⭐ Jangan lupa memberikan bintang di GitHub jika tool ini membantu!**
