#!/data/data/com.termux/files/usr/bin/bash

echo "[+] Updating system..."
pkg update && pkg upgrade -y

echo "[+] Installing dependencies..."
pkg install python git clang cmake libffi libffi-dev openssl openssl-dev -y

echo "[+] Upgrading pip tools..."
pip install --upgrade pip wheel setuptools

echo "[+] Installing Streamlit and core packages..."
pip install streamlit==1.49.1
pip install pandas==2.3.2
pip install numpy==2.2.2
pip install plotly==5.24.1
pip install matplotlib==3.9.2
pip install seaborn==0.13.2
pip install wordcloud==1.9.4
pip install streamlit-folium==0.23.1
pip install graphviz==0.20.3
pip install pillow==10.4.0
pip install pydeck==0.9.1  
pip install scipy==1.14.1
pip install scikit-learn==1.5.2   
pip install xgboost==2.1.1
pip install statsmodels==0.14.3   
pip install tensorflow-cpu==2.20.0
pip install keras==3.11.3
pip install networkx==3.4.2
pip install rich==14.2.0
pip install requests==2.32.3
pip install openpyxl==3.1.5
pip install xlrd==2.0.1
pip install numpy-stl==3.2.0
pip install trimesh==4.3.2
pip install streamlit-modal==0.1.2
pip install opencv-python-headless
pip install mlxtend==0.23.4
pip install streamlit --no-cache-dir

echo "[+] Cloning Launcher repo..."
if [ -d "Launcher" ]; then
    echo "[!] Folder Launcher already exists, skipping clone..."
else
    git clone https://github.com/DwiDevelopes/Launcher.git
fi

cd Launcher/build/lib/streamlit_launcher || {
    echo "[✗] Directory Launcher/build/lib/streamlit_launcher not found!"
    exit 1
}

echo "[+] Installing project requirements..."
pip install -r requirements.txt --no-cache-dir || {
    echo "[!] Some packages failed, installing missing common modules..."
    pip install pandas psutil flask streamlit numpy plotly matplotlib seaborn wordcloud streamlit-folium graphviz pillow pydeck scipy scikit-learn xgboost statsmodels tensorflow-cpu keras networkx rich requests openpyxl xlrd numpy-stl trimesh streamlit-modal opencv-python-headless mlxtend streamlit --no-cache-dir
}

echo "[+] Running Streamlit server..."
streamlit run dashboard.py --server.address=0.0.0.0 --server.port=8501
