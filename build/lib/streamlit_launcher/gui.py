import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import threading
import sys
import os
from PIL import Image, ImageTk


class StreamlitLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Streamlit Launcher")
        self.root.geometry("400x540")
        self.root.minsize(400, 540)
        self.root.resizable(False, False)
        try:
            icon_path = r"C:\Users\User\Downloads\streamlit_launcher\streamlit_launcher\gambar.ico"

            # Pastikan file ada
            if not os.path.exists(icon_path):
                raise FileNotFoundError("Icon file tidak ditemukan.")

            # Cek apakah file valid dengan PIL
            with Image.open(icon_path) as img_test:
                img_test.verify()  # validasi file ico

            # Set icon ke Tkinter window
            self.root.iconbitmap(icon_path)

        except Exception as e:
            print(f"[WARNING] Gagal load icon: {e}")

        self.process = None
        self.is_running = False

        # Init speech
        self.setup_speech()

        # Default language
        self.current_language = "English"
        self.languages = {
            "English": self.english_texts(),
            "Chinese": self.chinese_texts(),
            "Japanese": self.japanese_texts()
        }

        # Selalu pakai Female Voice
        self.voice_gender = "Female"
        self.set_voice_gender("Female")

        # Find dashboard.py
        self.script_path = self.find_dashboard()

        # Load image
        self.load_image()

        # Setup UI
        self.setup_ui()
        self.update_ui_text()

        # Welcome
        self.play_welcome_speech()

    # ============ SPEECH SETUP ==============
    def setup_speech(self):
        try:
            import pyttsx3
            self.client = pyttsx3.init()
            self.voices = self.client.getProperty("voices")
            self.speech_available = True
            
            # Setel langsung ke suara wanita
            self.set_voice_gender("Female")
        except Exception as e:
            print(f"Text-to-speech unavailable: {e}")
            self.client = None
            self.voices = []
            self.speech_available = False

    def set_voice_gender(self, gender):
        if not self.speech_available:
            return

        self.voice_gender = gender

        # Daftar kata kunci untuk female voice
        female_keywords = ["zira", "samantha", "hazel", "eva", "anna", "helen", "linda", "susan"]

        selected_voice = None
        for voice in self.voices:
            name = voice.name.lower()
            if gender == "Female":
                if any(key in name for key in female_keywords):
                    selected_voice = voice
                    break
            else:
                if "david" in name or "mark" in name or "george" in name:
                    selected_voice = voice
                    break

        # Jika ketemu, pakai itu
        if selected_voice:
            self.client.setProperty("voice", selected_voice.id)
        else:
            # fallback → pakai voice pertama saja
            self.client.setProperty("voice", self.voices[0].id)


    def speak_text(self, text):
        if self.speech_available and self.client:
            def speak():
                try:
                    # Pastikan selalu menggunakan suara wanita
                    self.set_voice_gender("Female")
                    self.client.say(text)
                    self.client.runAndWait()
                except Exception as e:
                    print("Speech error:", e)
            threading.Thread(target=speak, daemon=True).start()

    # ============ LANGUAGE PACKS ==============
    def english_texts(self):
        return {
            "title": "Streamlit Launcher",
            "port": "Port:",
            "start": "Start Server",
            "stop": "Stop Server",
            "log": "Log",
            "status_ready": "Ready",
            "language": "Language:",
            "welcome_speech": "Welcome to Streamlit Launcher",
            "start_speech": "Starting Streamlit Launcher",
            "stop_speech": "Stopping Streamlit Launcher"
        }

    def chinese_texts(self):
        return {
            "title": "Streamlit 启动器",
            "port": "端口:",
            "start": "启动服务器",
            "stop": "停止服务器",
            "log": "日志",
            "status_ready": "准备就绪",
            "language": "语言:",
            "welcome_speech": "欢迎使用 Streamlit 启动器",
            "start_speech": "正在启动服务器",
            "stop_speech": "正在停止服务器"
        }

    def japanese_texts(self):
        return {
            "title": "Streamlit ランチャー",
            "port": "ポート:",
            "start": "サーバー起動",
            "stop": "サーバー停止",
            "log": "ログ",
            "status_ready": "準備完了",
            "language": "言語:",
            "welcome_speech": "ストリームリットランチャーへようこそ",
            "start_speech": "サーバーを起動しています",
            "stop_speech": "サーバーを停止しています"
        }

    # ============ IMAGE ==============
    def load_image(self):
        self.photo = None
        if os.path.exists(r"C:\Users\User\Downloads\streamlit_launcher\streamlit_launcher\gambar.jpg"):
            try:
                img_path = r"C:\Users\User\Downloads\streamlit_launcher\streamlit_launcher\gambar.jpg"

                # buka gambar dengan context manager supaya file otomatis ditutup
                with Image.open(img_path) as img:
                    img_resized = img.resize((500, 250), Image.LANCZOS)
                    self.photo = ImageTk.PhotoImage(img_resized)

            except Exception as e:
                print("[WARNING] Gagal load gambar:", e)
        else:
            print("[WARNING] File gambar tidak ditemukan.")

    def find_dashboard(self):
        base_dir = r"C:\Users\User\Downloads\streamlit_launcher\streamlit_launcher"
        dashboard_path = os.path.join(base_dir, "dashboard.py")
        app_path = os.path.join(base_dir, "app.py")

        if os.path.exists(dashboard_path):
            return os.path.abspath(dashboard_path)
        elif os.path.exists(app_path):
            return os.path.abspath(app_path)
        return dashboard_path  # default kalau tidak ada


    # ============ UI ==============
    def setup_ui(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        for i in range(8):
            main_frame.rowconfigure(i, weight=0)
        main_frame.rowconfigure(6, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Title
        self.title_label = ttk.Label(main_frame, font=("Arial", 18, "bold"))
        self.title_label.grid(row=0, column=0, pady=5, sticky="n")

        # Language
        lang_frame = ttk.Frame(main_frame)
        lang_frame.grid(row=1, column=0, sticky="ew", pady=5)
        lang_frame.columnconfigure(1, weight=1)
        ttk.Label(lang_frame, text="Language:").grid(row=0, column=0, sticky="w")
        self.lang_var = tk.StringVar(value=self.current_language)
        lang_combo = ttk.Combobox(
            lang_frame, textvariable=self.lang_var,
            values=list(self.languages.keys()), state="readonly"
        )
        lang_combo.grid(row=0, column=1, sticky="ew")
        lang_combo.bind("<<ComboboxSelected>>", self.change_language)

        # Image
        if self.photo:
            ttk.Label(main_frame, image=self.photo).grid(row=2, column=0, pady=10, sticky="n")

        # Port
        port_frame = ttk.Frame(main_frame)
        port_frame.grid(row=3, column=0, sticky="ew", pady=5)
        port_frame.columnconfigure(1, weight=1)
        self.port_label = ttk.Label(port_frame, text="Port:")
        self.port_label.grid(row=0, column=0, sticky="w")
        self.port_var = tk.StringVar(value="8501")
        ttk.Entry(port_frame, textvariable=self.port_var, width=10).grid(row=0, column=1, sticky="w")

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, pady=10, sticky="ew")
        button_frame.columnconfigure((0, 1), weight=1)
        self.start_btn = ttk.Button(button_frame, command=self.start_server)
        self.start_btn.grid(row=0, column=0, padx=5, sticky="ew")
        self.stop_btn = ttk.Button(button_frame, command=self.stop_server, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5, sticky="ew")

        # Log
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=5)
        log_frame.grid(row=6, column=0, sticky="nsew", pady=5)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.config(state=tk.DISABLED)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor="w")
        status_bar.grid(row=7, column=0, sticky="ew")

    def update_ui_text(self):
        texts = self.languages[self.current_language]
        self.root.title(texts["title"])
        self.title_label.config(text=texts["title"])
        self.port_label.config(text=texts["port"])
        self.start_btn.config(text=texts["start"])
        self.stop_btn.config(text=texts["stop"])
        self.log_text.master.master.config(text=texts["log"])
        self.status_var.set(texts["status_ready"])

    # ============ EVENTS ==============
    def change_language(self, event):
        self.current_language = self.lang_var.get()
        self.update_ui_text()

    def play_welcome_speech(self):
        self.speak_text(self.languages[self.current_language]["welcome_speech"])

    def play_start_speech(self):
        self.speak_text(self.languages[self.current_language]["start_speech"])

    def play_stop_speech(self):
        self.speak_text(self.languages[self.current_language]["stop_speech"])

    # ============ SERVER CONTROL ==============
    def start_server(self):
        filename = self.script_path
        port = self.port_var.get()
        if not os.path.exists(filename):
            messagebox.showerror("Error", f"File '{filename}' not found")
            return
        self.play_start_speech()
        thread = threading.Thread(target=self.run_streamlit, args=(filename, port), daemon=True)
        thread.start()
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Running on port {port}")
        self.log_message(f"Starting server on port {port}...")

    def run_streamlit(self, filename, port):
        try:
            cmd = [sys.executable, "-m", "streamlit", "run", filename, "--server.port", port]
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True,
                encoding="utf-8", errors="replace"
            )
            for line in self.process.stdout:
                if not self.is_running:
                    break
                self.log_message(line.strip())
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log_message(f"Error: {str(e)}")

    def log_message(self, message):
        def update():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, update)

    def stop_server(self):
        self.play_stop_speech()
        if self.process:
            try:
                self.is_running = False
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                else:
                    import signal
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process = None
                self.log_message("Server stopped")
            except Exception as e:
                self.status_var.set(f"Error stopping: {e}")
                self.log_message(f"Error stopping: {e}")
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set(self.languages[self.current_language]["status_ready"])

    def on_closing(self):
        if self.is_running:
            if messagebox.askokcancel("Quit", "Server is running. Stop and quit?"):
                self.stop_server()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    
    # Set Windows style if available
    if sys.platform == "win32":
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
        
    app = StreamlitLauncher(root)
    root.mainloop()
    
def run_gui():
    root = tk.Tk()
    app = StreamlitLauncher(root)
    root.mainloop()