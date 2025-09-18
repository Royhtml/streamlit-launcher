import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import requests
from urllib.parse import urlparse, parse_qs, urljoin
import socket
import ssl
import datetime
from bs4 import BeautifulSoup
import threading
import re
import os
from PIL import Image, ImageTk
import sys
import subprocess
import webbrowser
import json
import time
import xml.etree.ElementTree as ET
from http.client import HTTPConnection
import ipaddress
import whois 
import nmap
import socket
import dns

class AdvancedVulnerabilityScanner:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Vulnerability Scanner Pro")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        self.icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
        if os.path.exists(self.icon_path):
            self.root.iconbitmap(self.icon_path)
        
        # Scan configuration
        self.scan_config = {
            'timeout': 10,
            'max_pages': 20,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'threads': 5
        }
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 9))
        self.style.configure('TButton', font=('Arial', 9))
        self.style.configure('Treeview', font=('Arial', 9), rowheight=25)
        self.style.configure('TNotebook', background='#f0f0f0')
        self.style.configure('TNotebook.Tab', padding=[10, 5])
        
        # Main UI components
        self.create_widgets()
        self.create_menu()
        
        # Initialize nmap scanner
        self.nm = nmap.PortScanner() if self.check_nmap_installed() else None
        
    def check_nmap_installed(self):
        try:
            subprocess.run(["nmap", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except:
            return False
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Report", command=self.save_report)
        file_menu.add_command(label="Load Config", command=self.load_config)
        file_menu.add_command(label="Save Config", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Port Scanner", command=self.open_port_scanner)
        tools_menu.add_command(label="WHOIS Lookup", command=self.open_whois_tool)
        tools_menu.add_command(label="DNS Lookup", command=self.open_dns_tool)
        tools_menu.add_command(label="Ping Tool", command=self.open_ping_tool)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.open_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_widgets(self):
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(pady=5, fill=tk.X)
        
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
            if os.path.exists(icon_path):
                img = Image.open(icon_path)
                img = img.resize((32, 32), Image.LANCZOS)
                self.tk_icon = ImageTk.PhotoImage(img)
                icon_label = ttk.Label(header_frame, image=self.tk_icon)
                icon_label.pack(side=tk.LEFT, padx=(0, 8))
        except Exception:
            pass
        
        ttk.Label(header_frame, text="Advanced Vulnerability Scanner Pro", 
                 font=('Arial', 14, 'bold')).pack(side=tk.LEFT)
        
        # Input section
        input_frame = ttk.Frame(self.root)
        input_frame.pack(pady=5, fill=tk.X, padx=10)
        
        ttk.Label(input_frame, text="Target URL/IP:").grid(row=0, column=0, sticky=tk.W)
        self.url_entry = ttk.Entry(input_frame, width=50)
        self.url_entry.grid(row=0, column=1, padx=5)
        self.url_entry.insert(0, "https://")
        
        self.scan_btn = ttk.Button(input_frame, text="Scan", command=self.start_scan)
        self.scan_btn.grid(row=0, column=2, padx=5)
        
        # Quick tools buttons
        self.wsl_btn = ttk.Button(input_frame, text="WSL", command=self.open_wsl_linux)
        self.wsl_btn.grid(row=0, column=3, padx=5)
        
        self.terminal_btn = ttk.Button(input_frame, text="Terminal", command=self.open_terminal)
        self.terminal_btn.grid(row=0, column=4, padx=5)
        
        # Scan options notebook
        options_notebook = ttk.Notebook(self.root)
        options_notebook.pack(pady=5, fill=tk.X, padx=10)
        
        # Basic scan options
        basic_frame = ttk.Frame(options_notebook)
        self.sql_var = tk.BooleanVar(value=True)
        self.xss_var = tk.BooleanVar(value=True)
        self.headers_var = tk.BooleanVar(value=True)
        self.ssl_var = tk.BooleanVar(value=True)
        self.crawl_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(basic_frame, text="SQL Injection", variable=self.sql_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(basic_frame, text="XSS", variable=self.xss_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(basic_frame, text="Headers", variable=self.headers_var).grid(row=0, column=2, sticky=tk.W)
        ttk.Checkbutton(basic_frame, text="SSL/TLS", variable=self.ssl_var).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(basic_frame, text="Crawl Pages", variable=self.crawl_var).grid(row=1, column=1, sticky=tk.W)
        
        options_notebook.add(basic_frame, text="Basic")
        
        # Advanced scan options
        adv_frame = ttk.Frame(options_notebook)
        self.csrf_var = tk.BooleanVar(value=True)
        self.dir_trav_var = tk.BooleanVar(value=True)
        self.cmdi_var = tk.BooleanVar(value=True)
        self.file_exp_var = tk.BooleanVar(value=True)
        self.ssrf_var = tk.BooleanVar(value=False)
        self.xxe_var = tk.BooleanVar(value=False)
        self.idor_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(adv_frame, text="CSRF", variable=self.csrf_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(adv_frame, text="Directory Traversal", variable=self.dir_trav_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(adv_frame, text="Command Injection", variable=self.cmdi_var).grid(row=0, column=2, sticky=tk.W)
        ttk.Checkbutton(adv_frame, text="Exposed Files", variable=self.file_exp_var).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(adv_frame, text="SSRF", variable=self.ssrf_var).grid(row=1, column=1, sticky=tk.W)
        ttk.Checkbutton(adv_frame, text="XXE", variable=self.xxe_var).grid(row=1, column=2, sticky=tk.W)
        ttk.Checkbutton(adv_frame, text="IDOR", variable=self.idor_var).grid(row=2, column=0, sticky=tk.W)
        
        options_notebook.add(adv_frame, text="Advanced")
        
        # Network scan options
        net_frame = ttk.Frame(options_notebook)
        self.port_scan_var = tk.BooleanVar(value=False)
        self.os_detect_var = tk.BooleanVar(value=False)
        self.service_scan_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(net_frame, text="Port Scan", variable=self.port_scan_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(net_frame, text="OS Detection", variable=self.os_detect_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(net_frame, text="Service Scan", variable=self.service_scan_var).grid(row=0, column=2, sticky=tk.W)
        
        options_notebook.add(net_frame, text="Network")
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(pady=5, fill=tk.X, padx=10)
        self.status = ttk.Label(self.root, text="Ready to scan")
        self.status.pack()
        
        # Results display
        notebook = ttk.Notebook(self.root)
        notebook.pack(pady=5, fill=tk.BOTH, expand=True, padx=10)
        
        # Vulnerabilities tab
        vuln_frame = ttk.Frame(notebook)
        self.results_tree = ttk.Treeview(vuln_frame, columns=('severity', 'type', 'details'), show='headings')
        self.results_tree.heading('severity', text='Severity')
        self.results_tree.heading('type', text='Vulnerability')
        self.results_tree.heading('details', text='Details')
        self.results_tree.column('severity', width=80, anchor=tk.CENTER)
        self.results_tree.column('type', width=150)
        self.results_tree.column('details', width=500)
        
        scrollbar = ttk.Scrollbar(vuln_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        
        # Add context menu for vulnerabilities
        self.vuln_menu = tk.Menu(self.root, tearoff=0)
        self.vuln_menu.add_command(label="View Details", command=self.show_vuln_details)
        self.vuln_menu.add_command(label="Copy Details", command=self.copy_vuln_details)
        self.vuln_menu.add_command(label="Export to Report", command=self.export_vuln_to_report)
        self.results_tree.bind("<Button-3>", self.show_vuln_context_menu)
        
        notebook.add(vuln_frame, text='Vulnerabilities')
        
        # Details tab
        details_frame = ttk.Frame(notebook)
        self.details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, font=('Consolas', 9))
        self.details_text.pack(fill=tk.BOTH, expand=True)
        notebook.add(details_frame, text='Scan Details')
        
        # Exposed Files tab
        files_frame = ttk.Frame(notebook)
        self.files_tree = ttk.Treeview(files_frame, columns=('type', 'url', 'status', 'size'), show='headings')
        self.files_tree.heading('type', text='Type')
        self.files_tree.heading('url', text='URL')
        self.files_tree.heading('status', text='Status')
        self.files_tree.heading('size', text='Size')
        self.files_tree.column('type', width=100)
        self.files_tree.column('url', width=350)
        self.files_tree.column('status', width=80)
        self.files_tree.column('size', width=80)
        
        files_scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.files_tree.yview)
        self.files_tree.configure(yscroll=files_scrollbar.set)
        files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.files_tree.pack(fill=tk.BOTH, expand=True)
        
        # Add context menu for files tree
        self.files_menu = tk.Menu(self.root, tearoff=0)
        self.files_menu.add_command(label="Open in Browser", command=self.open_selected_file)
        self.files_menu.add_command(label="Copy URL", command=self.copy_file_url)
        self.files_menu.add_command(label="Download File", command=self.download_selected_file)
        self.files_tree.bind("<Button-3>", self.show_files_context_menu)
        
        notebook.add(files_frame, text='Exposed Files')
        
        # Network Info tab
        network_frame = ttk.Frame(notebook)
        self.network_tree = ttk.Treeview(network_frame, columns=('service', 'port', 'state', 'version'), show='headings')
        self.network_tree.heading('service', text='Service')
        self.network_tree.heading('port', text='Port')
        self.network_tree.heading('state', text='State')
        self.network_tree.heading('version', text='Version')
        self.network_tree.column('service', width=150)
        self.network_tree.column('port', width=80)
        self.network_tree.column('state', width=80)
        self.network_tree.column('version', width=200)
        
        network_scrollbar = ttk.Scrollbar(network_frame, orient=tk.VERTICAL, command=self.network_tree.yview)
        self.network_tree.configure(yscroll=network_scrollbar.set)
        network_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.network_tree.pack(fill=tk.BOTH, expand=True)
        
        notebook.add(network_frame, text='Network Info')
        
        # Configure tags for severity colors
        self.results_tree.tag_configure('critical', background='#ff9999', foreground='#800000')
        self.results_tree.tag_configure('high', background='#ffcccc')
        self.results_tree.tag_configure('medium', background='#fff3cd')
        self.results_tree.tag_configure('low', background='#d4edda')
        self.results_tree.tag_configure('info', background='#d1ecf1')
        
        # Configure tags for network info
        self.network_tree.tag_configure('open', background='#ffcccc')
        self.network_tree.tag_configure('filtered', background='#fff3cd')
        self.network_tree.tag_configure('closed', background='#d4edda')
    
    def show_vuln_context_menu(self, event):
        item = self.results_tree.identify_row(event.y)
        if item:
            self.results_tree.selection_set(item)
            self.vuln_menu.post(event.x_root, event.y_root)
    
    def show_vuln_details(self):
        selected = self.results_tree.selection()
        if selected:
            details = self.results_tree.item(selected[0])['values'][2]
            self.show_details_window("Vulnerability Details", details)
    
    def copy_vuln_details(self):
        selected = self.results_tree.selection()
        if selected:
            details = self.results_tree.item(selected[0])['values'][2]
            self.root.clipboard_clear()
            self.root.clipboard_append(details)
            messagebox.showinfo("Copied", "Vulnerability details copied to clipboard")
    
    def export_vuln_to_report(self):
        selected = self.results_tree.selection()
        if selected:
            item = self.results_tree.item(selected[0])
            severity, vuln_type, details = item['values']
            
            report = f"Vulnerability Report\n\n"
            report += f"Severity: {severity}\n"
            report += f"Type: {vuln_type}\n"
            report += f"Details:\n{details}\n"
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
                title="Save Report As"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(report)
                messagebox.showinfo("Saved", f"Report saved to {filename}")
    
    def show_details_window(self, title, content):
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("600x400")
        
        text = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=('Consolas', 9))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(tk.END, content)
        text.config(state=tk.DISABLED)
        
        btn_frame = ttk.Frame(window)
        btn_frame.pack(pady=5)
        
        ttk.Button(btn_frame, text="Copy", command=lambda: self.copy_to_clipboard(content)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", command=window.destroy).pack(side=tk.LEFT, padx=5)
    
    def copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copied", "Content copied to clipboard")
    
    def show_files_context_menu(self, event):
        item = self.files_tree.identify_row(event.y)
        if item:
            self.files_tree.selection_set(item)
            self.files_menu.post(event.x_root, event.y_root)
    
    def open_selected_file(self):
        selected = self.files_tree.selection()
        if selected:
            url = self.files_tree.item(selected[0])['values'][1]
            webbrowser.open(url)
    
    def download_selected_file(self):
        selected = self.files_tree.selection()
        if selected:
            url = self.files_tree.item(selected[0])['values'][1]
            try:
                response = requests.get(url, stream=True, timeout=10)
                if response.status_code == 200:
                    filename = filedialog.asksaveasfilename(
                        initialfile=os.path.basename(urlparse(url).path),
                        title="Save File As"
                    )
                    if filename:
                        with open(filename, 'wb') as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)
                        messagebox.showinfo("Download Complete", f"File saved to {filename}")
            except Exception as e:
                messagebox.showerror("Download Error", f"Failed to download file: {str(e)}")
    
    def copy_file_url(self):
        selected = self.files_tree.selection()
        if selected:
            url = self.files_tree.item(selected[0])['values'][1]
            self.root.clipboard_clear()
            self.root.clipboard_append(url)
            messagebox.showinfo("Copied", "URL copied to clipboard")
    
    def open_wsl_linux(self):
        """Open WSL (Windows Subsystem for Linux) terminal"""
        try:
            subprocess.Popen("wsl", creationflags=subprocess.CREATE_NEW_CONSOLE)
            messagebox.showinfo("WSL", "WSL terminal opened successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open WSL: {str(e)}")
    
    def open_terminal(self):
        """Open system terminal"""
        try:
            if sys.platform == "win32":
                subprocess.Popen("cmd", creationflags=subprocess.CREATE_NEW_CONSOLE)
            elif sys.platform == "linux":
                subprocess.Popen("x-terminal-emulator")
            elif sys.platform == "darwin":
                subprocess.Popen("open -a Terminal .")
            messagebox.showinfo("Terminal", "Terminal opened successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open terminal: {str(e)}")
    
    def save_report(self):
        """Save scan results to a file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("HTML Files", "*.html"), ("All Files", "*.*")],
                title="Save Report As"
            )
            
            if filename:
                if filename.endswith('.html'):
                    self.save_html_report(filename)
                else:
                    self.save_text_report(filename)
                
                messagebox.showinfo("Saved", f"Report saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {str(e)}")
    
    def save_text_report(self, filename):
        """Save report as plain text"""
        with open(filename, 'w') as f:
            # Write header
            f.write("="*80 + "\n")
            f.write("Advanced Vulnerability Scanner Report\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Target: {self.url_entry.get()}\n")
            f.write("="*80 + "\n\n")
            
            # Write vulnerabilities
            f.write("Vulnerabilities:\n")
            f.write("-"*80 + "\n")
            for item in self.results_tree.get_children():
                severity, vuln_type, details = self.results_tree.item(item)['values']
                f.write(f"[{severity}] {vuln_type}:\n{details}\n\n")
            
            # Write exposed files
            f.write("\nExposed Files:\n")
            f.write("-"*80 + "\n")
            for item in self.files_tree.get_children():
                file_type, url, status, size = self.files_tree.item(item)['values']
                f.write(f"{file_type}: {url} ({status}, {size})\n")
            
            # Write network info
            if self.network_tree.get_children():
                f.write("\nNetwork Information:\n")
                f.write("-"*80 + "\n")
                for item in self.network_tree.get_children():
                    service, port, state, version = self.network_tree.item(item)['values']
                    f.write(f"{service} on port {port} ({state}): {version}\n")
            
            # Write scan details
            f.write("\nScan Details:\n")
            f.write("-"*80 + "\n")
            f.write(self.details_text.get("1.0", tk.END))
    
    def save_html_report(self, filename):
        """Save report as HTML"""
        with open(filename, 'w') as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n")
            f.write("<title>Vulnerability Scan Report</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; line-height: 1.6; }\n")
            f.write("h1, h2 { color: #333; }\n")
            f.write(".critical { background-color: #ff9999; color: #800000; padding: 2px 5px; }\n")
            f.write(".high { background-color: #ffcccc; padding: 2px 5px; }\n")
            f.write(".medium { background-color: #fff3cd; padding: 2px 5px; }\n")
            f.write(".low { background-color: #d4edda; padding: 2px 5px; }\n")
            f.write(".info { background-color: #d1ecf1; padding: 2px 5px; }\n")
            f.write("pre { background-color: #f5f5f5; padding: 10px; border-radius: 3px; }\n")
            f.write("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
            f.write("th { background-color: #f2f2f2; }\n")
            f.write("</style>\n</head>\n<body>\n")
            
            # Write header
            f.write("<h1>Advanced Vulnerability Scanner Report</h1>\n")
            f.write(f"<p><strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            f.write(f"<p><strong>Target:</strong> {self.url_entry.get()}</p>\n")
            f.write("<hr>\n")
            
            # Write vulnerabilities
            f.write("<h2>Vulnerabilities</h2>\n")
            f.write("<table>\n")
            f.write("<tr><th>Severity</th><th>Type</th><th>Details</th></tr>\n")
            for item in self.results_tree.get_children():
                severity, vuln_type, details = self.results_tree.item(item)['values']
                f.write(f"<tr><td><span class='{severity.lower()}'>{severity}</span></td>")
                f.write(f"<td>{vuln_type}</td><td><pre>{details}</pre></td></tr>\n")
            f.write("</table>\n")
            
            # Write exposed files
            f.write("<h2>Exposed Files</h2>\n")
            f.write("<table>\n")
            f.write("<tr><th>Type</th><th>URL</th><th>Status</th><th>Size</th></tr>\n")
            for item in self.files_tree.get_children():
                file_type, url, status, size = self.files_tree.item(item)['values']
                f.write(f"<tr><td>{file_type}</td><td><a href='{url}' target='_blank'>{url}</a></td>")
                f.write(f"<td>{status}</td><td>{size}</td></tr>\n")
            f.write("</table>\n")
            
            # Write network info
            if self.network_tree.get_children():
                f.write("<h2>Network Information</h2>\n")
                f.write("<table>\n")
                f.write("<tr><th>Service</th><th>Port</th><th>State</th><th>Version</th></tr>\n")
                for item in self.network_tree.get_children():
                    service, port, state, version = self.network_tree.item(item)['values']
                    f.write(f"<tr><td>{service}</td><td>{port}</td><td>{state}</td><td>{version}</td></tr>\n")
                f.write("</table>\n")
            
            # Write scan details
            f.write("<h2>Scan Details</h2>\n")
            f.write(f"<pre>{self.details_text.get('1.0', tk.END)}</pre>\n")
            
            f.write("</body>\n</html>")
    
    def load_config(self):
        """Load scan configuration from file"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                title="Load Configuration"
            )
            
            if filename:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                # Update UI with loaded configuration
                self.url_entry.delete(0, tk.END)
                self.url_entry.insert(0, config.get('target_url', ''))
                
                # Update scan options
                self.sql_var.set(config.get('sql_injection', True))
                self.xss_var.set(config.get('xss', True))
                self.headers_var.set(config.get('headers', True))
                self.ssl_var.set(config.get('ssl', True))
                self.crawl_var.set(config.get('crawl', True))
                self.csrf_var.set(config.get('csrf', True))
                self.dir_trav_var.set(config.get('dir_traversal', True))
                self.cmdi_var.set(config.get('command_injection', True))
                self.file_exp_var.set(config.get('exposed_files', True))
                self.ssrf_var.set(config.get('ssrf', False))
                self.xxe_var.set(config.get('xxe', False))
                self.idor_var.set(config.get('idor', False))
                self.port_scan_var.set(config.get('port_scan', False))
                self.os_detect_var.set(config.get('os_detection', False))
                self.service_scan_var.set(config.get('service_scan', False))
                
                # Update scan configuration
                self.scan_config.update(config.get('scan_config', {}))
                
                messagebox.showinfo("Loaded", "Configuration loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                title="Save Configuration As"
            )
            
            if filename:
                config = {
                    'target_url': self.url_entry.get(),
                    'scan_config': self.scan_config,
                    'sql_injection': self.sql_var.get(),
                    'xss': self.xss_var.get(),
                    'headers': self.headers_var.get(),
                    'ssl': self.ssl_var.get(),
                    'crawl': self.crawl_var.get(),
                    'csrf': self.csrf_var.get(),
                    'dir_traversal': self.dir_trav_var.get(),
                    'command_injection': self.cmdi_var.get(),
                    'exposed_files': self.file_exp_var.get(),
                    'ssrf': self.ssrf_var.get(),
                    'xxe': self.xxe_var.get(),
                    'idor': self.idor_var.get(),
                    'port_scan': self.port_scan_var.get(),
                    'os_detection': self.os_detect_var.get(),
                    'service_scan': self.service_scan_var.get()
                }
                
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=4)
                
                messagebox.showinfo("Saved", f"Configuration saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def open_port_scanner(self):
        """Open port scanner tool"""
        port_window = tk.Toplevel(self.root)
        port_window.title("Port Scanner")
        port_window.geometry("500x400")
        
        ttk.Label(port_window, text="Target Host:").pack(pady=(10, 0))
        host_entry = ttk.Entry(port_window, width=30)
        host_entry.pack()
        host_entry.insert(0, self.url_entry.get())
        
        ttk.Label(port_window, text="Port Range (e.g., 1-1024):").pack(pady=(10, 0))
        port_entry = ttk.Entry(port_window, width=30)
        port_entry.pack()
        port_entry.insert(0, "1-1024")
        
        result_text = scrolledtext.ScrolledText(port_window, wrap=tk.WORD, height=15)
        result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        def scan_ports():
            host = host_entry.get().strip()
            port_range = port_entry.get().strip()
            
            if not host:
                messagebox.showerror("Error", "Please enter a target host")
                return
            
            try:
                # Validate port range
                if '-' in port_range:
                    start, end = map(int, port_range.split('-'))
                else:
                    start = end = int(port_range)
                
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Scanning {host} ports {start}-{end}...\n\n")
                port_window.update()
                
                open_ports = []
                
                for port in range(start, end + 1):
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(1)
                        result = sock.connect_ex((host, port))
                        if result == 0:
                            service = socket.getservbyport(port, 'tcp') if port <= 65535 else 'unknown'
                            open_ports.append((port, service))
                            result_text.insert(tk.END, f"Port {port} ({service}) is open\n")
                        sock.close()
                    except:
                        result_text.insert(tk.END, f"Port {port}: Error scanning\n")
                    
                    port_window.update()
                
                result_text.insert(tk.END, f"\nScan complete. Found {len(open_ports)} open ports.\n")
                
                if open_ports:
                    result_text.insert(tk.END, "\nOpen ports:\n")
                    for port, service in open_ports:
                        result_text.insert(tk.END, f"- {port}: {service}\n")
                
            except ValueError:
                messagebox.showerror("Error", "Invalid port range format")
            except Exception as e:
                messagebox.showerror("Error", f"Port scan failed: {str(e)}")
        
        ttk.Button(port_window, text="Scan Ports", command=scan_ports).pack(pady=5)
    
    def open_whois_tool(self):
        """Open WHOIS lookup tool"""
        whois_window = tk.Toplevel(self.root)
        whois_window.title("WHOIS Lookup")
        whois_window.geometry("600x400")
        
        ttk.Label(whois_window, text="Domain or IP:").pack(pady=(10, 0))
        domain_entry = ttk.Entry(whois_window, width=30)
        domain_entry.pack()
        domain_entry.insert(0, self.url_entry.get())
        
        result_text = scrolledtext.ScrolledText(whois_window, wrap=tk.WORD, height=20)
        result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        def perform_whois():
            domain = domain_entry.get().strip()
            if not domain:
                messagebox.showerror("Error", "Please enter a domain or IP")
                return
            
            try:
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Performing WHOIS lookup for {domain}...\n\n")
                whois_window.update()
                
                # Extract domain from URL if needed
                if domain.startswith(('http://', 'https://')):
                    domain = urlparse(domain).netloc
                
                # Perform WHOIS lookup
                w = whois.whois(domain)
                
                result_text.insert(tk.END, f"WHOIS Results for {domain}:\n\n")
                
                # Display basic info
                result_text.insert(tk.END, f"Domain Name: {w.domain_name}\n")
                result_text.insert(tk.END, f"Registrar: {w.registrar}\n")
                result_text.insert(tk.END, f"Creation Date: {w.creation_date}\n")
                result_text.insert(tk.END, f"Expiration Date: {w.expiration_date}\n")
                result_text.insert(tk.END, f"Name Servers: {', '.join(w.name_servers) if w.name_servers else 'N/A'}\n\n")
                
                # Display raw WHOIS data
                result_text.insert(tk.END, "Raw WHOIS Data:\n")
                result_text.insert(tk.END, "-"*50 + "\n")
                result_text.insert(tk.END, str(w.text) + "\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"WHOIS lookup failed: {str(e)}")
        
        ttk.Button(whois_window, text="Lookup", command=perform_whois).pack(pady=5)
    
    def open_dns_tool(self):
        """Open DNS lookup tool"""
        dns_window = tk.Toplevel(self.root)
        dns_window.title("DNS Lookup")
        dns_window.geometry("600x400")
        
        ttk.Label(dns_window, text="Domain:").pack(pady=(10, 0))
        domain_entry = ttk.Entry(dns_window, width=30)
        domain_entry.pack()
        domain_entry.insert(0, self.url_entry.get())
        
        result_text = scrolledtext.ScrolledText(dns_window, wrap=tk.WORD, height=20)
        result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        def perform_dns_lookup():
            domain = domain_entry.get().strip()
            if not domain:
                messagebox.showerror("Error", "Please enter a domain")
                return
            
            try:
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Performing DNS lookup for {domain}...\n\n")
                dns_window.update()
                
                # Extract domain from URL if needed
                if domain.startswith(('http://', 'https://')):
                    domain = urlparse(domain).netloc
                
                # Perform DNS lookups
                record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'CNAME', 'SOA']
                
                for record_type in record_types:
                    try:
                        result_text.insert(tk.END, f"{record_type} Records:\n")
                        answers = dns.resolver.resolve(domain, record_type)
                        for rdata in answers:
                            result_text.insert(tk.END, f"- {rdata}\n")
                        result_text.insert(tk.END, "\n")
                    except dns.resolver.NoAnswer:
                        result_text.insert(tk.END, f"No {record_type} records found\n\n")
                    except dns.resolver.NXDOMAIN:
                        result_text.insert(tk.END, f"Domain {domain} does not exist\n\n")
                        break
                    except Exception as e:
                        result_text.insert(tk.END, f"Error querying {record_type} records: {str(e)}\n\n")
                    
                    dns_window.update()
                
                result_text.insert(tk.END, "DNS lookup complete.\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"DNS lookup failed: {str(e)}")
        
        ttk.Button(dns_window, text="Lookup", command=perform_dns_lookup).pack(pady=5)
    
    def open_ping_tool(self):
        """Open ping tool"""
        ping_window = tk.Toplevel(self.root)
        ping_window.title("Ping Tool")
        ping_window.geometry("500x300")
        
        ttk.Label(ping_window, text="Host:").pack(pady=(10, 0))
        host_entry = ttk.Entry(ping_window, width=30)
        host_entry.pack()
        host_entry.insert(0, self.url_entry.get())
        
        ttk.Label(ping_window, text="Count (1-10):").pack(pady=(10, 0))
        count_entry = ttk.Entry(ping_window, width=10)
        count_entry.pack()
        count_entry.insert(0, "4")
        
        result_text = scrolledtext.ScrolledText(ping_window, wrap=tk.WORD, height=10)
        result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        def perform_ping():
            host = host_entry.get().strip()
            if not host:
                messagebox.showerror("Error", "Please enter a host")
                return
            
            try:
                count = int(count_entry.get())
                if count < 1 or count > 10:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Please enter a count between 1 and 10")
                return
            
            try:
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Pinging {host} {count} times...\n\n")
                ping_window.update()
                
                # Extract host from URL if needed
                if host.startswith(('http://', 'https://')):
                    host = urlparse(host).netloc
                
                # Platform-specific ping command
                if sys.platform == "win32":
                    cmd = ["ping", "-n", str(count), host]
                else:
                    cmd = ["ping", "-c", str(count), host]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                for line in process.stdout:
                    result_text.insert(tk.END, line)
                    ping_window.update()
                
                result_text.insert(tk.END, "\nPing complete.\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Ping failed: {str(e)}")
        
        ttk.Button(ping_window, text="Ping", command=perform_ping).pack(pady=5)
    
    def open_documentation(self):
        """Open documentation in browser"""
        try:
            webbrowser.open("https://github.com/yourusername/vulnerability-scanner/docs")
        except:
            messagebox.showerror("Error", "Could not open documentation")
    
    def show_about(self):
        """Show about dialog"""
        about_window = tk.Toplevel(self.root)
        about_window.title("About")
        about_window.geometry("400x300")
        
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
            if os.path.exists(icon_path):
                img = Image.open(icon_path)
                img = img.resize((64, 64), Image.LANCZOS)
                tk_icon = ImageTk.PhotoImage(img)
                icon_label = ttk.Label(about_window, image=tk_icon)
                icon_label.image = tk_icon  # Keep a reference
                icon_label.pack(pady=10)
        except Exception:
            pass
        
        ttk.Label(about_window, text="Advanced Vulnerability Scanner Pro", 
                 font=('Arial', 12, 'bold')).pack()
        
        ttk.Label(about_window, text="Version 2.0").pack()
        ttk.Label(about_window, text="\nA comprehensive security scanning tool").pack()
        ttk.Label(about_window, text="for identifying vulnerabilities in web applications.").pack()
        
        ttk.Label(about_window, text="\n© 2023 Security Tools Inc.").pack()
        
        btn_frame = ttk.Frame(about_window)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Close", command=about_window.destroy).pack()
    
    def start_scan(self):
        target = self.url_entry.get().strip()
        if not target:
            messagebox.showerror("Error", "Please enter a target URL or IP")
            return
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        for item in self.files_tree.get_children():
            self.files_tree.delete(item)
        for item in self.network_tree.get_children():
            self.network_tree.delete(item)
        self.details_text.delete(1.0, tk.END)
        
        self.scan_btn.config(state=tk.DISABLED)
        
        scan_thread = threading.Thread(target=self.run_scan, args=(target,), daemon=True)
        scan_thread.start()
    
    def run_scan(self, target):
        try:
            # Check if target is an IP address
            try:
                ipaddress.ip_address(target)
                is_ip = True
            except ValueError:
                is_ip = False
            
            if not is_ip and not target.startswith(('http://', 'https://')):
                target = f"http://{target}"
            
            parsed_url = urlparse(target) if not is_ip else None
            domain = parsed_url.netloc if not is_ip else target
            base_url = f"{parsed_url.scheme}://{domain}" if not is_ip else None
            
            # Calculate total steps for progress
            total_steps = sum([
                self.sql_var.get(),
                self.xss_var.get(),
                self.headers_var.get(),
                self.ssl_var.get(),
                self.csrf_var.get(),
                self.dir_trav_var.get(),
                self.cmdi_var.get(),
                self.crawl_var.get(),
                self.file_exp_var.get(),
                self.ssrf_var.get(),
                self.xxe_var.get(),
                self.idor_var.get(),
                self.port_scan_var.get(),
                self.os_detect_var.get(),
                self.service_scan_var.get()
            ])
            current_step = 0
            
            # Network scans (work for both IP and URL)
            if self.port_scan_var.get() and self.nm:
                self.update_status("Performing port scan...")
                self.port_scan(domain)
                current_step += 1
                self.update_progress(current_step, total_steps)
            
            if self.os_detect_var.get() and self.nm:
                self.update_status("Detecting OS...")
                self.os_detection(domain)
                current_step += 1
                self.update_progress(current_step, total_steps)
            
            if self.service_scan_var.get() and self.nm:
                self.update_status("Scanning services...")
                self.service_scan(domain)
                current_step += 1
                self.update_progress(current_step, total_steps)
            
            # Web application scans (only for URLs)
            if not is_ip:
                # SSL/TLS Check
                if self.ssl_var.get():
                    self.update_status("Checking SSL/TLS configuration...")
                    self.check_ssl(domain)
                    current_step += 1
                    self.update_progress(current_step, total_steps)
                
                # Headers Analysis
                if self.headers_var.get():
                    self.update_status("Analyzing HTTP headers...")
                    self.analyze_headers(target)
                    current_step += 1
                    self.update_progress(current_step, total_steps)
                
                # SQL Injection Test
                if self.sql_var.get():
                    self.update_status("Testing for SQL Injection...")
                    self.test_sql_injection(target)
                    current_step += 1
                    self.update_progress(current_step, total_steps)
                
                # XSS Test
                if self.xss_var.get():
                    self.update_status("Testing for XSS vulnerabilities...")
                    self.test_xss(target)
                    current_step += 1
                    self.update_progress(current_step, total_steps)
                
                # CSRF Test
                if self.csrf_var.get():
                    self.update_status("Checking for CSRF vulnerabilities...")
                    self.test_csrf(target)
                    current_step += 1
                    self.update_progress(current_step, total_steps)
                
                # Directory Traversal Test
                if self.dir_trav_var.get():
                    self.update_status("Testing for Directory Traversal...")
                    self.test_directory_traversal(target)
                    current_step += 1
                    self.update_progress(current_step, total_steps)
                
                # Command Injection Test
                if self.cmdi_var.get():
                    self.update_status("Testing for Command Injection...")
                    self.test_command_injection(target)
                    current_step += 1
                    self.update_progress(current_step, total_steps)
                
                # SSRF Test
                if self.ssrf_var.get():
                    self.update_status("Testing for SSRF vulnerabilities...")
                    self.test_ssrf(target)
                    current_step += 1
                    self.update_progress(current_step, total_steps)
                
                # XXE Test
                if self.xxe_var.get():
                    self.update_status("Testing for XXE vulnerabilities...")
                    self.test_xxe(target)
                    current_step += 1
                    self.update_progress(current_step, total_steps)
                
                # IDOR Test
                if self.idor_var.get():
                    self.update_status("Testing for IDOR vulnerabilities...")
                    self.test_idor(target)
                    current_step += 1
                    self.update_progress(current_step, total_steps)
                
                # Crawl Pages
                if self.crawl_var.get():
                    self.update_status("Crawling website for links...")
                    self.crawl_pages(base_url)
                    current_step += 1
                    self.update_progress(current_step, total_steps)
                
                # Check for exposed files
                if self.file_exp_var.get():
                    self.update_status("Checking for exposed files...")
                    self.check_exposed_files(base_url)
                    current_step += 1
                    self.update_progress(current_step, total_steps)
            
            self.update_status("Scan completed successfully!")
            messagebox.showinfo("Scan Complete", "Vulnerability scan completed!")
        
        except Exception as e:
            self.add_result("High", "Scan Error", f"Error during scan: {str(e)}")
            self.update_status(f"Error: {str(e)}")
        
        finally:
            self.scan_btn.config(state=tk.NORMAL)
    
    def port_scan(self, host):
        """Perform port scan using nmap"""
        try:
            if not self.nm:
                self.add_result("Medium", "Port Scan", "Nmap is not installed. Port scanning disabled.")
                return
            
            self.nm.scan(hosts=host, arguments='-F')  # Fast scan (top 100 ports)
            
            if host not in self.nm.all_hosts():
                self.add_result("Medium", "Port Scan", "Host not reachable for port scanning")
                return
            
            open_ports = []
            
            for proto in self.nm[host].all_protocols():
                ports = self.nm[host][proto].keys()
                for port in ports:
                    port_info = self.nm[host][proto][port]
                    state = port_info['state']
                    service = port_info['name']
                    version = port_info.get('version', 'unknown')
                    
                    if state == 'open':
                        open_ports.append((service, port, state, version))
                        self.network_tree.insert('', tk.END, 
                                               values=(service, port, state, version),
                                               tags=(state,))
            
            if open_ports:
                self.add_result("Medium", "Open Ports", 
                              f"Found {len(open_ports)} open ports on {host}")
            else:
                self.add_result("Low", "Port Scan", "No open ports found")
            
            self.details_text.insert(tk.END, f"\nPort Scan Results for {host}:\n")
            self.details_text.insert(tk.END, f"Scanned {len(self.nm[host].all_ports())} ports\n")
            self.details_text.insert(tk.END, f"Found {len(open_ports)} open ports\n\n")
        
        except Exception as e:
            self.add_result("Medium", "Port Scan Error", f"Port scan failed: {str(e)}")
    
    def os_detection(self, host):
        """Perform OS detection using nmap"""
        try:
            if not self.nm:
                self.add_result("Medium", "OS Detection", "Nmap is not installed. OS detection disabled.")
                return
            
            self.nm.scan(hosts=host, arguments='-O')  # OS detection scan
            
            if host not in self.nm.all_hosts():
                self.add_result("Medium", "OS Detection", "Host not reachable for OS detection")
                return
            
            os_info = self.nm[host].get('osmatch', [])
            
            if os_info:
                best_guess = os_info[0]
                os_name = best_guess['name']
                accuracy = best_guess['accuracy']
                
                self.add_result("Low", "OS Detection", 
                              f"Likely OS: {os_name} (Accuracy: {accuracy}%)")
                
                self.details_text.insert(tk.END, f"\nOS Detection Results for {host}:\n")
                self.details_text.insert(tk.END, f"Best guess: {os_name} ({accuracy}% accuracy)\n")
                
                if len(os_info) > 1:
                    self.details_text.insert(tk.END, "\nOther possible matches:\n")
                    for match in os_info[1:]:
                        self.details_text.insert(tk.END, f"- {match['name']} ({match['accuracy']}%)\n")
            else:
                self.add_result("Low", "OS Detection", "Could not determine operating system")
                self.details_text.insert(tk.END, f"\nOS Detection Results for {host}:\n")
                self.details_text.insert(tk.END, "Could not determine operating system\n")
        
        except Exception as e:
            self.add_result("Medium", "OS Detection Error", f"OS detection failed: {str(e)}")
    
    def service_scan(self, host):
        """Perform service version detection using nmap"""
        try:
            if not self.nm:
                self.add_result("Medium", "Service Scan", "Nmap is not installed. Service scan disabled.")
                return
            
            self.nm.scan(hosts=host, arguments='-sV')  # Service version scan
            
            if host not in self.nm.all_hosts():
                self.add_result("Medium", "Service Scan", "Host not reachable for service scan")
                return
            
            service_info = []
            
            for proto in self.nm[host].all_protocols():
                ports = self.nm[host][proto].keys()
                for port in ports:
                    port_data = self.nm[host][proto][port]
                    if port_data['state'] == 'open':
                        service_info.append((
                            port_data['name'],
                            port,
                            port_data['state'],
                            port_data.get('product', '') + ' ' + port_data.get('version', '')
                        ))
            
            if service_info:
                self.add_result("Low", "Service Scan", 
                              f"Found {len(service_info)} services on {host}")
                
                self.details_text.insert(tk.END, f"\nService Scan Results for {host}:\n")
                for service in service_info:
                    self.details_text.insert(tk.END, 
                                           f"Port {service[1]}: {service[0]} - {service[3]}\n")
            else:
                self.add_result("Low", "Service Scan", "No services detected")
                self.details_text.insert(tk.END, f"\nService Scan Results for {host}:\n")
                self.details_text.insert(tk.END, "No services detected\n")
        
        except Exception as e:
            self.add_result("Medium", "Service Scan Error", f"Service scan failed: {str(e)}")
    
    def check_exposed_files(self, base_url):
        """Check for exposed files on hosting platforms like Vercel, Netlify, etc."""
        try:
            parsed = urlparse(base_url)
            domain = parsed.netloc
            scheme = parsed.scheme
            
            # Common exposed files and directories
            common_files = [
                # Configuration files
                "_config.yml", "config.yml", "config.json", "package.json", 
                "composer.json", "package-lock.json", "yarn.lock",
                "dockerfile", "docker-compose.yml", ".env", ".env.example",
                ".gitignore", ".htaccess", "robots.txt", "sitemap.xml",
                
                # Platform specific files
                "vercel.json", "netlify.toml", "now.json", "firebase.json",
                "_redirects", "_headers",
                
                # Source code files
                "index.php", "index.html", "main.js", "app.js", "server.js",
                "style.css", "main.css", "app.css", "README.md", "LICENSE",
                
                # Backup files
                "backup.zip", "backup.tar.gz", "backup.sql", "dump.sql",
                "database.sql", "backup.rar", "backup.db",
                
                # Admin interfaces
                "admin.php", "admin.html", "wp-admin", "administrator",
                "login.php", "login.html", "wp-login.php",
                
                # API endpoints
                "api/v1", "graphql", "graphiql", "api.json", "swagger.json",
                "openapi.json", "api.php", "api.js",
                
                # Log files
                "logs", "error.log", "access.log", "debug.log"
            ]
            
            # Platform-specific file patterns
            platform_patterns = {
                "vercel": [
                    "/_next/static/chunks/pages/", 
                    "/_next/static/development/",
                    "/_next/static/css/",
                    "/api/",
                    "/public/"
                ],
                "netlify": [
                    "/.netlify/functions/",
                    "/public/",
                    "/static/",
                    "/dist/"
                ],
                "github": [
                    "/.github/workflows/",
                    "/.github/",
                    "/actions/"
                ],
                "firebase": [
                    "/__/firebase/",
                    "/__/auth/",
                    "/__/database/"
                ]
            }
            
            # Check if the domain matches known hosting platforms
            platform = None
            if "vercel.app" in domain:
                platform = "vercel"
            elif "netlify.app" in domain:
                platform = "netlify"
            elif "github.io" in domain:
                platform = "github"
            elif "firebaseapp.com" in domain or "web.app" in domain:
                platform = "firebase"
            
            found_files = []
            
            # Check common files
            for file in common_files:
                test_url = f"{scheme}://{domain}/{file}"
                try:
                    response = requests.head(test_url, timeout=5, allow_redirects=False)
                    if response.status_code == 200:
                        size = response.headers.get('content-length', 'unknown')
                        found_files.append(("file", test_url, "200 OK", size))
                        self.files_tree.insert('', tk.END, 
                                             values=("File", test_url, "200 OK", size))
                except:
                    continue
            
            # Check platform-specific patterns if platform is detected
            if platform:
                for pattern in platform_patterns.get(platform, []):
                    test_url = f"{scheme}://{domain}{pattern}"
                    try:
                        response = requests.head(test_url, timeout=5, allow_redirects=False)
                        if response.status_code == 200:
                            size = response.headers.get('content-length', 'unknown')
                            found_files.append(("directory", test_url, "200 OK", size))
                            self.files_tree.insert('', tk.END, 
                                                 values=("Directory", test_url, "200 OK", size))
                    except:
                        continue
            
            # Special check for .git directory
            test_url = f"{scheme}://{domain}/.git/"
            try:
                response = requests.head(test_url, timeout=5, allow_redirects=False)
                if response.status_code in (200, 403):
                    size = response.headers.get('content-length', 'unknown')
                    found_files.append(("directory", test_url, f"{response.status_code}", size))
                    self.files_tree.insert('', tk.END, 
                                         values=("Git", test_url, f"{response.status_code}", size))
            except:
                pass
            
            # Check for exposed source code
            self.check_exposed_source_code(base_url)
            
            if found_files:
                self.add_result("Medium", "Exposed Files", f"Found {len(found_files)} exposed files/directories")
            else:
                self.add_result("Low", "Exposed Files", "No obvious exposed files found")
            
            self.details_text.insert(tk.END, f"\nExposed Files Check:\nFound {len(found_files)} files/directories\n")
        
        except Exception as e:
            self.add_result("Medium", "Exposed Files Error", f"Exposed files check failed: {str(e)}")
    
    def check_exposed_source_code(self, base_url):
        """Check for exposed source code files on platforms like Vercel, Netlify"""
        try:
            parsed = urlparse(base_url)
            domain = parsed.netloc
            scheme = parsed.scheme
            
            # Common source code patterns for Vercel/Next.js
            vercel_patterns = [
                "/_next/static/chunks/pages/_app.js",
                "/_next/static/chunks/main.js",
                "/_next/static/chunks/webpack.js",
                "/_next/static/css/styles.chunk.css"
            ]
            
            # Common source code patterns for Netlify
            netlify_patterns = [
                "/static/js/main.chunk.js",
                "/static/js/runtime-main.js",
                "/static/css/main.chunk.css"
            ]
            
            # Check if the domain matches known hosting platforms
            if "vercel.app" in domain:
                for pattern in vercel_patterns:
                    test_url = f"{scheme}://{domain}{pattern}"
                    try:
                        response = requests.head(test_url, timeout=5, allow_redirects=False)
                        if response.status_code == 200:
                            size = response.headers.get('content-length', 'unknown')
                            self.files_tree.insert('', tk.END, 
                                                 values=("Source", test_url, "200 OK", size))
                    except:
                        continue
            
            elif "netlify.app" in domain:
                for pattern in netlify_patterns:
                    test_url = f"{scheme}://{domain}{pattern}"
                    try:
                        response = requests.head(test_url, timeout=5, allow_redirects=False)
                        if response.status_code == 200:
                            size = response.headers.get('content-length', 'unknown')
                            self.files_tree.insert('', tk.END, 
                                                 values=("Source", test_url, "200 OK", size))
                    except:
                        continue
            
            # Special check for source map files
            source_map_patterns = [
                "/static/js/main.js.map",
                "/static/js/bundle.js.map",
                "/static/js/vendor.js.map",
                "/app.js.map",
                "/main.js.map"
            ]
            
            for pattern in source_map_patterns:
                test_url = f"{scheme}://{domain}{pattern}"
                try:
                    response = requests.head(test_url, timeout=5, allow_redirects=False)
                    if response.status_code == 200:
                        size = response.headers.get('content-length', 'unknown')
                        self.files_tree.insert('', tk.END, 
                                             values=("Source Map", test_url, "200 OK", size))
                except:
                    continue
            
        except Exception as e:
            self.add_result("Medium", "Source Code Check Error", f"Source code check failed: {str(e)}")
    
    def update_status(self, message):
        self.status.config(text=message)
        self.root.update()
    
    def update_progress(self, current, total):
        progress = (current / total) * 100
        self.progress['value'] = progress
        self.root.update()
    
    def add_result(self, severity, vuln_type, details):
        tag = severity.lower()
        self.results_tree.insert('', tk.END, values=(severity, vuln_type, details), tags=(tag,))
        self.details_text.insert(tk.END, f"[{severity}] {vuln_type}: {details}\n\n")
        self.details_text.see(tk.END)
        self.root.update()
    
    def check_ssl(self, domain):
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443)) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
            
            # Check certificate expiration
            expiry_date = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
            days_remaining = (expiry_date - datetime.datetime.now()).days
            
            details = "SSL Certificate:\nIssuer: {}\n".format(cert['issuer'][1][0][1])
            details += "Subject: {}\n".format(cert['subject'][1][0][1])
            details += "Serial Number: {}\n".format(cert['serialNumber'])
            details += "Valid From: {}\n".format(cert['notBefore'])
            details += "Valid Until: {} ({} days remaining)\n".format(cert['notAfter'], days_remaining)
            
            # Check certificate algorithms
            details += "\nCertificate Algorithms:\n"
            details += "Signature Algorithm: {}\n".format(cert['signatureAlgorithm'])
            
            # Check for weak protocols
            weak_protocols = self.detect_weak_ssl_protocols(domain)
            if weak_protocols:
                details += "\nWeak Protocols: {}\n".format(', '.join(weak_protocols))
                self.add_result("High", "Weak SSL Protocols", f"Server supports: {', '.join(weak_protocols)}")
            
            # Check for weak ciphers
            weak_ciphers = self.detect_weak_ciphers(domain)
            if weak_ciphers:
                details += "\nWeak Ciphers:\n"
                for cipher in weak_ciphers:
                    details += f"- {cipher}\n"
                self.add_result("High", "Weak SSL Ciphers", f"Server supports weak ciphers: {', '.join(weak_ciphers[:3])}...")
            
            if days_remaining < 30:
                self.add_result("High", "SSL Expiry", f"Certificate expires in {days_remaining} days")
            elif days_remaining < 90:
                self.add_result("Medium", "SSL Expiry", f"Certificate expires in {days_remaining} days")
            
            self.details_text.insert(tk.END, details + "\n")
        
        except Exception as e:
            self.add_result("High", "SSL Error", f"SSL verification failed: {str(e)}")
    
    def detect_weak_ssl_protocols(self, domain):
        weak_protocols = []
        protocols = {
            'SSLv2': ssl.PROTOCOL_SSLv2,
            'SSLv3': ssl.PROTOCOL_SSLv3,
            'TLSv1': ssl.PROTOCOL_TLSv1,
            'TLSv1.1': ssl.PROTOCOL_TLSv1_1
        }
        
        for name, proto in protocols.items():
            try:
                context = ssl.SSLContext(proto)
                with socket.create_connection((domain, 443)) as sock:
                    with context.wrap_socket(sock, server_hostname=domain):
                        weak_protocols.append(name)
            except:
                continue
        
        return weak_protocols
    
    def detect_weak_ciphers(self, domain):
        weak_ciphers = [
            'DES', '3DES', 'RC4', 'RC2', 'IDEA', 'SEED',
            'MD5', 'SHA1', 'NULL', 'ANON', 'ADH', 'EXP',
            'CBC', 'CAMELLIA', 'PSK', 'SRP'
        ]
        
        detected = []
        
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443)) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cipher = ssock.cipher()
                    if cipher:
                        for weak in weak_ciphers:
                            if weak in cipher[0]:
                                detected.append(cipher[0])
                                break
        except:
            pass
        
        return detected
    
    def analyze_headers(self, url):
        try:
            response = requests.get(url, timeout=10, allow_redirects=True)
            headers = response.headers
            
            details = "Security Headers Analysis:\n"
            missing_headers = []
            security_headers = {
                'X-XSS-Protection': '1; mode=block',
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                'Content-Security-Policy': '',
                'Strict-Transport-Security': '',
                'Referrer-Policy': 'no-referrer',
                'Feature-Policy': '',
                'Permissions-Policy': ''
            }
            
            for header, expected in security_headers.items():
                if header not in headers:
                    missing_headers.append(header)
                elif expected and isinstance(expected, list) and headers[header] not in expected:
                    self.add_result("Medium", f"Misconfigured {header}", 
                                  f"Expected one of {expected}, got {headers[header]}")
                elif expected and isinstance(expected, str) and headers[header] != expected:
                    self.add_result("Medium", f"Misconfigured {header}", 
                                  f"Expected {expected}, got {headers[header]}")
            
            if missing_headers:
                self.add_result("Medium", "Missing Security Headers", 
                              f"Missing: {', '.join(missing_headers)}")
            
            # Check for server information disclosure
            if 'server' in headers:
                self.add_result("Low", "Server Disclosure", f"Server header: {headers['server']}")
            
            # Check for CORS misconfiguration
            if 'access-control-allow-origin' in headers and headers['access-control-allow-origin'] == '*':
                self.add_result("Medium", "Permissive CORS", "Access-Control-Allow-Origin is set to '*'")
            
            # Check for clickjacking protection
            if 'x-frame-options' not in headers:
                self.add_result("Medium", "Clickjacking", "Missing X-Frame-Options header")
            
            # Check for content type sniffing protection
            if 'x-content-type-options' not in headers:
                self.add_result("Low", "Content Type Sniffing", "Missing X-Content-Type-Options header")
            
            self.details_text.insert(tk.END, details + "\n")
        
        except Exception as e:
            self.add_result("Medium", "Headers Error", f"Header analysis failed: {str(e)}")
    
    def test_sql_injection(self, url):
        payloads = [
            "'", "\"", "' OR '1'='1", "' OR 1=1--", 
            "' OR 1=1#", "' OR 1=1/*", "' UNION SELECT null,version()--",
            "' UNION SELECT username,password FROM users--",
            "1 AND 1=1", "1 AND 1=2", "1' AND SLEEP(5)--",
            "1' OR IF(1=1,SLEEP(5),0)--"
        ]
        
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            
            if not params:
                self.add_result("Info", "SQLi Test", "No parameters to test")
                return
            
            vulnerable = False
            details = "SQL Injection Tests:\n"
            
            for param in params:
                for payload in payloads:
                    test_url = url.replace(f"{param}={params[param][0]}", f"{param}={payload}")
                    try:
                        start_time = time.time()
                        response = requests.get(test_url, timeout=5)
                        elapsed = time.time() - start_time
                        
                        if self.detect_sql_errors(response.text):
                            vulnerable = True
                            details += f"Potential SQLi in {param} with payload: {payload} (Error-based)\n"
                            break
                        elif elapsed > 4:  # Time-based detection
                            vulnerable = True
                            details += f"Potential SQLi in {param} with payload: {payload} (Time-based, {elapsed:.2f}s)\n"
                            break
                        elif payload in response.text:  # Boolean-based detection
                            vulnerable = True
                            details += f"Potential SQLi in {param} with payload: {payload} (Boolean-based)\n"
                            break
                    except:
                        continue
            
            if vulnerable:
                self.add_result("High", "SQL Injection", "Potential SQLi vulnerabilities detected")
            else:
                self.add_result("Low", "SQL Injection", "No obvious SQLi vulnerabilities found")
            
            self.details_text.insert(tk.END, details + "\n")
        
        except Exception as e:
            self.add_result("Medium", "SQLi Error", f"SQLi test failed: {str(e)}")
    
    def detect_sql_errors(self, content):
        errors = [
            "SQL syntax", "MySQL server", "ORA-", "syntax error",
            "unclosed quotation mark", "Microsoft OLE DB Provider",
            "ODBC Driver", "PostgreSQL", "SQLite", "MariaDB",
            "SQL error", "database error", "query failed",
            "syntax error", "unknown column", "table not found"
        ]
        return any(error.lower() in content.lower() for error in errors)
    
    def test_xss(self, url):
        payloads = [
            "<script>alert(1)</script>", 
            "<img src=x onerror=alert(1)>",
            "\"><script>alert(1)</script>",
            "javascript:alert(1)",
            "onmouseover=alert(1)",
            "onload=alert(1)",
            "onfocus=alert(1)",
            "svg/onload=alert(1)",
            "alert`1`",
            "eval('alert(1)')"
        ]
        
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            
            if not params:
                self.add_result("Info", "XSS Test", "No parameters to test")
                return
            
            vulnerable = False
            details = "XSS Tests:\n"
            
            for param in params:
                for payload in payloads:
                    test_url = url.replace(f"{param}={params[param][0]}", f"{param}={payload}")
                    try:
                        response = requests.get(test_url, timeout=5)
                        if payload in response.text:
                            vulnerable = True
                            details += f"Potential XSS in {param} with payload: {payload}\n"
                            break
                    except:
                        continue
            
            if vulnerable:
                self.add_result("High", "XSS", "Potential XSS vulnerabilities detected")
            else:
                self.add_result("Low", "XSS", "No obvious XSS vulnerabilities found")
            
            self.details_text.insert(tk.END, details + "\n")
        
        except Exception as e:
            self.add_result("Medium", "XSS Error", f"XSS test failed: {str(e)}")
    
    def test_csrf(self, url):
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            forms = soup.find_all('form')
            
            vulnerable = False
            details = "CSRF Tests:\n"
            
            for form in forms:
                if not form.find('input', {'name': 'csrf_token'}) and \
                   not form.find('input', {'name': 'csrfmiddlewaretoken'}) and \
                   not form.find('input', {'name': '_token'}):
                    vulnerable = True
                    action = form.get('action', 'current URL')
                    method = form.get('method', 'GET').upper()
                    details += f"Form without CSRF protection found (Action: {action}, Method: {method})\n"
                    break
            
            if vulnerable:
                self.add_result("Medium", "CSRF", "Forms without CSRF protection detected")
            else:
                self.add_result("Low", "CSRF", "No obvious CSRF vulnerabilities found")
            
            self.details_text.insert(tk.END, details + "\n")
        
        except Exception as e:
            self.add_result("Medium", "CSRF Error", f"CSRF test failed: {str(e)}")
    
    def test_directory_traversal(self, url):
        payloads = [
            "../../../../etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "....//....//etc/passwd",
            "%2e%2e%2fetc%2fpasswd",
            "..\\..\\..\\windows\\win.ini",
            "%2e%2e%5cwindows%5cwin.ini"
        ]
        
        try:
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            paths = [p for p in parsed.path.split('/') if p]
            
            vulnerable = False
            details = "Directory Traversal Tests:\n"
            
            for payload in payloads:
                test_url = f"{base_url}/{payload}"
                try:
                    response = requests.get(test_url, timeout=5)
                    if "root:" in response.text or "bin:" in response.text or "[fonts]" in response.text:
                        vulnerable = True
                        details += f"Potential directory traversal with payload: {payload}\n"
                        break
                except:
                    continue
            
            if vulnerable:
                self.add_result("High", "Directory Traversal", "Potential directory traversal vulnerabilities detected")
            else:
                self.add_result("Low", "Directory Traversal", "No obvious directory traversal vulnerabilities found")
            
            self.details_text.insert(tk.END, details + "\n")
        
        except Exception as e:
            self.add_result("Medium", "Dir Traversal Error", f"Directory traversal test failed: {str(e)}")
    
    def test_command_injection(self, url):
        payloads = [
            ";id", "|id", "`id`", "$(id)", 
            "|| ping -c 1 localhost", "&& ping -c 1 localhost",
            "| dir", "&& dir", "; dir",
            "| ls", "&& ls", "; ls"
        ]
        
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            
            if not params:
                self.add_result("Info", "Command Injection", "No parameters to test")
                return
            
            vulnerable = False
            details = "Command Injection Tests:\n"
            
            for param in params:
                for payload in payloads:
                    test_url = url.replace(f"{param}={params[param][0]}", f"{param}={params[param][0]}{payload}")
                    try:
                        response = requests.get(test_url, timeout=5)
                        if "uid=" in response.text or "bytes from" in response.text or "Volume Serial" in response.text:
                            vulnerable = True
                            details += f"Potential command injection in {param} with payload: {payload}\n"
                            break
                    except:
                        continue
            
            if vulnerable:
                self.add_result("High", "Command Injection", "Potential command injection vulnerabilities detected")
            else:
                self.add_result("Low", "Command Injection", "No obvious command injection vulnerabilities found")
            
            self.details_text.insert(tk.END, details + "\n")
        
        except Exception as e:
            self.add_result("Medium", "CMD Injection Error", f"Command injection test failed: {str(e)}")
    
    def test_ssrf(self, url):
        """Test for Server-Side Request Forgery vulnerabilities"""
        test_servers = [
            "http://169.254.169.254/latest/meta-data/",  # AWS metadata
            "http://metadata.google.internal/computeMetadata/v1/",  # GCP metadata
            "http://169.254.169.253/latest/meta-data/",  # Azure metadata
            "http://localhost:80",
            "http://127.0.0.1:80"
        ]
        
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            
            if not params:
                self.add_result("Info", "SSRF Test", "No parameters to test")
                return
            
            vulnerable = False
            details = "SSRF Tests:\n"
            
            for param in params:
                for server in test_servers:
                    test_url = url.replace(f"{param}={params[param][0]}", f"{param}={server}")
                    try:
                        response = requests.get(test_url, timeout=5)
                        if response.status_code == 200 and any(
                            keyword in response.text.lower() 
                            for keyword in ["instance-id", "ami-id", "compute", "metadata"]
                        ):
                            vulnerable = True
                            details += f"Potential SSRF in {param} with payload: {server}\n"
                            break
                    except:
                        continue
            
            if vulnerable:
                self.add_result("High", "SSRF", "Potential SSRF vulnerabilities detected")
            else:
                self.add_result("Low", "SSRF", "No obvious SSRF vulnerabilities found")
            
            self.details_text.insert(tk.END, details + "\n")
        
        except Exception as e:
            self.add_result("Medium", "SSRF Error", f"SSRF test failed: {str(e)}")
    
    def test_xxe(self, url):
        """Test for XML External Entity vulnerabilities"""
        xxe_payload = """<?xml version="1.0" encoding="ISO-8859-1"?>
<!DOCTYPE foo [ <!ELEMENT foo ANY >
<!ENTITY xxe SYSTEM "file:///etc/passwd" >]>
<foo>&xxe;</foo>"""
        
        try:
            # First check if the endpoint accepts XML
            response = requests.get(url, timeout=5)
            if "xml" not in response.headers.get('content-type', '').lower():
                self.add_result("Info", "XXE Test", "Endpoint doesn't appear to accept XML")
                return
            
            # Try XXE injection
            headers = {'Content-Type': 'application/xml'}
            response = requests.post(url, data=xxe_payload, headers=headers, timeout=5)
            
            if "root:" in response.text:
                self.add_result("Critical", "XXE", "XML External Entity injection vulnerability detected")
                self.details_text.insert(tk.END, "XXE Test:\nVulnerable to XXE injection\n")
            else:
                self.add_result("Low", "XXE", "No obvious XXE vulnerabilities found")
                self.details_text.insert(tk.END, "XXE Test:\nNo XXE vulnerability detected\n")
        
        except Exception as e:
            self.add_result("Medium", "XXE Error", f"XXE test failed: {str(e)}")
    
    def test_idor(self, url):
        """Test for Insecure Direct Object References"""
        try:
            # This is a simple test that checks for sequential IDs
            # In a real scanner, you'd need more sophisticated tests
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            
            id_params = [p for p in params if 'id' in p.lower()]
            
            if not id_params:
                self.add_result("Info", "IDOR Test", "No ID parameters to test")
                return
            
            vulnerable = False
            details = "IDOR Tests:\n"
            
            for param in id_params:
                original_id = params[param][0]
                if original_id.isdigit():
                    test_id = str(int(original_id) + 1)
                    test_url = url.replace(f"{param}={original_id}", f"{param}={test_id}")
                    
                    try:
                        original_response = requests.get(url, timeout=5)
                        test_response = requests.get(test_url, timeout=5)
                        
                        if test_response.status_code == 200 and \
                           test_response.text != original_response.text and \
                           len(test_response.text) > 100:  # Basic check to avoid 404 pages
                            vulnerable = True
                            details += f"Potential IDOR in {param} by changing {original_id} to {test_id}\n"
                    except:
                        continue
            
            if vulnerable:
                self.add_result("Medium", "IDOR", "Potential IDOR vulnerabilities detected")
            else:
                self.add_result("Low", "IDOR", "No obvious IDOR vulnerabilities found")
            
            self.details_text.insert(tk.END, details + "\n")
        
        except Exception as e:
            self.add_result("Medium", "IDOR Error", f"IDOR test failed: {str(e)}")
    
    def crawl_pages(self, base_url):
        try:
            visited = set()
            to_visit = {base_url}
            max_pages = self.scan_config['max_pages']
            
            details = "Crawling Results:\n"
            
            while to_visit and len(visited) < max_pages:
                url = to_visit.pop()
                if url in visited:
                    continue
                
                try:
                    response = requests.get(url, timeout=5)
                    visited.add(url)
                    details += f"Found: {url} ({response.status_code})\n"
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find links
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if href.startswith('http') and base_url in href and href not in visited:
                            to_visit.add(href)
                        elif href.startswith('/'):
                            absolute = urljoin(base_url, href)
                            if absolute not in visited:
                                to_visit.add(absolute)
                    
                    # Check for forms
                    forms = soup.find_all('form')
                    if forms:
                        details += f"Found {len(forms)} forms on {url}\n"
                    
                    # Check for sensitive content
                    sensitive_keywords = [
                        'password', 'secret', 'key', 'token', 
                        'admin', 'login', 'credentials', 'backup'
                    ]
                    page_text = soup.get_text().lower()
                    found_keywords = [kw for kw in sensitive_keywords if kw in page_text]
                    if found_keywords:
                        details += f"Found sensitive keywords: {', '.join(found_keywords)}\n"
                    
                    self.root.update()
                except:
                    continue
            
            details += f"\nCrawled {len(visited)} pages\n"
            self.details_text.insert(tk.END, details + "\n")
            
            if len(visited) >= max_pages:
                self.add_result("Info", "Crawl Limit", f"Limited to {max_pages} pages for demo")
        
        except Exception as e:
            self.add_result("Medium", "Crawl Error", f"Crawling failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    
    # Set Windows style if available
    if sys.platform == "win32":
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
        
    app = AdvancedVulnerabilityScanner(root)
    root.mainloop()