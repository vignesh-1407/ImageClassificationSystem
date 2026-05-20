"""
============================================================
  IMAGE CLASSIFICATION SYSTEM — GUI VERSION
  Built with Python + Tkinter + TensorFlow
============================================================
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading

# Add src to path so we can import predict module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from predict import get_prediction

# ── Color Palette ──────────────────────────────────────────────────────────────
BG_DARK      = "#0D1117"
BG_CARD      = "#161B22"
BG_PANEL     = "#1C2128"
ACCENT_GREEN = "#00FF87"
ACCENT_BLUE  = "#58A6FF"
ACCENT_RED   = "#FF4D4D"
ACCENT_YELLOW= "#F0C040"
TEXT_PRIMARY = "#E6EDF3"
TEXT_MUTED   = "#8B949E"
BTN_UPLOAD   = "#1D4ED8"
BTN_PREDICT  = "#238636"
BTN_HOVER    = "#30363D"


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classification System — AI")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        self.image_path = None
        self.original_image = None
        
        self._build_ui()
        self._update_status("Ready — Upload an image to begin", ACCENT_YELLOW)

    # ── UI Builder ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=BG_CARD, height=60)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="🧠  IMAGE CLASSIFICATION SYSTEM",
                 font=("Segoe UI", 18, "bold"),
                 fg=ACCENT_BLUE, bg=BG_CARD).pack(side="left", padx=20, pady=12)

        tk.Label(header, text="Powered by CNN + TensorFlow",
                 font=("Segoe UI", 10),
                 fg=TEXT_MUTED, bg=BG_CARD).pack(side="left", padx=4, pady=12)

        # Version badge
        badge = tk.Label(header, text=" v1.0 ", font=("Segoe UI", 9, "bold"),
                         fg=BG_DARK, bg=ACCENT_BLUE, padx=6, pady=2)
        badge.pack(side="right", padx=20, pady=18)

        # ── Main body ───────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=12, pady=8)

        # LEFT — Image Preview
        left = tk.Frame(body, bg=BG_DARK)
        left.pack(side="left", fill="both", expand=True)

        # Image container
        img_container = tk.Frame(left, bg=BG_CARD, bd=0,
                                  highlightthickness=2,
                                  highlightbackground=ACCENT_BLUE)
        img_container.pack(fill="both", expand=True, pady=(0, 8))

        self.image_label = tk.Label(img_container, bg="#000000",
                                     text="🖼️  No Image Selected\nClick UPLOAD IMAGE to browse",
                                     font=("Segoe UI", 14), fg=TEXT_MUTED)
        self.image_label.pack(fill="both", expand=True)

        # Status bar below image
        status_bar = tk.Frame(left, bg=BG_CARD, height=32)
        status_bar.pack(fill="x")
        status_bar.pack_propagate(False)

        self.status_label = tk.Label(status_bar, text="● Ready",
                                      font=("Segoe UI", 10, "bold"),
                                      fg=ACCENT_YELLOW, bg=BG_CARD)
        self.status_label.pack(side="left", padx=12, pady=6)

        # RIGHT — Control Panel & Results
        right = tk.Frame(body, bg=BG_DARK, width=320)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        self._build_controls(right)
        self._build_results_panel(right)

    def _build_controls(self, parent):
        card = tk.Frame(parent, bg=BG_CARD, padx=12, pady=12)
        card.pack(fill="x", pady=(0, 8))

        tk.Label(card, text="ACTIONS",
                 font=("Segoe UI", 9, "bold"),
                 fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w")
        tk.Frame(card, bg=BG_PANEL, height=1).pack(fill="x", pady=6)

        # UPLOAD button
        self.btn_upload = self._make_button(card, "📂  UPLOAD IMAGE", BTN_UPLOAD, self.upload_image)
        self.btn_upload.pack(fill="x", pady=(0, 6))

        # PREDICT button
        self.btn_predict = self._make_button(card, "✨  PREDICT", BTN_PREDICT, self.start_prediction)
        self.btn_predict.pack(fill="x")
        self.btn_predict.config(state="disabled")

    def _build_results_panel(self, parent):
        card = tk.Frame(parent, bg=BG_CARD, padx=12, pady=12)
        card.pack(fill="both", expand=True)

        tk.Label(card, text="PREDICTION RESULTS",
                 font=("Segoe UI", 9, "bold"),
                 fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w")
        tk.Frame(card, bg=BG_PANEL, height=1).pack(fill="x", pady=6)

        # Container for results
        self.results_frame = tk.Frame(card, bg=BG_CARD)
        self.results_frame.pack(fill="both", expand=True, pady=10)
        
        self.result_widgets = []
        
        # Initial empty state
        self._show_empty_results()

    def _show_empty_results(self):
        self._clear_results()
        lbl = tk.Label(self.results_frame, text="Results will appear here\nafter prediction.",
                       font=("Segoe UI", 10, "italic"), fg=TEXT_MUTED, bg=BG_CARD)
        lbl.pack(pady=40)
        self.result_widgets.append(lbl)

    def _clear_results(self):
        for widget in self.result_widgets:
            widget.destroy()
        self.result_widgets.clear()

    def _make_button(self, parent, text, color, command):
        btn = tk.Button(parent, text=text,
                        font=("Segoe UI", 10, "bold"),
                        fg="white", bg=color,
                        activeforeground="white",
                        activebackground=color,
                        relief="flat", cursor="hand2",
                        padx=10, pady=10,
                        command=command, bd=0)
        btn.bind("<Enter>", lambda e: btn.config(bg=self._lighten(color)) if btn['state'] == 'normal' else None)
        btn.bind("<Leave>", lambda e: btn.config(bg=color) if btn['state'] == 'normal' else None)
        return btn

    def _lighten(self, hex_color):
        r = min(255, int(hex_color[1:3], 16) + 30)
        g = min(255, int(hex_color[3:5], 16) + 30)
        b = min(255, int(hex_color[5:7], 16) + 30)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _update_status(self, msg, color=TEXT_PRIMARY):
        self.status_label.config(text=f"●  {msg}", fg=color)

    # ── Actions ────────────────────────────────────────────────────────
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        self.image_path = file_path
        self._show_empty_results()
        
        try:
            self.original_image = Image.open(self.image_path).convert("RGB")
            self._display_image()
            self.btn_predict.config(state="normal", bg=BTN_PREDICT)
            self._update_status(f"Loaded: {os.path.basename(self.image_path)}", ACCENT_BLUE)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            self.image_path = None
            self.original_image = None
            self.btn_predict.config(state="disabled", bg=TEXT_MUTED)
            self.image_label.config(image="", text="🖼️  Failed to Load Image")

    def _display_image(self):
        if not self.original_image:
            return
            
        # Update UI first
        self.root.update_idletasks()
        
        lw = self.image_label.winfo_width()
        lh = self.image_label.winfo_height()
        
        if lw < 10 or lh < 10:
            lw, lh = 600, 500
            
        w, h = self.original_image.size
        ratio = min(lw / w, lh / h)
        nw, nh = int(w * ratio), int(h * ratio)
        
        resized_img = self.original_image.resize((nw, nh), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized_img)
        
        self.image_label.config(image=tk_img, text="")
        self.image_label.image = tk_img
        
        # Handle window resize to dynamically update image
        self.image_label.bind("<Configure>", self._on_resize)
        
    def _on_resize(self, event):
        # Throttle resizing or handle dynamically
        if self.original_image and event.width > 10 and event.height > 10:
            # simple debounce logic could be added here if it's too slow
            pass

    def start_prediction(self):
        if not self.image_path:
            return
            
        self.btn_predict.config(state="disabled", bg=TEXT_MUTED)
        self.btn_upload.config(state="disabled", bg=TEXT_MUTED)
        self._update_status("Predicting... Please wait", ACCENT_YELLOW)
        
        # Show loading indicator in results
        self._clear_results()
        lbl = tk.Label(self.results_frame, text="⏳ Analyzing Image...",
                       font=("Segoe UI", 12, "bold"), fg=ACCENT_BLUE, bg=BG_CARD)
        lbl.pack(pady=40)
        self.result_widgets.append(lbl)
        
        # Run prediction in background to keep GUI responsive
        threading.Thread(target=self._run_prediction, daemon=True).start()
        
    def _run_prediction(self):
        result = get_prediction(self.image_path)
        
        # Update GUI on main thread
        self.root.after(0, self._handle_prediction_result, result)
        
    def _handle_prediction_result(self, result):
        self.btn_upload.config(state="normal", bg=BTN_UPLOAD)
        self.btn_predict.config(state="normal", bg=BTN_PREDICT)
        self._clear_results()
        
        if "error" in result:
            self._update_status("Prediction Failed", ACCENT_RED)
            messagebox.showerror("Prediction Error", result["error"])
            self._show_empty_results()
            return
            
        self._update_status("Prediction Complete", ACCENT_GREEN)
        
        # Display top 3 results
        predictions = result["predictions"]
        
        tk.Label(self.results_frame, text="TOP PREDICTIONS",
                 font=("Segoe UI", 11, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD).pack(anchor="w", pady=(0, 15))
                 
        for i, pred in enumerate(predictions):
            class_name = pred["class_name"].upper()
            confidence = pred["confidence"]
            
            # Determine color based on rank
            if i == 0:
                color = ACCENT_GREEN
            elif i == 1:
                color = ACCENT_BLUE
            else:
                color = ACCENT_YELLOW
                
            # Container for this prediction
            pred_frame = tk.Frame(self.results_frame, bg=BG_PANEL, padx=15, pady=10)
            pred_frame.pack(fill="x", pady=5)
            self.result_widgets.append(pred_frame)
            
            # Label row
            lbl_row = tk.Frame(pred_frame, bg=BG_PANEL)
            lbl_row.pack(fill="x")
            
            tk.Label(lbl_row, text=class_name, font=("Segoe UI", 12, "bold"),
                     fg=color, bg=BG_PANEL).pack(side="left")
            tk.Label(lbl_row, text=f"{confidence:.2f}%", font=("Segoe UI", 12, "bold"),
                     fg=TEXT_PRIMARY, bg=BG_PANEL).pack(side="right")
                     
            # Progress bar simulation
            bar_bg = tk.Frame(pred_frame, bg=BG_DARK, height=8)
            bar_bg.pack(fill="x", pady=(8, 0))
            bar_bg.pack_propagate(False)
            
            bar_fill = tk.Frame(bar_bg, bg=color, width=int(confidence * 2.5)) # Rough width scaling
            bar_fill.pack(side="left", fill="y")


def apply_style():
    style = ttk.Style()
    style.theme_use("clam")

if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        import subprocess, sys
        print("[Installing Pillow…]")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image, ImageTk
        
    try:
        import tensorflow
    except ImportError:
        import subprocess, sys
        print("[Installing TensorFlow…]")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])

    root = tk.Tk()
    apply_style()
    app = ImageClassifierApp(root)
    root.mainloop()
