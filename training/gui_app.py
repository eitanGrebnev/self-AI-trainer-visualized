import shutil
import subprocess
import sys
import threading
import json
import re
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"

CHAT_TRAIN_SCRIPT = TRAINING_ROOT / "training_for_chat" / "train_me.py"
CHAT_CHAT_SCRIPT = TRAINING_ROOT / "training_for_chat" / "chat.py"
STYLE_TRAIN_SCRIPT = TRAINING_ROOT / "training_for_style" / "train_style.py"
STYLE_CHAT_SCRIPT = TRAINING_ROOT / "training_for_style" / "chat_style.py"

TRAINED_MATERIAL_DIR = PROJECT_ROOT / "trained material"
STYLE_MATERIAL_DIR = PROJECT_ROOT / "style material"
CHAT_MODEL_DIR = PROJECT_ROOT / "my_custom_model"
STYLE_MODEL_DIR = PROJECT_ROOT / "my_style_model"
CONSENT_FILE = TRAINING_ROOT / ".user_consent.json"
MIT_LICENSE_FILE = TRAINING_ROOT / "LICENSE_MIT.txt"

SUPPORTED_EXTENSIONS = {".txt", ".docx", ".pdf"}

DEFAULT_PACKAGES = [
    "unsloth",
    "torch",
    "transformers",
    "datasets",
    "trl",
    "python-docx",
    "pypdf",
    "rich",
    "accelerate",
    "bitsandbytes",
    "sentencepiece",
]


class TrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Unsloth Trainer Studio")
        self.root.geometry("1080x760")
        self.root.minsize(980, 680)

        self.current_process = None
        self.chat_window = None
        self.mode_var = tk.StringVar(value="talking")
        self.theme_var = tk.StringVar(value="dark")
        self.custom_theme = None

        self._configure_styles()
        self._build_ui()
        self.ensure_directories()
        self.refresh_status()
        self.prompt_license_acceptance_if_needed()

    def _configure_styles(self):
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

        self.palette = {
            "dark": {
                "bg": "#0f172a",
                "panel": "#111827",
                "panel_alt": "#1f2937",
                "text": "#e5e7eb",
                "muted": "#9ca3af",
                "accent": "#38bdf8",
                "ok": "#22c55e",
            },
            "light": {
                "bg": "#f4f7fb",
                "panel": "#ffffff",
                "panel_alt": "#e8eef5",
                "text": "#0f172a",
                "muted": "#475569",
                "accent": "#0369a1",
                "ok": "#15803d",
            },
            "kawaii": {
                "bg": "#fff0f6",
                "panel": "#fff7fb",
                "panel_alt": "#ffd8ea",
                "text": "#7a2152",
                "muted": "#a14577",
                "accent": "#ff4fa3",
                "ok": "#ff74b8",
            },
        }
        self.apply_theme()

    def apply_theme(self):
        if self.theme_var.get() == "custom" and self.custom_theme:
            colors = self.custom_theme
        else:
            colors = self.palette[self.theme_var.get()]
        self.root.configure(bg=colors["bg"])

        self.style.configure("TFrame", background=colors["bg"])
        self.style.configure("Card.TFrame", background=colors["panel"])
        self.style.configure("Panel.TFrame", background=colors["panel_alt"])
        self.style.configure("TLabel", background=colors["bg"], foreground=colors["text"], font=("Segoe UI", 10))
        self.style.configure("Title.TLabel", background=colors["bg"], foreground=colors["text"], font=("Segoe UI Semibold", 18))
        self.style.configure("Subtle.TLabel", background=colors["bg"], foreground=colors["muted"], font=("Segoe UI", 9))
        self.style.configure("Badge.TLabel", background=colors["panel_alt"], foreground=colors["ok"], font=("Segoe UI Semibold", 9))
        self.style.configure("TButton", font=("Segoe UI Semibold", 9), padding=8)
        self.style.configure("Primary.TButton", foreground=colors["accent"])
        self.style.configure("TRadiobutton", background=colors["panel"], foreground=colors["text"], font=("Segoe UI", 10))
        self.style.map("TButton", background=[("active", colors["panel_alt"])])

        if hasattr(self, "output"):
            self.output.configure(
                bg=colors["panel"],
                fg=colors["text"],
                insertbackground=colors["text"],
                selectbackground=colors["accent"],
                relief=tk.FLAT,
                padx=10,
                pady=10,
            )

    def _build_ui(self):
        shell = ttk.Frame(self.root, padding=16)
        shell.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(shell)
        header.pack(fill=tk.X)

        ttk.Label(header, text="Unsloth Trainer Studio", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="Train and chat with clean mode separation", style="Subtle.TLabel").pack(side=tk.LEFT, padx=12)

        theme_panel = ttk.Frame(header, style="Panel.TFrame", padding=(10, 6))
        theme_panel.pack(side=tk.RIGHT)
        ttk.Label(theme_panel, text="Theme", style="Subtle.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(theme_panel, text="Dark", value="dark", variable=self.theme_var, command=self.apply_theme).pack(side=tk.LEFT)
        ttk.Radiobutton(theme_panel, text="Light", value="light", variable=self.theme_var, command=self.apply_theme).pack(side=tk.LEFT)
        ttk.Radiobutton(theme_panel, text="Kawaii :3", value="kawaii", variable=self.theme_var, command=self.apply_theme).pack(side=tk.LEFT)
        ttk.Button(theme_panel, text="Custom CSS Theme", command=self.open_custom_css_dialog).pack(side=tk.LEFT, padx=(8, 0))

        control_card = ttk.Frame(shell, style="Card.TFrame", padding=14)
        control_card.pack(fill=tk.X, pady=(12, 10))

        mode_row = ttk.Frame(control_card, style="Card.TFrame")
        mode_row.pack(fill=tk.X)
        ttk.Label(mode_row, text="Training Mode", font=("Segoe UI Semibold", 10)).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Talking (school material Q&A)", value="talking", variable=self.mode_var, command=self.refresh_status).pack(side=tk.LEFT, padx=12)
        ttk.Radiobutton(mode_row, text="Rephrasing/Style", value="style", variable=self.mode_var, command=self.refresh_status).pack(side=tk.LEFT)

        action_row = ttk.Frame(control_card, style="Card.TFrame")
        action_row.pack(fill=tk.X, pady=(10, 0))

        self.btn_setup = ttk.Button(action_row, text="Setup Environment", command=self.setup_environment)
        self.btn_setup.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_upload = ttk.Button(action_row, text="Upload Files", command=self.upload_files)
        self.btn_upload.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_train = ttk.Button(action_row, text="Train Model", style="Primary.TButton", command=self.run_train)
        self.btn_train.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_chat = ttk.Button(action_row, text="Start Chat", command=self.run_chat)
        self.btn_chat.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_stop = ttk.Button(action_row, text="Stop", command=self.stop_process)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_build = ttk.Button(action_row, text="Build EXE", command=self.build_exe)
        self.btn_build.pack(side=tk.LEFT, padx=(0, 8))

        status_bar = ttk.Frame(shell, style="Panel.TFrame", padding=(12, 8))
        status_bar.pack(fill=tk.X)
        self.status_label = ttk.Label(status_bar, text="Status: idle")
        self.status_label.pack(side=tk.LEFT)

        self.path_label = ttk.Label(status_bar, text="", style="Subtle.TLabel")
        self.path_label.pack(side=tk.LEFT, padx=12)

        self.model_state_label = ttk.Label(status_bar, text="Model: not ready", style="Badge.TLabel")
        self.model_state_label.pack(side=tk.RIGHT)

        log_card = ttk.Frame(shell, style="Card.TFrame", padding=12)
        log_card.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        ttk.Label(log_card, text="Activity Log", font=("Segoe UI Semibold", 10)).pack(anchor="w")

        output_frame = ttk.Frame(log_card, style="Card.TFrame")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.output = tk.Text(output_frame, wrap=tk.WORD, height=30, borderwidth=0, highlightthickness=0)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output.configure(yscrollcommand=scrollbar.set)
        self.output.configure(state=tk.DISABLED)
        self.apply_theme()

    def current_mode_paths(self):
        if self.mode_var.get() == "talking":
            return {
                "train_script": CHAT_TRAIN_SCRIPT,
                "chat_script": CHAT_CHAT_SCRIPT,
                "data_dir": TRAINED_MATERIAL_DIR,
                "model_dir": CHAT_MODEL_DIR,
                "mode_name": "Talking",
            }
        return {
            "train_script": STYLE_TRAIN_SCRIPT,
            "chat_script": STYLE_CHAT_SCRIPT,
            "data_dir": STYLE_MATERIAL_DIR,
            "model_dir": STYLE_MODEL_DIR,
            "mode_name": "Style",
        }

    def ensure_directories(self):
        TRAINED_MATERIAL_DIR.mkdir(parents=True, exist_ok=True)
        STYLE_MATERIAL_DIR.mkdir(parents=True, exist_ok=True)
        CHAT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        STYLE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def log(self, text):
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, text + "\n")
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)

    def set_status(self, text):
        self.status_label.config(text=f"Status: {text}")

    def model_is_ready(self, model_dir: Path) -> bool:
        return (model_dir / "adapter_model.safetensors").exists()

    def refresh_status(self):
        paths = self.current_mode_paths()
        mode_name = paths["mode_name"]
        data_dir = paths["data_dir"]
        model_dir = paths["model_dir"]

        model_ready = self.model_is_ready(model_dir)
        self.btn_chat.config(state=(tk.NORMAL if model_ready else tk.DISABLED))
        self.model_state_label.config(text=("Model: ready" if model_ready else "Model: not ready"))

        self.path_label.config(text=f"{mode_name} | data: {data_dir.name} | model: {model_dir.name}")

    def _run_subprocess(self, cmd, description, on_success=None):
        if self.current_process is not None:
            messagebox.showwarning("Busy", "A process is already running. Stop it first.")
            return

        self.log(f"\n--- {description} ---")
        self.log("Command: " + " ".join(str(c) for c in cmd))
        self.set_status(description)

        def worker():
            try:
                self.current_process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in self.current_process.stdout:
                    self.root.after(0, self.log, line.rstrip())
                return_code = self.current_process.wait()
                self.root.after(0, self.log, f"Process finished with exit code {return_code}")
                if return_code == 0 and on_success is not None:
                    self.root.after(0, on_success)
            except Exception as exc:
                self.root.after(0, self.log, f"Process error: {exc}")
            finally:
                self.current_process = None
                self.root.after(0, self.set_status, "idle")
                self.root.after(0, self.refresh_status)

        threading.Thread(target=worker, daemon=True).start()

    def upload_files(self):
        paths = self.current_mode_paths()
        target = paths["data_dir"]

        files = filedialog.askopenfilenames(
            title="Select training files",
            filetypes=[("Documents", "*.txt *.docx *.pdf"), ("All files", "*.*")],
        )
        if not files:
            return

        copied = 0
        skipped = 0
        for src in files:
            src_path = Path(src)
            if src_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped += 1
                continue
            dst = target / src_path.name
            shutil.copy2(src_path, dst)
            copied += 1

        self.log(f"Uploaded {copied} files to {target}")
        if skipped:
            self.log(f"Skipped {skipped} files with unsupported extension")

    def setup_environment(self):
        self.ensure_directories()
        pip_cmd = [sys.executable, "-m", "pip", "install", *DEFAULT_PACKAGES]
        self._run_subprocess(pip_cmd, "Installing dependencies")

    def run_train(self):
        paths = self.current_mode_paths()
        script = paths["train_script"]
        data_dir = paths["data_dir"]

        if not any(data_dir.glob("*")):
            messagebox.showwarning("No Data", f"Add files to {data_dir} first.")
            return

        cmd = [sys.executable, str(script)]
        self._run_subprocess(cmd, f"Training {paths['mode_name']} model")

    def run_chat(self):
        if self.current_process is not None:
            messagebox.showwarning("Busy", "A training/setup/build process is running. Stop it first.")
            return

        paths = self.current_mode_paths()
        model_dir = paths["model_dir"]
        if not self.model_is_ready(model_dir):
            messagebox.showwarning("Model Missing", "Train the model first. Chat is disabled until a model exists.")
            self.refresh_status()
            return

        if self.chat_window is not None:
            try:
                self.chat_window.focus_force()
                return
            except Exception:
                self.chat_window = None

        chat_title = f"{paths['mode_name']} Chat"
        try:
            self.chat_window = ChatSessionWindow(
                parent=self,
                title=chat_title,
                script_path=paths["chat_script"],
                colors=self.get_active_colors(),
            )
        except RuntimeError as exc:
            self.chat_window = None
            self.log(str(exc))

    def stop_process(self):
        if self.current_process is None:
            self.log("No running process to stop.")
            return

        try:
            self.current_process.terminate()
            self.log("Sent terminate signal to running process.")
        except Exception as exc:
            self.log(f"Failed to stop process: {exc}")

    def build_exe(self):
        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--onefile",
            "--windowed",
            "--name",
            "UnslothTrainerGUI",
            str(Path(__file__).resolve()),
        ]
        self._run_subprocess(cmd, "Building GUI EXE")

    def open_custom_css_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Custom CSS Theme")
        dialog.geometry("700x500")
        dialog.transient(self.root)
        dialog.grab_set()

        outer = ttk.Frame(dialog, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text="Paste CSS-like theme variables", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Supported keys: --bg, --panel, --panel-alt, --text, --muted, --accent, --ok\n"
                "Example:\n"
                "--bg: #10131a; --panel: #171b24; --panel-alt: #202738; --text: #e5e7eb;"
            ),
            style="Subtle.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        text = tk.Text(outer, wrap=tk.WORD, height=16, borderwidth=1, relief=tk.SOLID)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(
            "1.0",
            "--bg: #fff5fb;\n"
            "--panel: #ffffff;\n"
            "--panel-alt: #ffd6ea;\n"
            "--text: #6b2147;\n"
            "--muted: #9a4a72;\n"
            "--accent: #ff4fa3;\n"
            "--ok: #ff87c1;\n",
        )

        buttons = ttk.Frame(outer, padding=(0, 8, 0, 0))
        buttons.pack(fill=tk.X)

        def apply_custom():
            raw = text.get("1.0", tk.END)
            parsed = self.parse_custom_css(raw)
            if parsed is None:
                messagebox.showerror("Invalid theme", "Could not parse your CSS variables. Check key names and hex colors.")
                return
            self.custom_theme = parsed
            self.theme_var.set("custom")
            self.apply_theme()
            self.log("Applied custom CSS theme.")
            dialog.destroy()

        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Apply Theme", style="Primary.TButton", command=apply_custom).pack(side=tk.RIGHT, padx=(0, 8))

        self.root.wait_window(dialog)

    def parse_custom_css(self, css_text):
        base = dict(self.palette["light"])
        key_aliases = {
            "bg": "bg",
            "background": "bg",
            "panel": "panel",
            "panel-alt": "panel_alt",
            "panel_alt": "panel_alt",
            "text": "text",
            "muted": "muted",
            "accent": "accent",
            "ok": "ok",
        }
        pattern = re.compile(r"--?([a-zA-Z0-9_-]+)\s*:\s*([^;\n]+)")
        matches = pattern.findall(css_text)
        if not matches:
            return None

        updated = False
        for raw_key, raw_value in matches:
            key = raw_key.strip().lower()
            value = raw_value.strip()
            mapped = key_aliases.get(key)
            if mapped is None:
                continue
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
                continue
            base[mapped] = value
            updated = True

        return base if updated else None

    def get_active_colors(self):
        if self.theme_var.get() == "custom" and self.custom_theme:
            return self.custom_theme
        return self.palette[self.theme_var.get()]

    def prompt_license_acceptance_if_needed(self):
        if self._load_consent():
            return

        license_text = self._read_license_text()
        dialog = tk.Toplevel(self.root)
        dialog.title("License Agreement")
        dialog.geometry("780x560")
        dialog.transient(self.root)
        dialog.grab_set()

        outer = ttk.Frame(dialog, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text="License Agreement", font=("Segoe UI Semibold", 14)).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "You must accept this agreement before using the app. "
                "This is a sample MIT license and notice block."
            ),
            style="Subtle.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        license_box = tk.Text(outer, wrap=tk.WORD, height=20, borderwidth=1, relief=tk.SOLID)
        license_box.insert("1.0", license_text)
        license_box.configure(state=tk.DISABLED)
        license_box.pack(fill=tk.BOTH, expand=True)

        accept_var = tk.BooleanVar(value=False)
        controls = ttk.Frame(outer, padding=(0, 10, 0, 0))
        controls.pack(fill=tk.X)
        ttk.Checkbutton(
            controls,
            text="I have read and accept the license terms.",
            variable=accept_var,
        ).pack(side=tk.LEFT)

        button_row = ttk.Frame(outer, padding=(0, 8, 0, 0))
        button_row.pack(fill=tk.X)

        def on_decline():
            dialog.destroy()
            self.root.destroy()

        def on_accept():
            if not accept_var.get():
                messagebox.showwarning("Agreement required", "You must accept the terms to continue.")
                return
            self._save_consent()
            dialog.destroy()

        ttk.Button(button_row, text="Decline", command=on_decline).pack(side=tk.RIGHT)
        ttk.Button(button_row, text="Accept", style="Primary.TButton", command=on_accept).pack(side=tk.RIGHT, padx=(0, 8))

        self.root.wait_window(dialog)

    def _read_license_text(self):
        if MIT_LICENSE_FILE.exists():
            return MIT_LICENSE_FILE.read_text(encoding="utf-8", errors="ignore")
        return "MIT License\n\nCopyright (c) [year] [name]"

    def _load_consent(self):
        if not CONSENT_FILE.exists():
            return False
        try:
            payload = json.loads(CONSENT_FILE.read_text(encoding="utf-8"))
            return bool(payload.get("accepted"))
        except Exception:
            return False

    def _save_consent(self):
        CONSENT_FILE.write_text(json.dumps({"accepted": True}, indent=2), encoding="utf-8")


class ChatSessionWindow:
    def __init__(self, parent, title, script_path, colors):
        self.parent = parent
        self.script_path = Path(script_path)
        self.colors = colors
        self.process = None
        self.awaiting_response = False
        self.mode = "style" if "chat_style.py" in self.script_path.name else "talking"
        self.voice_sample = ""

        if self.mode == "style":
            sample = self._prompt_voice_sample()
            if sample is None:
                raise RuntimeError("Style chat cancelled: voice sample is required.")
            self.voice_sample = sample

        self.win = tk.Toplevel(parent.root)
        self.win.title(title)
        self.win.geometry("920x620")
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_ui()
        self._start_chat_process()

    def focus_force(self):
        self.win.deiconify()
        self.win.lift()
        self.win.focus_force()

    def _build_ui(self):
        wrapper = ttk.Frame(self.win, padding=12)
        wrapper.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(wrapper)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Live Chat", font=("Segoe UI Semibold", 12)).pack(side=tk.LEFT)
        self.status_label = ttk.Label(header, text="Status: starting", style="Subtle.TLabel")
        self.status_label.pack(side=tk.RIGHT)

        transcript_frame = ttk.Frame(wrapper)
        transcript_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 8))

        self.transcript = tk.Text(
            transcript_frame,
            wrap=tk.WORD,
            borderwidth=0,
            highlightthickness=0,
            bg=self.colors["panel"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            padx=10,
            pady=10,
        )
        self.transcript.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.transcript.tag_configure("user", foreground=self.colors["accent"], font=("Segoe UI Semibold", 10))
        self.transcript.tag_configure("assistant", foreground=self.colors["text"], font=("Segoe UI", 10))
        self.transcript.tag_configure("system", foreground=self.colors["muted"], font=("Consolas", 9))
        self.transcript.configure(state=tk.DISABLED)

        scroll = ttk.Scrollbar(transcript_frame, orient=tk.VERTICAL, command=self.transcript.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.transcript.configure(yscrollcommand=scroll.set)

        ttk.Label(wrapper, text="Enter = send | Shift+Enter = newline", style="Subtle.TLabel").pack(anchor="w", pady=(0, 6))

        input_row = ttk.Frame(wrapper)
        input_row.pack(fill=tk.X)
        self.entry = tk.Text(
            input_row,
            wrap=tk.WORD,
            height=5,
            borderwidth=1,
            relief=tk.SOLID,
            bg=self.colors["panel_alt"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            padx=8,
            pady=8,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", self._handle_enter)
        self.entry.bind("<Shift-Return>", self._handle_shift_enter)

        self.send_btn = ttk.Button(input_row, text="Send", style="Primary.TButton", command=self.send_message)
        self.send_btn.pack(side=tk.LEFT, padx=(8, 0))

    def _append(self, text, tag="system"):
        self.transcript.configure(state=tk.NORMAL)
        self.transcript.insert(tk.END, text + "\n", tag)
        self.transcript.see(tk.END)
        self.transcript.configure(state=tk.DISABLED)

    def _prompt_voice_sample(self):
        dialog = tk.Toplevel(self.parent.root)
        dialog.title("Style Voice Sample")
        dialog.geometry("680x420")
        dialog.transient(self.parent.root)
        dialog.grab_set()

        outer = ttk.Frame(dialog, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text="Paste your voice sample", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        ttk.Label(
            outer,
            text="This sample sets the style for the rewrite chat session.",
            style="Subtle.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        sample_box = tk.Text(outer, wrap=tk.WORD, height=14, borderwidth=1, relief=tk.SOLID)
        sample_box.pack(fill=tk.BOTH, expand=True)

        result = {"value": None}

        buttons = ttk.Frame(outer, padding=(0, 8, 0, 0))
        buttons.pack(fill=tk.X)

        def on_cancel():
            dialog.destroy()

        def on_use():
            value = sample_box.get("1.0", tk.END).strip()
            if not value:
                messagebox.showwarning("Voice sample required", "Please paste a voice sample to continue.")
                return
            result["value"] = value
            dialog.destroy()

        ttk.Button(buttons, text="Cancel", command=on_cancel).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Use Sample", style="Primary.TButton", command=on_use).pack(side=tk.RIGHT, padx=(0, 8))

        self.parent.root.wait_window(dialog)
        return result["value"]

    def _set_status(self, text):
        self.status_label.config(text=f"Status: {text}")

    def _start_chat_process(self):
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(self.script_path)],
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,  
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env={**dict(os.environ), "STYLE_VOICE_SAMPLE": self.voice_sample},
            )
            self._set_status("ready")
            self._append("Chat process started.", "system")
        except Exception as exc:
            self._append(f"Failed to start chat process: {exc}", "system")
            self._set_status("error")
            return

        thread = threading.Thread(target=self._read_output_loop, daemon=True)
        thread.start()

    def _read_output_loop(self):
        try:
            for raw_line in self.process.stdout:
                line = raw_line.rstrip("\n")
                self.parent.root.after(0, self._handle_model_line, line)
        except Exception as exc:
            self.parent.root.after(0, self._append, f"Read error: {exc}", "system")
        finally:
            self.parent.root.after(0, self._set_status, "closed")

    def _handle_model_line(self, line):
        stripped = line.strip()
        if not stripped:
            return
        ignored_prefixes = (
            "Unsloth ",
            "🦥 Unsloth",
            "\\   /|",
            "O^O/ ",
            "\"-____-\"",
            "==((====))==",
            "Commands:",
            "CHAT WITH YOUR",
            "STYLE REWRITE CHAT",
            "Type '",
            "Paste AI",
            "Loading model...",
            "Loading style rewrite model...",
            "============================================================",
            "================================================================",
        )
        if stripped.startswith(ignored_prefixes):
            return
        if stripped.startswith("[You]:") or stripped.startswith("[Text To Rewrite]:") or stripped.startswith("[Voice Sample]"):
            return
        if stripped.startswith("[Model]:"):
            content = stripped.replace("[Model]:", "", 1).strip()
            self._append(f"Assistant: {content}", "assistant")
            self.awaiting_response = False
            self.entry.configure(state=tk.NORMAL)
            self.send_btn.configure(state=tk.NORMAL)
            self.entry.focus_set()
            self._set_status("ready")
            return
        if stripped.startswith("[Rewritten]:"):
            content = stripped.replace("[Rewritten]:", "", 1).strip()
            self._append(f"Assistant: {content}", "assistant")
            self.awaiting_response = False
            self.entry.configure(state=tk.NORMAL)
            self.send_btn.configure(state=tk.NORMAL)
            self.entry.focus_set()
            self._set_status("ready")
            return
        self._append(stripped, "system")

    def _handle_enter(self, _event):
        self.send_message()
        return "break"

    def _handle_shift_enter(self, _event):
        self.entry.insert(tk.INSERT, "\n")
        return "break"

    def send_message(self):
        if self.process is None or self.process.poll() is not None:
            self._append("Chat process is not running.", "system")
            self._set_status("closed")
            return
        if self.awaiting_response:
            return

        text = self.entry.get("1.0", tk.END).strip()
        if not text:
            return

        self._append(f"You: {text}", "user")
        self.entry.delete("1.0", tk.END)
        self.awaiting_response = True
        self.entry.configure(state=tk.DISABLED)
        self.send_btn.configure(state=tk.DISABLED)
        self._set_status("responding")

        try:
            self.process.stdin.write("<<GUI_BLOCK_START>>\n")
            self.process.stdin.write(text + "\n")
            self.process.stdin.write("<<GUI_BLOCK_END>>\n")
            self.process.stdin.flush()
        except Exception as exc:
            self._append(f"Failed to send message: {exc}", "system")
            self.awaiting_response = False
            self.entry.configure(state=tk.NORMAL)
            self.send_btn.configure(state=tk.NORMAL)
            self.entry.focus_set()
            self._set_status("error")

    def on_close(self):
        try:
            if self.process and self.process.poll() is None:
                self.process.stdin.write("quit\n")
                self.process.stdin.flush()
                self.process.terminate()
        except Exception:
            pass
        self.parent.chat_window = None
        self.win.destroy()


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("clam")
    app = TrainerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
