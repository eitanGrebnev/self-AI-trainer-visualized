import os
import sys
import re
from pathlib import Path

# --- CRITICAL: Disable torch.compile BEFORE any imports ---
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Patch torch.compile to be a no-op
import torch
original_compile = torch.compile
def dummy_compile(model=None, *args, **kwargs):
    # Handle both @torch.compile and @torch.compile(dynamic=True) usage
    if model is None:
        # Called with kwargs only (decorator factory): @torch.compile(dynamic=True)
        return lambda fn: fn
    if callable(model):
        # Called as @torch.compile directly on a function
        return model
    # Called as torch.compile(model_instance)
    return model
torch.compile = dummy_compile

print("Applying Windows Compatibility Patch...")

import time
import random
import gc
from dataclasses import dataclass
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import docx
from pypdf import PdfReader

# --- VISUALIZATION IMPORTS ---
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.table import Table
    from rich.align import Align
    from rich.style import Style
except ImportError:
    print("---------------------------------------------------")
    print("ERROR: You need to install 'rich' for the visualizations!")
    print("Please run: pip install rich")
    print("---------------------------------------------------")
    exit()


# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOUR_FILES_FOLDER = str(PROJECT_ROOT / "trained material")

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
OUTPUT_DIR = str(PROJECT_ROOT / "my_custom_model")

MAX_SEQ_LENGTH = 2048  # 4080 Super can handle this with 4-bit quant + grad checkpointing
MAX_STEPS = 120
USE_DASHBOARD = True

SYSTEM_PROMPT = (
    "You are a supportive school assistant. Speak clearly in a warm, conversational tone. "
    "Answer only using the assignment/course material provided in your training. "
    "If details are missing, say you are not sure and ask a follow-up question."
)

PROMPT_TEMPLATES = [
    "Explain this assignment material in simple words and keep the important requirements:\n\n{chunk}",
    "Summarize the key instructions, deadlines, and grading expectations from this material:\n\n{chunk}",
    "Turn this class material into a student-friendly explanation with actionable next steps:\n\n{chunk}",
]


# --- PATH DETECTIVE ---
def validate_path(path):
    path = Path(path)
    if path.exists():
        return str(path)
    
    current_dir = os.getcwd()
    local_path = os.path.join(current_dir, str(path))
    if os.path.exists(local_path):
        return local_path
        
    print(f"\n[red]CRITICAL ERROR: Could not find folder:[/red] {path}")
    print(f"Current working directory: {current_dir}")
    print("\nAvailable folders here:")
    found = False
    for item in os.listdir(current_dir):
        if os.path.isdir(item):
            print(f" - {item}")
            found = True
    if not found:
        print(" - (No folders found)")
    print("\nPlease update YOUR_FILES_FOLDER in the script with one of these names.\n")
    return None


# --- VISUALIZATION CLASSES ---
class TrainingDashboard:
    def __init__(self):
        self.layout = Layout()
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=5)
        )
        self.layout["body"].split_row(
            Layout(name="stats", ratio=1),
            Layout(name="network", ratio=2)
        )
        self.loss_history = []
        self.start_time = time.time()


    def get_header(self):
        return Panel(
            Align.center(f"[bold white]LLAMA-3 TRAINING COMMAND CENTER[/bold white] | [cyan]{MODEL_NAME}[/cyan]"),
            style="on blue",
            border_style="blue"
        )


    def get_stats_panel(self, step, logs):
        table = Table(show_header=False, expand=True, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")
        
        loss = logs.get("loss", 0.0)
        lr = logs.get("learning_rate", 0.0)
        epoch = logs.get("epoch", 0.0)
        
        if loss > 0: self.loss_history.append(loss)
        avg_loss = sum(self.loss_history)/len(self.loss_history) if self.loss_history else 0
        elapsed = time.time() - self.start_time
        speed = step / elapsed if elapsed > 0 else 0


        table.add_row("Current Step", f"{step}/{MAX_STEPS}")
        table.add_row("Current Loss", f"{loss:.4f}")
        table.add_row("Average Loss", f"{avg_loss:.4f}")
        table.add_row("Learning Rate", f"{lr:.2e}")
        table.add_row("Epoch", f"{epoch:.2f}")
        table.add_row("Speed", f"{speed:.2f} it/s")
        
        return Panel(table, title="[bold]System Diagnostics[/bold]", border_style="green")


    def get_network_activity_panel(self, step):
        rows = []
        chars = ["#", "=", "+", "-"]
        
        layer_names = ["Embeddings", "Attn Layer 1", "Attn Layer 2", "Attn Layer 3", 
                       "MLP Layer 1", "MLP Layer 2", "Output Head"]
        
        rows.append(f"[bold underline]Real-time Gradient Flow Simulation:[/bold underline]\n")
        
        for layer in layer_names:
            activity = ""
            for _ in range(40):
                if random.random() > 0.78:
                    char = random.choice(chars)
                    activity += f"[green]{char}[/green]"
                else:
                    activity += "[bright_black].[/bright_black]"
            
            status = "[bold green]UPDATING[/bold green]" if random.random() > 0.7 else "[dim]WAITING[/dim] "
            rows.append(f"{layer.ljust(15)} │ {activity} │ {status}")


        return Panel("\n".join(rows), title="[bold]Neural Network Activity[/bold]", border_style="magenta")


    def get_footer(self, step):
        percent = min(100, (step / MAX_STEPS) * 100) if MAX_STEPS > 0 else 0
        bar_length = 50
        filled = int(bar_length * (percent / 100))
        bar = "█" * filled + "░" * (bar_length - filled)
        return Panel(
            Align.center(f"[bold yellow]Training Progress:[/bold yellow] [{bar}] {percent:.1f}%"),
            border_style="yellow"
        )


    def update(self, step, logs):
        self.layout["header"].update(self.get_header())
        self.layout["stats"].update(self.get_stats_panel(step, logs))
        self.layout["network"].update(self.get_network_activity_panel(step))
        self.layout["footer"].update(self.get_footer(step))
        return self.layout


class RichVisualizerCallback(TrainerCallback):
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.step = 0
        self.logs = {}
    
    def on_step_end(self, args, state, control, **kwargs):
        self.step = state.global_step


    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs: self.logs = logs


def extract_text_from_files(folder_path):
    valid_path = validate_path(folder_path)
    if not valid_path:
        return []

    all_texts = []
    print(f"Scanning {valid_path}...")
    
    file_count = 0
    
    for root, dirs, files in os.walk(valid_path):
        for file in files:
            file_path = os.path.join(root, file)
            text = ""
            try:
                if file.endswith(".docx"):
                    doc = docx.Document(file_path)
                    text = "\n".join([p.text for p in doc.paragraphs])
                    print(f" [green]Found:[/green] {file}")
                elif file.endswith(".pdf"):
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text: text += page_text + "\n"
                    print(f" [green]Found:[/green] {file}")
                elif file.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    print(f" [green]Found:[/green] {file}")

                if len(text.strip()) > 100:
                    all_texts.append({"text": text})
                    file_count += 1
            except Exception as e:
                print(f" [red]Skipping:[/red] {file} ({e})")

    return all_texts


def _clean_text(text):
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _chunk_text(text, max_chars=1400):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(para) <= max_chars:
                current = para
            else:
                # Hard split very long paragraphs to keep samples trainable.
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i:i + max_chars])
                current = ""
    if current:
        chunks.append(current)
    return chunks


def build_chat_examples(raw_data, tokenizer):
    examples = []
    for item in raw_data:
        cleaned = _clean_text(item["text"])
        if len(cleaned) < 120:
            continue
        for idx, chunk in enumerate(_chunk_text(cleaned)):
            template = PROMPT_TEMPLATES[idx % len(PROMPT_TEMPLATES)]
            user_prompt = template.format(chunk=chunk)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": chunk},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            examples.append({"text": text})
    return examples


# AGGRESSIVE MEMORY CLEANUP
print("Clearing GPU memory...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# 1. PREPARE RAW DATA
raw_data = extract_text_from_files(YOUR_FILES_FOLDER)

if len(raw_data) == 0:
    print("No usable text found! Exiting...")
    exit()

# 2. LOAD MODEL + TOKENIZER
print("Loading Model... (This might take a minute)")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,
    gpu_memory_utilization = 0.7,  # 4080 Super has 16GB, plenty for 4-bit 8B model
)

# 3. BUILD CHAT-FORMAT TRAINING SET
chat_data = build_chat_examples(raw_data, tokenizer)
if len(chat_data) == 0:
    print("No chat-format examples could be created. Exiting...")
    exit()
print(f"Built {len(chat_data)} chat-style training samples.")
dataset = Dataset.from_list(chat_data)

# 4. ADD LORA ADAPTERS
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",    
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("[green]✓ Using standard logits path (fused CE bypassed via env var)[/green]")

# 4b. SETUP VISUALIZATION
dashboard = TrainingDashboard()
visualizer_callback = RichVisualizerCallback(dashboard)

# FINAL MEMORY CLEANUP
print("Final memory cleanup...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 5. TRAIN - Chat-style supervised fine-tuning
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    tokenizer = tokenizer,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = MAX_STEPS,
        learning_rate = 5e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = OUTPUT_DIR,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        disable_tqdm = True, 
        report_to = "none",
        gradient_checkpointing = True,
        max_grad_norm = 1.0,
        save_steps = 30,
        group_by_length = True,
    ),
    callbacks=[visualizer_callback]
)

print("Initializing Dashboard...")
time.sleep(1)

try:
    if USE_DASHBOARD:
        with Live(dashboard.layout, refresh_per_second=4, screen=True) as live:
            def update_live_view():
                live.update(dashboard.update(visualizer_callback.step, visualizer_callback.logs))

            def patched_on_step_end(args, state, control, **kwargs):
                visualizer_callback.step = state.global_step
                update_live_view()

            def patched_on_log(args, state, control, logs=None, **kwargs):
                if logs:
                    visualizer_callback.logs = logs
                update_live_view()

            visualizer_callback.on_step_end = patched_on_step_end
            visualizer_callback.on_log = patched_on_log
            trainer.train()
    else:
        print("Dashboard disabled. Streaming standard trainer logs...")
        trainer.train()
except Exception as e:
    print(f"\n[red]Training error: {e}[/red]")
    print("\nTry reinstalling packages:")
    print("pip uninstall -y triton torch")
    print("pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121")
    print("pip install --no-deps unsloth")
    raise

# 6. SAVE
print(f"\nDone! Model saved to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
