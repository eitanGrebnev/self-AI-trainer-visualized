import os
import re
import random
import gc
from pathlib import Path

# Disable torch.compile for Windows + Unsloth stability.
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

def dummy_compile(model=None, *args, **kwargs):
    if model is None:
        return lambda fn: fn
    if callable(model):
        return model
    return model

torch.compile = dummy_compile

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import docx
from pypdf import PdfReader

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STYLE_FILES_FOLDER = str(PROJECT_ROOT / "style material")
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
OUTPUT_DIR = str(PROJECT_ROOT / "my_style_model")
MAX_SEQ_LENGTH = 2048
MAX_STEPS = 55

SYSTEM_PROMPT = (
    "You rewrite text in the user's personal writing voice. "
    "Keep the meaning, facts, and structure accurate while changing tone and phrasing. "
    "Sound natural, human, and conversational. "
    "Do not add new facts, citations, or extra claims."
)


def validate_path(path):
    path = Path(path)
    if path.exists():
        return str(path)

    local_path = os.path.join(os.getcwd(), str(path))
    if os.path.exists(local_path):
        return local_path

    print(f"Could not find folder: {path}")
    print(f"Working directory: {os.getcwd()}")
    return None


def extract_text_from_files(folder_path):
    valid_path = validate_path(folder_path)
    if not valid_path:
        return []

    all_texts = []
    for root, _, files in os.walk(valid_path):
        for file in files:
            file_path = os.path.join(root, file)
            text = ""
            try:
                if file.endswith(".docx"):
                    doc = docx.Document(file_path)
                    text = "\n".join([p.text for p in doc.paragraphs])
                elif file.endswith(".pdf"):
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                elif file.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
            except Exception as e:
                print(f"Skipping {file}: {e}")
                continue

            if len(text.strip()) > 120:
                all_texts.append({"text": text, "source": file})

    return all_texts


def _clean_text(text):
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _chunk_text(text, max_chars=650):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

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
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i:i + max_chars])
                current = ""

    if current:
        chunks.append(current)

    return chunks


def to_plain_version(text):
    """Create a flatter baseline wording so the model can learn 'plain -> my voice'."""
    t = text
    t = re.sub(r"\b(can't)\b", "cannot", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(won't)\b", "will not", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(i'm)\b", "I am", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(don't)\b", "do not", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(it's)\b", "it is", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_style_transfer_examples(raw_data, tokenizer):
    examples = []

    for item in raw_data:
        cleaned = _clean_text(item["text"])
        if len(cleaned) < 180:
            continue

        chunks = _chunk_text(cleaned)
        if len(chunks) < 2:
            continue

        for idx, target_chunk in enumerate(chunks):
            style_chunk = chunks[(idx + 1) % len(chunks)]
            plain_text = to_plain_version(target_chunk)

            user_prompt = (
                "Rewrite this text so it sounds like my writing voice.\n"
                "Keep the meaning exactly the same and keep key assignment details.\n\n"
                f"Voice sample:\n{style_chunk}\n\n"
                f"Text to rewrite:\n{plain_text}"
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": target_chunk},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            examples.append({"text": text})

    random.shuffle(examples)
    return examples


print("Clearing GPU memory...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

raw_data = extract_text_from_files(STYLE_FILES_FOLDER)
if len(raw_data) == 0:
    print("No style files found. Add .txt/.docx/.pdf files to 'style material' and retry.")
    raise SystemExit(1)

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
    gpu_memory_utilization=0.7,
)

chat_data = build_style_transfer_examples(raw_data, tokenizer)
if len(chat_data) == 0:
    print("Could not build style-transfer samples. Add more personal writing data.")
    raise SystemExit(1)

print(f"Built {len(chat_data)} style-transfer training samples.")
dataset = Dataset.from_list(chat_data)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=8,
        max_steps=MAX_STEPS,
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir=OUTPUT_DIR,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        save_steps=30,
        group_by_length=True,
    ),
)

print("Starting style training...")
trainer.train()

print(f"Done. Style model saved to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
