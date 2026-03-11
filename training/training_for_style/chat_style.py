import os
from pathlib import Path
import sys

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Prevent Windows cp1252 console crashes when model emits non-ASCII characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch


def dummy_compile(model=None, *args, **kwargs):
    if model is None:
        return lambda fn: fn
    if callable(model):
        return model
    return model


torch.compile = dummy_compile

from unsloth import FastLanguageModel

BLOCK_START = "<<GUI_BLOCK_START>>"
BLOCK_END = "<<GUI_BLOCK_END>>"

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = str(PROJECT_ROOT / "my_style_model")
MAX_SEQ_LENGTH = 2048
MAX_TURNS = 3

print("Loading style rewrite model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

print("\n" + "=" * 64)
print("  STYLE REWRITE CHAT")
print("  Commands: 'quit' to exit, 'clear' to clear history")
print("  Paste AI text and it rewrites it in your voice")
print("=" * 64 + "\n")

conversation_history = []
voice_sample = os.environ.get("STYLE_VOICE_SAMPLE", "").strip()


def read_user_input(prompt):
    # Supports both normal single-line terminal input and GUI multiline block input.
    print(prompt, end="", flush=True)
    raw = sys.stdin.readline()
    if raw == "":
        return None
    first = raw.rstrip("\n")
    if first != BLOCK_START:
        return first.strip()

    lines = []
    while True:
        line = sys.stdin.readline()
        if line == "":
            break
        line = line.rstrip("\n")
        if line == BLOCK_END:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def trim_history(history, max_turns=MAX_TURNS):
    return history[-(max_turns * 2):]


def sanitize_rewrite_output(response, original_text):
    # Remove common instruction leakage lines and keep content-only output.
    cleaned_lines = []
    for line in response.splitlines():
        s = line.strip()
        if not s:
            continue
        lower = s.lower()
        if lower.startswith("return only"):
            continue
        if lower.startswith("text to rewrite"):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned:
        return original_text
    return cleaned


while True:
    if not voice_sample:
        voice_sample = read_user_input("[Voice Sample] Paste a short sample of your writing once: ")
        if voice_sample is None:
            print("Goodbye!")
            break
        if not voice_sample:
            continue

    user_input = read_user_input("\n[Text To Rewrite]: ")

    if user_input is None:
        print("Goodbye!")
        break

    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    if user_input.lower() == "clear":
        conversation_history = []
        print("(Rewrite history cleared)")
        continue

    system_prompt = (
        "You are a rewriting assistant for one student. "
        "Rewrite text to sound like the student's voice sample. "
        "Keep the original meaning and facts exactly. "
        "Do not add new claims or remove key assignment details. "
        "Do not inject new emotions, complaints, or personal confessions not present in the input. "
        "Do not add profanity unless it is already in the source text. "
        "Return only the rewritten text."
    )

    user_prompt = (
        "Rewrite the text below in my writing style.\n\n"
        f"My voice sample:\n{voice_sample}\n\n"
        f"Text to rewrite:\n{user_input}\n\n"
        "Return only the rewritten text."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(trim_history(conversation_history))
    messages.append({"role": "user", "content": user_prompt})

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"### Instruction:\n{user_prompt}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=420,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.18,
            no_repeat_ngram_size=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    response = sanitize_rewrite_output(response, user_input)

    print(f"\n[Rewritten]: {response}")

    conversation_history.append({"role": "user", "content": user_prompt})
    conversation_history.append({"role": "assistant", "content": response})
