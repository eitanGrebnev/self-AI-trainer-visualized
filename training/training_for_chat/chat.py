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

# Patch torch.compile
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

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = str(PROJECT_ROOT / "my_custom_model")
MAX_SEQ_LENGTH = 2048
MAX_TURNS = 4

print("Loading model... (this takes a minute)")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_DIR,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

print("\n" + "="*60)
print("  CHAT WITH YOUR FINE-TUNED LLAMA-3 MODEL")
print("  Type 'quit' or 'exit' to stop")
print("  Type 'clear' to reset conversation")
print("="*60 + "\n")

conversation_history = []


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


def trimmed_history(history, max_turns=MAX_TURNS):
    # Keep only recent turns so the model does not drift from the user's latest question.
    keep_messages = max_turns * 2
    return history[-keep_messages:]

while True:
    user_input = read_user_input("\n[You]: ")

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
        print("(Conversation cleared)")
        continue

    messages = [
        {
            "role": "system",
            "content": (
                "You are a supportive school assistant with a warm conversational voice. "
                "Use clear, student-friendly language with a natural spoken style. "
                "Stay on-topic and answer the user's exact question using their assignment context. "
                "If unsure, say what is missing instead of making things up."
            ),
        }
    ]
    messages.extend(trimmed_history(conversation_history))
    messages.append({"role": "user", "content": user_input})

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 320,
            temperature = 0.45,
            top_p = 0.85,
            repetition_penalty = 1.2,
            no_repeat_ngram_size = 4,
            do_sample = True,
            use_cache = True,
            pad_token_id = tokenizer.eos_token_id,
            eos_token_id = tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    print(f"\n[Model]: {response}")

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": response})
