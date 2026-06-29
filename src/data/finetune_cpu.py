"""
finetune_cpu.py  —  CPU-compatible LoRA fine-tuning
                    No Unsloth required — pure HuggingFace + PEFT

Works on:
  - Lightning.ai free CPU studio
  - Your local machine
  - Any CPU-only environment

Expected time: 4–8 hours on CPU (run overnight)
Expected result: lora-adapter/ folder ready to merge + export

Usage:
  pip install transformers peft trl datasets accelerate
  python finetune_cpu.py
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
TRAINING_FILE  = "training_data.jsonl"
OUTPUT_ADAPTER = "lora-adapter"
OUTPUT_MERGED  = "merged-model"

MAX_SEQ_LENGTH = 512
LORA_RANK      = 8      # lower rank = less memory on CPU
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.05
EPOCHS         = 3
BATCH_SIZE     = 1      # CPU: batch size 1 to avoid OOM
GRAD_ACCUM     = 16     # effective batch = 1 × 16 = 16
LEARNING_RATE  = 2e-4
WARMUP_RATIO   = 0.05
SAVE_STEPS     = 200
LOGGING_STEPS  = 50

# ---------------------------------------------------------------------------
# 1. LOAD TOKENIZER + MODEL
# ---------------------------------------------------------------------------

print(f"Loading tokenizer: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading model: {BASE_MODEL}")
print("This may take a few minutes on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype  = torch.float32,   # float32 for CPU — no float16 support
    trust_remote_code = True,
    device_map   = "cpu",
)
model.config.use_cache = False      # disable KV cache during training
model.enable_input_require_grads()

print(f"Model loaded. Parameters: {model.num_parameters():,}")

# ---------------------------------------------------------------------------
# 2. ATTACH LORA
# ---------------------------------------------------------------------------

print("\nAttaching LoRA adapter...")

lora_config = LoraConfig(
    r              = LORA_RANK,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# 3. LOAD + FORMAT TRAINING DATA
# ---------------------------------------------------------------------------

print(f"\nLoading training data from {TRAINING_FILE}...")

if not Path(TRAINING_FILE).exists():
    raise FileNotFoundError(f"{TRAINING_FILE} not found. Upload it first.")

raw_examples = []
with open(TRAINING_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            raw_examples.append(json.loads(line))

print(f"Loaded {len(raw_examples)} examples")

def format_example(example: dict) -> dict:
    conversations = example["conversations"]
    parts = []
    for turn in conversations:
        role    = turn["role"]
        content = turn["content"]
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    return {"text": "\n".join(parts)}

print("Formatting examples...")
formatted = [format_example(ex) for ex in raw_examples]
dataset   = Dataset.from_list(formatted)
split     = dataset.train_test_split(test_size=0.1, seed=42)

print(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")
print(f"\nSample (first 200 chars):")
print(split["train"][0]["text"][:200])

# ---------------------------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------------------------

print("\nStarting training...")
print("On CPU this takes 4–8 hours. Run overnight and come back tomorrow.")
print("Progress is saved every 200 steps — safe to resume if interrupted.\n")
model.config.pad_token_id = tokenizer.pad_token_id
trainer = SFTTrainer(
    model         = model,
    train_dataset = split["train"],
    eval_dataset  = split["test"],
    args = SFTConfig(
        dataset_text_field          = "text",
        max_length              = MAX_SEQ_LENGTH,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        num_train_epochs            = EPOCHS,
        learning_rate               = LEARNING_RATE,
        warmup_ratio                = WARMUP_RATIO,
        lr_scheduler_type           = "cosine",
        fp16                        = False,   # CPU: no fp16
        bf16                        = False,   # CPU: no bf16
        logging_steps               = LOGGING_STEPS,
        save_steps                  = SAVE_STEPS,
        eval_strategy               = "steps",
        eval_steps                  = SAVE_STEPS,
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        output_dir                  = OUTPUT_ADAPTER,
        report_to                   = "none",
        seed                        = 42,
        packing                     = False,
    ),
)

stats = trainer.train()

print(f"\nTraining complete.")
print(f"  Time  : {stats.metrics['train_runtime']/3600:.1f} hours")
print(f"  Loss  : {stats.metrics['train_loss']:.4f}")

# ---------------------------------------------------------------------------
# 5. SAVE LORA ADAPTER
# ---------------------------------------------------------------------------

print(f"\nSaving LoRA adapter to {OUTPUT_ADAPTER}/...")
model.save_pretrained(OUTPUT_ADAPTER)
tokenizer.save_pretrained(OUTPUT_ADAPTER)
print("Adapter saved.")

# ---------------------------------------------------------------------------
# 6. MERGE LORA INTO BASE MODEL
# ---------------------------------------------------------------------------

print(f"\nMerging LoRA into base model -> {OUTPUT_MERGED}/...")
print("This takes a few minutes...")

from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype       = torch.float32,
    trust_remote_code = True,
    device_map        = "cpu",
)
merged_model = PeftModel.from_pretrained(base_model, OUTPUT_ADAPTER)
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained(OUTPUT_MERGED)
tokenizer.save_pretrained(OUTPUT_MERGED)
print(f"Merged model saved to {OUTPUT_MERGED}/")

# ---------------------------------------------------------------------------
# 7. CONVERT TO GGUF  (requires llama.cpp)
# ---------------------------------------------------------------------------

print("\nConverting to GGUF...")
print("Installing llama.cpp converter...")

os.system("pip install gguf -q")

convert_script = Path("convert_hf_to_gguf.py")
if not convert_script.exists():
    os.system(
        "wget -q https://raw.githubusercontent.com/ggerganov/llama.cpp/"
        "master/convert_hf_to_gguf.py"
    )

ret = os.system(
    f"python convert_hf_to_gguf.py {OUTPUT_MERGED} "
    f"--outfile tool-assistant.gguf --outtype q8_0"
)

if ret == 0:
    print("\nGGUF conversion complete: tool-assistant.gguf")
else:
    print("\nGGUF conversion failed — download the merged-model/ folder instead")
    print("and convert locally with llama.cpp.")

print("\n" + "="*60)
print("DONE")
print("="*60)
print(f"  Adapter : {OUTPUT_ADAPTER}/")
print(f"  Merged  : {OUTPUT_MERGED}/")
print(f"  GGUF    : tool-assistant.gguf")
print()
print("NEXT STEPS:")
print("  1. Download tool-assistant.gguf to your local machine")
print("  2. Run: ollama create tool-assistant -f Modelfile")
print('  3. In app.py change model="qwen2.5:1.5b" to model="tool-assistant"')
print("="*60)