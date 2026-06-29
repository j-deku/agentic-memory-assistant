# finetune.py  —  Step 3: LoRA Fine-tuning with Unsloth
# ============================================================
# Run this on Google Colab (free T4 GPU)
#
# Setup (run these in a Colab cell BEFORE this script):
#   !pip install unsloth
#   !pip install --upgrade transformers datasets trl
#
# Then upload training_data.jsonl to Colab and run this script.
#
# Expected time  : 30–60 min on free T4
# Expected result: lora-adapter/ folder + merged-model/ folder
# ============================================================

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ---------------------------------------------------------------------------
# CONFIG  — tweak these if needed
# ---------------------------------------------------------------------------

BASE_MODEL      = "unsloth/Qwen2.5-1.5B-Instruct"   # free, fits on T4
TRAINING_FILE   = "training_data.jsonl"
OUTPUT_ADAPTER  = "lora-adapter"
OUTPUT_MERGED   = "merged-model"

MAX_SEQ_LENGTH  = 512     # tool calls are short — 512 is plenty
LORA_RANK       = 16      # higher = more capacity, more VRAM; 16 is the sweet spot
LORA_ALPHA      = 32      # typically 2× rank
LORA_DROPOUT    = 0.05
EPOCHS          = 3       # 3 epochs on 2000 examples = ~6000 gradient steps
BATCH_SIZE      = 4       # per-device batch size; gradient accumulation handles the rest
GRAD_ACCUM      = 4       # effective batch = BATCH_SIZE × GRAD_ACCUM = 16
LEARNING_RATE   = 2e-4
WARMUP_RATIO    = 0.05
SAVE_STEPS      = 100
LOGGING_STEPS   = 10

# ---------------------------------------------------------------------------
# 1. LOAD BASE MODEL WITH UNSLOTH
# ---------------------------------------------------------------------------

print("Loading base model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = BASE_MODEL,
    max_seq_length  = MAX_SEQ_LENGTH,
    dtype           = None,        # auto-detect: float16 on T4, bfloat16 on A100
    load_in_4bit    = True,        # QLoRA — fits on free T4 (16GB VRAM)
)

# Apply Qwen2.5 chat template so tokens match what app.py expects
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

print(f"Base model loaded: {BASE_MODEL}")
print(f"Model parameters : {model.num_parameters():,}")

# ---------------------------------------------------------------------------
# 2. ATTACH LORA ADAPTER
# ---------------------------------------------------------------------------

print("\nAttaching LoRA adapter...")

model = FastLanguageModel.get_peft_model(
    model,
    r                   = LORA_RANK,
    lora_alpha          = LORA_ALPHA,
    lora_dropout        = LORA_DROPOUT,
    target_modules      = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # FFN
    ],
    bias                = "none",
    use_gradient_checkpointing = "unsloth",   # saves VRAM on long sequences
    random_state        = 42,
    use_rslora          = False,
    loftq_config        = None,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params : {trainable:,} ({100 * trainable / total:.2f}% of total)")

# ---------------------------------------------------------------------------
# 3. LOAD AND FORMAT TRAINING DATA
# ---------------------------------------------------------------------------

print(f"\nLoading training data from {TRAINING_FILE}...")

if not Path(TRAINING_FILE).exists():
    raise FileNotFoundError(
        f"{TRAINING_FILE} not found.\n"
        "Upload it to Colab first, or run generate_data.py locally then upload."
    )

raw_examples = []
with open(TRAINING_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            raw_examples.append(json.loads(line))

print(f"Loaded {len(raw_examples)} examples")

# Verify structure
sample = raw_examples[0]
assert "conversations" in sample, "Expected 'conversations' key in each JSONL line"
assert len(sample["conversations"]) == 3, "Expected system + user + assistant turns"

def format_example(example: dict) -> dict:
    """
    Convert ChatML conversation dict to a single formatted string
    using the Qwen2.5 chat template.

    Input:
      {"conversations": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
      ]}

    Output:
      {"text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n..."}
    """
    conversations = example["conversations"]
    text = tokenizer.apply_chat_template(
        conversations,
        tokenize          = False,
        add_generation_prompt = False,
    )
    return {"text": text}


print("Formatting examples with chat template...")
formatted = [format_example(ex) for ex in raw_examples]
dataset   = Dataset.from_list(formatted)

# 90/10 train/eval split
split     = dataset.train_test_split(test_size=0.1, seed=42)
train_ds  = split["train"]
eval_ds   = split["test"]

print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")
print(f"\nSample formatted text (first 300 chars):")
print(train_ds[0]["text"][:300])

# ---------------------------------------------------------------------------
# 4. TRAINING
# ---------------------------------------------------------------------------

print("\nStarting training...")

trainer = SFTTrainer(
    model      = model,
    tokenizer  = tokenizer,
    train_dataset = train_ds,
    eval_dataset  = eval_ds,
    args = SFTConfig(
        dataset_text_field      = "text",
        max_seq_length          = MAX_SEQ_LENGTH,
        per_device_train_batch_size  = BATCH_SIZE,
        per_device_eval_batch_size   = BATCH_SIZE,
        gradient_accumulation_steps  = GRAD_ACCUM,
        num_train_epochs        = EPOCHS,
        learning_rate           = LEARNING_RATE,
        warmup_ratio            = WARMUP_RATIO,
        lr_scheduler_type       = "cosine",
        fp16                    = not torch.cuda.is_bf16_supported(),
        bf16                    = torch.cuda.is_bf16_supported(),
        logging_steps           = LOGGING_STEPS,
        save_steps              = SAVE_STEPS,
        eval_strategy           = "steps",
        eval_steps              = SAVE_STEPS,
        save_total_limit        = 2,
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        output_dir              = OUTPUT_ADAPTER,
        report_to               = "none",   # set to "wandb" if you use W&B
        seed                    = 42,
        packing                 = False,    # packing off: tool calls need clean boundaries
    ),
)

# Show GPU memory before training
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {gpu_stats.name} | Total: {max_memory}GB | Reserved: {start_gpu_memory}GB")

trainer_stats = trainer.train()

# Training summary
print("\nTraining complete.")
print(f"  Runtime        : {trainer_stats.metrics['train_runtime']:.0f}s "
      f"({trainer_stats.metrics['train_runtime']/60:.1f} min)")
print(f"  Samples/sec    : {trainer_stats.metrics['train_samples_per_second']:.1f}")
print(f"  Final loss     : {trainer_stats.metrics['train_loss']:.4f}")

if torch.cuda.is_available():
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"  Peak VRAM used : {used_memory}GB / {max_memory}GB")

# ---------------------------------------------------------------------------
# 5. SAVE LORA ADAPTER
# ---------------------------------------------------------------------------

print(f"\nSaving LoRA adapter to {OUTPUT_ADAPTER}/...")
model.save_pretrained(OUTPUT_ADAPTER)
tokenizer.save_pretrained(OUTPUT_ADAPTER)
print("Adapter saved.")

# ---------------------------------------------------------------------------
# 6. QUICK INFERENCE TEST  (before merging)
# ---------------------------------------------------------------------------

print("\nRunning inference test on 5 examples...")

FastLanguageModel.for_inference(model)

test_inputs = [
    "Remind me to drink water in 2 hours",
    "What reminders do I have?",
    "Search for the weather in Accra",
    "Set a reminder to call mum every Sunday at 6pm",
    "Don't let me forget to take my medication at 9pm tonight",
]

SYSTEM_MSG = (
    "You are a helpful assistant with access to these tools: "
    "set_reminder, get_upcoming_reminders, open_and_search. "
    "When the user wants to set a reminder, check reminders, or search the web, "
    'respond with a tool call in JSON format: {"name": "<tool_name>", "arguments": {<args>}}'
)

all_passed = True
for user_input in test_inputs:
    messages = [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": user_input},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize              = True,
        add_generation_prompt = True,
        return_tensors        = "pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens = 128,
            temperature    = 0.1,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )

    # Decode only the new tokens (not the prompt)
    new_tokens = output_ids[0][input_ids.shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Basic validation — response should be valid JSON with a "name" key
    try:
        parsed = json.loads(response)
        tool_name = parsed.get("name", "")
        valid = tool_name in {"set_reminder", "get_upcoming_reminders", "open_and_search"}
        status = "PASS" if valid else "FAIL"
        if not valid:
            all_passed = False
    except json.JSONDecodeError:
        status = "FAIL (not valid JSON)"
        all_passed = False

    print(f"  [{status}] '{user_input[:45]}...' -> {response[:80]}")

if all_passed:
    print("\nAll inference tests passed.")
else:
    print("\nSome tests failed — consider training for more epochs or checking your data.")

# ---------------------------------------------------------------------------
# 7. MERGE LORA INTO BASE MODEL + SAVE
# ---------------------------------------------------------------------------

print(f"\nMerging LoRA into base model -> {OUTPUT_MERGED}/...")
print("This takes ~5 minutes and requires extra RAM — normal on Colab.")

model.save_pretrained_merged(
    OUTPUT_MERGED,
    tokenizer,
    save_method = "merged_16bit",   # full precision — needed for GGUF conversion
)

print(f"Merged model saved to {OUTPUT_MERGED}/")

# ---------------------------------------------------------------------------
# 8. EXPORT TO GGUF  (optional — can also run export.py separately)
# ---------------------------------------------------------------------------

print("\nExporting to GGUF (Q4_K_M quantization)...")
print("This is the format Ollama loads.")

model.save_pretrained_gguf(
    "tool-assistant-gguf",
    tokenizer,
    quantization_method = "q4_k_m",  # best balance of size vs quality for tool calling
)

print("\nGGUF export complete.")
print("\n" + "="*60)
print("PIPELINE SUMMARY")
print("="*60)
print(f"  Adapter saved  : {OUTPUT_ADAPTER}/")
print(f"  Merged model   : {OUTPUT_MERGED}/")
print(f"  GGUF model     : tool-assistant-gguf/")
print()
print("NEXT STEPS:")
print("  1. Download tool-assistant-gguf/ to your local machine")
print("  2. Run: ollama create tool-assistant -f Modelfile")
print("  3. In app.py change:")
print('       _llm = ChatOllama(model="qwen2.5:1.5b", ...)')
print("     to:")
print('       _llm = ChatOllama(model="tool-assistant", ...)')
print("="*60)