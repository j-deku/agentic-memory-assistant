import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ===============================
# SPEED + REPRODUCIBILITY
# ===============================
random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(os.cpu_count())

# ===============================
# CONFIG
# ===============================
BATCH_SIZE = 16
BLOCK_SIZE = 128
MAX_ITERS = 50
LEARNING_RATE = 3e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_EMBD = 512
N_HEAD = 8
N_LAYER = 6
DROPOUT = 0.1

MODEL_FILE = "chat_transformer.pt"
TOKENIZER_FILE = "tokenizer.json"
DATA_FILE = "chat_data.txt"

print("Using device:", DEVICE)

# ===============================
# 1. LOCAL DATASET PIPELINE
# ===============================
def load_local_data():
    if os.path.exists(DATA_FILE):
        print("Loading local dataset...")
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    print("No dataset found → using synthetic dataset")

    base = [
        "<|user|>Hello<|assistant|>Hi!<|end|>",
        "<|user|>What is AI?<|assistant|>AI is artificial intelligence.<|end|>",
        "<|user|>Tell me a joke<|assistant|>Why did the robot cross the road?<|end|>",
        "<|user|>How are you?<|assistant|>I'm fine, thanks!<|end|>",
        "<|user|>Good morning<|assistant|>Good morning!<|end|>",
    ]

    return base * 2000


dialogs = load_local_data()
random.shuffle(dialogs)

split = int(len(dialogs) * 0.9)
train_texts = dialogs[:split]
val_texts = dialogs[split:]

print("Total samples:", len(dialogs))

# ===============================
# 2. TOKENIZER (FULL LOCAL)
# ===============================
if os.path.exists(TOKENIZER_FILE):
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

else:
    print("Training tokenizer locally...")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=8000,
        special_tokens=[
            "[UNK]", "[PAD]",
            "<|user|>", "<|assistant|>", "<|end|>"
        ]
    )

    tokenizer.train_from_iterator(dialogs, trainer)
    tokenizer.save(TOKENIZER_FILE)

VOCAB_SIZE = tokenizer.get_vocab_size()

def encode(text):
    return tokenizer.encode(text).ids

def decode(ids):
    return tokenizer.decode(ids, skip_special_tokens=False)

# Save dataset locally (optional but useful)
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        for d in dialogs:
            f.write(d + "\n")

# ===============================
# 3. TOKENIZATION
# ===============================
print("Tokenizing dataset...")

def tokenize(data):
    tokens = []
    for d in data:
        tokens.extend(encode(d))
    return tokens


train_tokens = tokenize(train_texts)
val_tokens = tokenize(val_texts)

train_data = torch.tensor(train_tokens, dtype=torch.long)
val_data = torch.tensor(val_tokens, dtype=torch.long)

print("Train tokens:", len(train_tokens))
print("Val tokens:", len(val_tokens))

# ===============================
# 4. BATCH SAMPLER
# ===============================
def get_batch(split="train"):
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])

    return x.to(DEVICE), y.to(DEVICE)

# ===============================
# 5. TRANSFORMER MODEL
# ===============================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        return self.dropout(wei) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.sa = MultiHeadAttention(N_HEAD, head_size)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ChatTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)

        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok = self.token_embedding(idx)
        pos = self.pos_embedding(torch.arange(T, device=DEVICE))

        x = tok + pos
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B*T, VOCAB_SIZE),
                targets.view(B*T)
            )

        return logits, loss

    def generate(self, idx, max_new_tokens=60, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_token), dim=1)

            if next_token.item() == tokenizer.token_to_id("<|end|>"):
                break

        return idx

# ===============================
# 6. TRAINING
# ===============================
model = ChatTransformer().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

if os.path.exists(MODEL_FILE):
    print("Loading model checkpoint...")
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))

print("\nTraining started...\n")

for step in range(1, MAX_ITERS + 1):
    xb, yb = get_batch("train")

    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 100 == 0 or step == 1:
        print(f"Step {step} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), MODEL_FILE)
print("Model saved!")

# ===============================
# 7. CHAT MODE
# ===============================
print("\n🤖 CHAT MODE READY (type 'quit')\n")

model.eval()
history = ""

while True:
    user = input("You: ")
    if user.lower() == "quit":
        break

    prompt = f"<|user|>{user}<|assistant|>"
    history += prompt

    tokens = encode(history)[-BLOCK_SIZE:]
    x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out = model.generate(x)

    out = out[0].tolist()
    new_tokens = out[len(tokens):]

    reply = decode(new_tokens).replace("<|end|>", "").strip()

    print("Bot:", reply)

    history += reply + "<|end|>"