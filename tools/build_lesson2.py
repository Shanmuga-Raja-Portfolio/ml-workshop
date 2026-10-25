"""Builds lesson2 notebook: tokenization + tiny GPT trained from scratch."""
import json
from pathlib import Path

MD = "markdown"
CODE = "code"

cells = []

def md(src): cells.append({"cell_type": MD, "id": f"md{len(cells)}", "metadata": {}, "source": src.splitlines(keepends=True)})
def code(src): cells.append({"cell_type": CODE, "id": f"code{len(cells)}", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)})

md("""\
# Lesson 2 — Text becomes numbers: build a tiny GPT from scratch

In Lesson 1 the network ate numbers and predicted numbers. Language models eat **text**. Today:

1. Build a **tokenizer** by hand (characters) and meet a real one (subwords)
2. Hand-roll the two ideas inside every GPT: **embeddings** and **causal self-attention**
3. Train a ~2.7M-parameter transformer **from random weights** to write fake Shakespeare

Same five-line training loop from Lesson 1 — nothing new there, on purpose.

**Time:** ~5 minutes on GPU. On CPU the notebook automatically shortens the training run (see section 4).""")

code("""\
import json
from pathlib import Path
import torch

def load_config():
    for p in [Path("workshop_config.json"), Path("../../workshop_config.json"), Path("../../../workshop_config.json")]:
        if p.exists():
            return json.loads(p.read_text())
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {"device": "cuda", "dtype": "bfloat16", "vram_gb": vram,
                "tier": "gpu-high" if vram >= 12 else "gpu-low"}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"device": "mps", "dtype": "float16", "vram_gb": None, "tier": "gpu-low"}
    return {"device": "cpu", "dtype": "float32", "vram_gb": None, "tier": "cpu"}

CFG = load_config()
DEVICE, DTYPE, TIER = CFG["device"], getattr(torch, CFG["dtype"]), CFG["tier"]
print(f"Workshop tier: {TIER} | device: {DEVICE} | dtype: {DTYPE}")

device = DEVICE  # this lesson uses lowercase `device` throughout
if TIER == "cpu":
    print("CPU tier: the training run is auto-shortened in section 4 - the concepts are identical.")""")

code("""\
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt""")

md("""\
## 1. A tokenizer, by hand

Computers only do numbers, so text must be converted first. The simplest scheme: every distinct **character** gets an ID. The set of IDs is the *vocabulary*.""")

code("""\
p = Path("data/tinyshakespeare.txt")
if not p.exists():
    p = Path("lessons/01_foundations/data/tinyshakespeare.txt")
text = p.read_text(encoding="utf-8")

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}          # string -> int
itos = {i: c for c, i in stoi.items()}              # int -> string
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

print(f"corpus: {len(text):,} characters | vocabulary: {vocab_size} characters")
ids = encode("To be, or not to be:")
print("encode('To be, or not to be:') ->", ids)
print("decode(ids) ->", repr(decode(ids)))""")

md("""\
Character-level is easy but wasteful: common words become long ID sequences. Real models use **subword** tokenizers (BPE): frequent chunks like `ing` or `the` get single IDs. Same idea, smarter vocabulary.""")

code("""\
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    s = "Tokenization is surprisingly interesting!"
    ids = tok(s)["input_ids"]
    print(f"Qwen3-1.7B vocabulary: {tok.vocab_size:,} entries (vs our {vocab_size})")
    print("pieces:", [tok.decode([i]) for i in ids])
    print("ids:   ", ids)
except Exception as e:
    print("modern-tokenizer demo skipped (offline?):", e)""")

md("""\
## 2. Training data: predict the next character

Each training example is a chunk of 256 characters. Input `x` = characters 0..255; target `y` = characters 1..256 — the same text **shifted by one**. The model sees everything up to position *t* and must guess character *t+1*. That's it. That is what GPT does.""")

code("""\
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data, val_data = data[:n], data[n:]

block_size = 256
batch_size = 32 if TIER == "gpu-low" else 64   # smaller cards get a lighter batch

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size - 1, (batch_size,))
    x = torch.stack([src[i : i + block_size] for i in ix])
    y = torch.stack([src[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

xb, yb = get_batch("train")
print(f"batch: {tuple(xb.shape)} inputs + {tuple(yb.shape)} targets")
print("x row 0 starts:", repr(decode(xb[0, :40].tolist())))
print("y row 0 starts:", repr(decode(yb[0, :40].tolist())), " <- same text, shifted 1")""")

md("""\
## 3. The model

Three new concepts, all small:

- **Token embedding**: a lookup table — character ID #46 becomes a 192-number vector. Position embeddings do the same for "where in the chunk am I".
- **Causal self-attention** (`Head` below): each position builds a *query* ("what am I looking for?") and *key* ("what do I contain?"). Queries dot with keys → attention scores; the causal mask forbids looking at future characters, which is why it can only *predict the next* thing.
- **The Block**: attention (mix information across positions) → MLP (think about each position), wrapped in residual connections. Stack 6 of them.""")

code("""\
class Head(nn.Module):
    \"\"\"One head of causal self-attention.\"\"\"
    def __init__(self, d_model, d_head):
        super().__init__()
        self.q = nn.Linear(d_model, d_head, bias=False)
        self.k = nn.Linear(d_model, d_head, bias=False)
        self.v = nn.Linear(d_model, d_head, bias=False)
        self.d_head = d_head

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = q @ k.transpose(-2, -1) / self.d_head ** 0.5      # (B,T,T) affinity
        mask = torch.tril(torch.ones(T, T, device=x.device))       # look left only
        scores = scores.masked_fill(mask == 0, float("-inf"))
        return scores.softmax(dim=-1) @ v                          # (B,T,d_head)

class MultiHead(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_model, d_model // n_head) for _ in range(n_head)])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHead(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))        # residual: add, don't replace
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_head=6, n_layer=6, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying: embeddings in reverse
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)  # small init = sane starting logits

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        return self.head(self.ln_f(x))                  # (B,T,vocab) raw scores per position

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=500, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size:])    # crop context to block_size
            probs = (logits[:, -1, :] / temperature).softmax(dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        self.train()
        return idx""")

code("""\
torch.manual_seed(1)
model = TinyGPT(vocab_size).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params:,}   (GPT-4 class models: ~1,000,000x more)")
print("sample BEFORE training:")
print(decode(model.generate(torch.tensor([[stoi['\\n']]], device=device), 300)[0].tolist()))""")

md("""\
Random noise, as expected — the model is just guessing from its random starting weights. (For reference: GPT-class models are ~1,000,000x bigger than this one. The *architecture* is the same shape.)

## 4. Train it

The same five lines as Lesson 1. The only new ingredient: `cross_entropy` loss — the standard loss for "pick the right next token out of 65 options".""",)

code("""\
STEPS = 4000 if TIER != "cpu" else 500
if TIER == "cpu":
    print(f"CPU tier: running {STEPS} steps instead of 4000 (~10-15 min). "
          "Loss won't fall as far - the shape of the curve is the lesson.")
losses = []
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-4)

for step in range(STEPS + 1):
    x, y = get_batch("train")                                  # 1. data
    logits = model(x)                                          # 2. forward
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    opt.zero_grad()                                            # 3. clear
    loss.backward()                                            # 4. backprop
    opt.step()                                                 # 5. nudge
    sched.step()
    losses.append(loss.item())
    if step % 500 == 0:
        print(f"step {step:5d}   loss {loss.item():.3f}")

plt.figure(figsize=(8, 3))
plt.plot(losses, lw=0.4)
plt.xlabel("step"); plt.ylabel("loss (how surprised the model is)")
plt.title("training loss"); plt.show()""")

md("""\
**You should see:** loss fall from ~4.2 (random guessing over 65 chars) to roughly 1.5-1.7, with the curve flattening. Loss = average surprise per character; 1.6 means "I'm rarely shocked by the next character." On CPU (500 steps) expect the same start and a landing around 2.3-2.8 instead — same curve, fewer steps.

## 5. Listen to it talk""")

code("""\
@torch.no_grad()
def split_loss(split):
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(20):
        x, y = get_batch(split)
        total += F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1)).item()
    if was_training: model.train()
    return total / 20

print(f"train loss: {split_loss('train'):.3f}   val loss: {split_loss('val'):.3f}")
print("(close together = memorizing the *style*, not the exact text - what we want)\\n")

print("sample AFTER training:")
print(decode(model.generate(torch.tensor([[stoi['\\n']]], device=device), 500, temperature=0.8)[0].tolist()))""")

md("""\
**You should see:** fake Shakespeare — mostly real words, speaker names with colons, line breaks in plausible places, occasional nonsense. From a model that has never seen a grammar rule, only 1.1MB of text and gradient descent. (On the 500-step CPU run expect more babble than Shakespeare — common words and plausible punctuation, fewer real words. Re-run on a GPU, or re-run the training cell a few more times, and it sharpens up.)

## Exercises

Edit and re-run. Each teaches something real:""")

code("""\
# 1. TEMPERATURE: re-run the sampling cell with temperature=0.3 (safe, repetitive)
#    and temperature=1.5 (chaotic). temperature divides the logits before softmax:
#    low = sharpen the distribution, high = flatten it.
#
# 2. PROMPTING: start generation from a real prefix instead of '\\n':
#      start = torch.tensor([encode("ROMEO:")], device=device)
#    You just did 'conditional generation' - the trick behind ChatGPT's answer box.
#
# 3. CAPACITY: rebuild with n_layer=8, d_model=256 (re-run model + training cells).
#    Does val loss improve? At what point does it just memorize (train << val)?
#
# 4. LONGER: train 3000 MORE steps by re-running the training cell twice.
#    Diminishing returns kick in - welcome to scaling.
#
# 5. THE BIG ONE: our vocab is 65 chars. Swap in the Qwen tokenizer from section 1
#    (encode its output the same way). Suddenly vocab_size ~ 151k - watch the
#    embedding table dominate the parameter count. This is why LLMs are big.

print("Your turn. Then compare notes with a friend - or your future self.")""")

md("""\
## Where you are now

You have trained, from nothing: a regression net (Lesson 1) and a transformer language model (Lesson 2). Both times you wrote the loop yourself.

**Next - Lesson 3:** nobody trains big models from nothing. We take a pretrained DistilBERT (66M params, already knows English) and fine-tune it into a sentiment classifier in minutes — the standard Hugging Face workflow you'll use in the wild.""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ml-training)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("lessons/01_foundations/lesson2_tiny_gpt_from_scratch.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "with", len(cells), "cells")
