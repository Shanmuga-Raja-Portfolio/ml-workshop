"""Builds lesson3 notebook: fine-tune DistilBERT on SST-2 with the HF Trainer."""
import json
from pathlib import Path

cells = []
def md(src): cells.append({"cell_type": "markdown", "id": f"md{len(cells)}", "metadata": {}, "source": src.splitlines(keepends=True)})
def code(src): cells.append({"cell_type": "code", "id": f"code{len(cells)}", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)})

md("""\
# Lesson 3 — Fine-tune your first real model

Lessons 1-2: you trained small nets **from random weights**. Almost nobody does that beyond toy scale. The real-world move:

1. Take a model already **pretrained** on billions of words (it knows grammar, facts, style)
2. Swap its head for your task (here: "positive or negative?")
3. **Fine-tune**: nudge ALL its weights with your labeled data — same 5-line loop, the `Trainer` just runs it for you

Today's model: **DistilBERT** (66M params). Task: movie-review sentiment (SST-2). Then we deliberately **break it with overfitting** so you can recognize the disease forever.

**Time:** ~5 minutes on GPU. On CPU the notebook automatically uses a 2,000-review training subset, keeping the run in the ~15-30 minute range.""")

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
# repo root, so run outputs land in <repo>/runs no matter where Jupyter started
ROOT = next((p.parent for p in [Path("workshop_config.json"), Path("../../workshop_config.json"),
                                Path("../../../workshop_config.json")] if p.exists()),
            Path("../..")).resolve()""")

code("""\
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score
import numpy as np

MODEL = "distilbert/distilbert-base-uncased"
raw = load_dataset("stanfordnlp/sst2")["train"].train_test_split(test_size=1000, seed=42)
#                 ^ 67k labeled movie-review sentences -> 66k train / 1k held-out validation
if TIER == "cpu":
    raw["train"] = raw["train"].shuffle(seed=42).select(range(2000))
    raw["test"] = raw["test"].select(range(500))
    print("CPU tier: training on a 2,000-review subset, evaluating on 500 - "
          "same workflow, smaller slice, so the run stays in the minutes range.")
print(raw)
print("\\nexample:", raw["train"][0])""")

md("""\
`label` 0 = negative, 1 = positive. We keep 1,000 reviews aside — **never train on data you evaluate on**, or you're grading your own homework.

## 1. Tokenize

DistilBERT's own tokenizer (WordPiece subwords — Lesson 2's smarter cousin). `map` applies it to every row in parallel.""")

code("""\
tok = AutoTokenizer.from_pretrained(MODEL)

def tokenize(batch):
    return tok(batch["sentence"], truncation=True)     # padding handled dynamically per-batch

ds = raw.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
collator = DataCollatorWithPadding(tok)
print(ds["train"][0])""")

md("""\
## 2. The model — pretrained body, fresh head

`AutoModelForSequenceClassification` loads the pretrained weights and bolts on a **new random** 2-class head. The body already knows English; the head knows nothing yet.""")

code("""\
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
n_all = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{n_all:,} parameters total | {n_trainable:,} trainable")
print("(compare: Lesson 2's from-scratch GPT was 2.7M. Fine-tuning moves 66M - and still fits easily.)")""")

md("""\
## 3. Train with the `Trainer`

The Trainer IS the Lesson-1 loop, industrialized: it handles batching, logging, evaluation, checkpointing. You configure it with `TrainingArguments` and give it a metrics function.""")

code("""\
def metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": accuracy_score(labels, logits.argmax(axis=-1))}

args = TrainingArguments(
    output_dir=str(ROOT / "runs" / "lesson3"),
    num_train_epochs=2,
    per_device_train_batch_size=16 if TIER == "gpu-low" else 32,
    per_device_eval_batch_size=32 if TIER == "gpu-low" else 64,
    learning_rate=5e-5,
    eval_strategy="epoch",
    logging_strategy="epoch",
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=collator,
    compute_metrics=metrics,
)

trainer.train()
print("\\nfinal:", trainer.evaluate())""")

md("""\
**You should see:** accuracy around **90-95%** after 2 epochs (on the 2,000-review CPU subset, ~85-90% — less data, same shape). DistilBERT reached in minutes what a from-scratch model couldn't reach with the whole dataset — that's the power of the pretrained start. (SOTA on this benchmark is ~97%; the remaining gap is mostly ambiguous reviews even humans argue about.)

## 4. Use it""")

code("""\
model.eval()

def predict(sentences):
    batch = tok(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = model(**batch).logits.softmax(dim=-1)
    for s, p in zip(sentences, probs):
        verdict = "POSITIVE" if p[1] > 0.5 else "NEGATIVE"
        print(f"{p[1]:.3f}  {verdict:8s}  {s}")

predict([
    "An absolute masterpiece, I cried twice.",
    "Two hours of my life I will never get back.",
    "The plot was thin, but the acting saved it.",
    "mediocre at best",
])""")

md("""\
**Look closely at that output** — one of the four is probably wrong. "Two hours of my life I will never get back" is how humans say *terrible movie*, but there's no negative word in it — so the model reads it positive. Indirect meaning, sarcasm, and "thin plot but great acting" style trade-offs are where 66M-param classifiers faceplant. Keep this failure in mind: it's the gap that chat-scale LLMs (Lesson 4) were built to close.

## 5. Now break it — the overfitting demo

Overfitting = the model **memorizes** training examples instead of learning the pattern. Symptom: training loss keeps falling while held-out accuracy stalls or drops. Let's force it: tiny training set (300 reviews), many epochs.""")

code("""\
starved = ds["train"].shuffle(seed=0).select(range(300))

model2 = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
args2 = TrainingArguments(
    output_dir=str(ROOT / "runs" / "lesson3_overfit"),
    num_train_epochs=10,
    per_device_train_batch_size=16 if TIER == "gpu-low" else 32,
    learning_rate=1e-4,
    eval_strategy="epoch",
    logging_strategy="epoch",
    report_to="none",
    seed=42,
)
trainer2 = Trainer(model=model2, args=args2, train_dataset=starved, eval_dataset=ds["test"],
                   data_collator=collator, compute_metrics=metrics)

hist = trainer2.train().metrics
print("train loss history:", [round(r["loss"], 3) for r in trainer2.state.log_history if "loss" in r])
print("val   loss history:", [round(r["eval_loss"], 3) for r in trainer2.state.log_history if "eval_loss" in r])
print("val   acc  history:", [round(r["eval_accuracy"], 3) for r in trainer2.state.log_history if "eval_accuracy" in r])""")

md("""\
**You should see:** training loss march toward ~0 (perfect memorization of 300 reviews) while validation loss bottoms out then **rises**, and accuracy plateaus well below the healthy run. That rising val-loss curve is overfitting - the single most important diagnostic in ML. Cures: more data, fewer epochs (early stopping), regularization. (On CPU, 10 epochs over 300 reviews takes a few minutes - the curves are worth the wait.)

## Exercises""")

code("""\
# 1. Try it on YOUR sentences - edit the predict() list with reviews of movies
#    you've seen. Find one it gets wrong. Sarcasm usually works.
#
# 2. Early stopping: in the overfit run, which epoch had the best val loss?
#    That's the epoch you should have stopped at. Re-run with num_train_epochs
#    set to it.
#
# 3. Freeze the body (feature extraction): for p in model3.distilbert.parameters(): p.requires_grad = False
#    Only the head trains now (~1.5k params). Faster, but ceiling is lower. Feel the trade-off.
#
# 4. Speed: on a GPU with bf16 support, add bf16=True to TrainingArguments.
#    Roughly free 2x speedup.
#
# 5. Push it to your Hub account (needs internet + your login):
#      trainer.push_to_hub("my-first-finetuned-classifier")
#    Congratulations, you've published a model.

print("Your turn.")""")

md("""\
## Where you are now

You've used the **standard industry workflow**: pretrained model + `datasets` + `Trainer` + honest evaluation. Everything above scales to bigger data unchanged.

**Next - Lesson 4:** the main event. LoRA-tune a 1.7B **instruction-tuned language model** (Qwen3) — updating <1% of its parameters — to change its behavior. Still the same loop.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3 (ml-training)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path("lessons/02_finetune_classifier/lesson3_finetune_distilbert.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "with", len(cells), "cells")
