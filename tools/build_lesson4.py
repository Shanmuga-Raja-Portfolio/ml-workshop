"""Builds lesson4 notebook: LoRA fine-tune Qwen3-1.7B into a pirate (TRL + PEFT).
Hyperparameters verified on a 16 GB GPU with bf16; the notebook self-scales by hardware tier."""
import json
from pathlib import Path

cells = []
def md(src): cells.append({"cell_type": "markdown", "id": f"md{len(cells)}", "metadata": {}, "source": src.splitlines(keepends=True)})
def code(src): cells.append({"cell_type": "code", "id": f"code{len(cells)}", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)})

md("""\
# Lesson 4 — The main event: LoRA-tune a real language model

Everything so far converges here. **Qwen3-1.7B**: a 1.7-billion-parameter chat model that already knows an enormous amount. We will change its *behavior* — make it answer everything in pirate dialect — by training **0.37% of its parameters**.

### Why LoRA?

Fine-tuning all 1.7B parameters means storing a full copy for every style you teach. LoRA (**Lo**w-**R**ank **A**daptation) instead:
- **freezes** the base model completely
- slips tiny trainable matrices alongside each attention layer (here: 6.4M params)
- at run time: base output + small learned adjustment

Style and behavior live in low-rank adjustments; knowledge stays in the frozen base. Result: your whole custom personality is a **26 MB file** that stacks on the shared 3.4 GB base — swap personalities like lenses.

**Time:** ~3 min on a recent GPU (4.4 GB VRAM measured on a 16 GB card). On CPU it runs as a ~20-step **smoke test** — every concept works, but the full pirate transformation needs a GPU (see section 4).""")

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

# repo root, so run outputs land in <repo>/runs no matter where Jupyter started
ROOT = next((p.parent for p in [Path("workshop_config.json"), Path("../../workshop_config.json"),
                                Path("../../../workshop_config.json")] if p.exists()),
            Path("../..")).resolve()
(ROOT / "runs").mkdir(exist_ok=True)""")

code("""\
import random
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

MODEL = "Qwen/Qwen3-1.7B"
random.seed(42)
torch.manual_seed(42)

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map=DEVICE)
print(f"loaded in {time.time()-t0:.0f}s | {sum(p.numel() for p in model.parameters()):,} parameters")
if DEVICE == "cuda":
    total = CFG["vram_gb"] or torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM in use: {torch.cuda.memory_allocated()/1e9:.1f} GB of {total:.0f}")""")

md("""\
## 1. The baseline — meet the model as-shipped

The chat template below is the whole interface: a language model is *just* next-token prediction, and the template is the format it was trained to converse in. (Qwen3 has a thinking mode; `enable_thinking=False` gives the direct-answer format we'll train on.)""")

code("""\
def chat(prompt, max_new_tokens=90):
    model.eval()                                    # dropout OFF - see the warning in section 5!
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

print("what the model actually sees:")
print(tok.apply_chat_template([{"role": "user", "content": "hi"}],
      tokenize=False, add_generation_prompt=True, enable_thinking=False))

TESTS = ["How do I make a cup of tea?",          # in training set later
         "How do I train a dog?",                # in training set later
         "How do I fix a flat tire?"]            # NOT in training set - remember this one
print("\\n=== BEFORE (base model) ===")
for q in TESTS:
    print(f"Q: {q}\\nA: {chat(q)}\\n")""")

md("""\
Competent, helpful, boring. Note the third question — we never show the model any tire data, so what happens to it after training is the **generalization test**.

## 2. Build the dataset

In real fine-tuning, dataset quality decides everything. We need (question, pirate-answer) pairs. Rather than downloading, we *engineer* them: 24 solid answers × 3 question phrasings, passed through a `pirateize()` dialect filter. Deterministic seed = reproducible dataset.""")

code("""\
TASKS = [
    ("make a cup of tea", "Boil fresh water, warm the pot first, add one spoon of tea leaves, and let it steep for four minutes before you pour."),
    ("cook rice", "Rinse the rice until the water runs clear, then simmer one cup of rice in two cups of water for about fifteen minutes."),
    ("learn programming", "Pick one language and stick with it. Build a small project every week, and read other people's code whenever you can."),
    ("get better sleep", "Go to bed at the same time every night, keep the room dark and cool, and avoid screens for an hour before bed."),
    ("save money", "Track every expense for a month, cut the three biggest wastes, and move savings away the day you get paid."),
    ("make friends", "Show up regularly to the same places, ask people questions about themselves, and follow up on what they told you."),
    ("learn a language", "Study a little every day instead of a lot once a week. Consume shows and music in the language, and speak from day one."),
    ("stay focused", "Work in short timed sprints, put the phone in another room, and write down the one task you are committing to before you start."),
    ("cook an omelette", "Whisk the eggs fully, use a pan at medium heat with butter, and fold just before it fully sets so it stays soft."),
    ("write a resume", "Lead with results and numbers, not duties. One page, tailored to the job, and no typos anywhere."),
    ("train a dog", "Reward the behavior you want within one second, keep sessions to five minutes, and never punish what you failed to teach."),
    ("give a good apology", "Say what you did, say why it hurt them, and say what you will change. Skip the word but entirely."),
    ("make coffee", "Use freshly ground beans, water just off the boil, and weigh both: about thirty grams of coffee for half a liter of water."),
    ("reduce stress", "Daily walks, slow breathing for two minutes when it spikes, and saying no to one commitment per week."),
    ("clean a room fast", "Set a fifteen minute timer, work in one direction around the room, and touch each item only once."),
    ("start running", "Begin with run-walk intervals three times a week, increase distance by ten percent weekly, and rest whenever something hurts."),
    ("remember things", "Write it down the moment you hear it, review your notes within a day, and connect each new fact to something you already know."),
    ("choose a book", "Read page seventy, not the cover. If that page pulls you forward, the book will too."),
    ("make bread", "Weigh everything, knead until the dough is smooth, let it rise until doubled, and bake hot with steam at the start."),
    ("plan a trip", "Fix the budget first, book the two big things early, and leave one full day with nothing scheduled."),
    ("negotiate a salary", "Research the range, name a number first, stay silent after they respond, and get every agreement in writing."),
    ("take a good photo", "Light matters more than the camera: face the light source, check the background, and take three more shots than you need."),
    ("learn to draw", "Draw every day for twenty minutes, copy artists you admire, and fill pages with hands and faces, the two hardest things."),
    ("build good habits", "Make it tiny, attach it to something you already do, and never miss twice in a row."),
]

PIRATE_WORDS = {
    "you": "ye", "your": "yer", "yours": "yer", "my": "me",
    "is": "be", "are": "be", "am": "be", "hello": "ahoy", "friend": "matey",
    "friends": "mateys", "yes": "aye", "money": "doubloons", "water": "brine",
    "food": "grub", "people": "crew", "person": "soul", "work": "toil",
    "phone": "spyglass", "room": "cabin", "house": "ship", "job": "voyage",
    "day": "day at sea", "week": "tide", "never": "ne'er", "and": "an'", "of": "o'",
}

def pirateize(text):
    words = []
    for word in text.split(" "):
        core = word.strip(".,:;!?")
        if PIRATE_WORDS.get(core.lower()):
            word = word.replace(core, PIRATE_WORDS[core.lower()])
        words.append(word)
    s = " ".join(words).rstrip(".")
    opener = random.choice(["Arr,", "Arr,", "Yarr,", "Aye,", "Shiver me timbers,", "Avast,"])
    closer = random.choice([", arrr", ", ye scurvy dog", ", arrr", "! Har har!", ", savvy?"])
    return f"{opener} {s}{closer}"

PHRASINGS = ["How do I {t}?", "What's the best way to {t}?", "Any tips on how to {t}?"]
pairs = [(p.format(t=task), pirateize(answer)) for task, answer in TASKS for p in PHRASINGS]
random.shuffle(pairs)
print(f"{len(pairs)} training examples")
print("q:", pairs[0][0]); print("a:", pairs[0][1])""")

code("""\
# Format each pair with the model's own chat template -> one "text" example per row
rows = [{"text": tok.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
            tokenize=False, enable_thinking=False)}
        for q, a in pairs]
ds = Dataset.from_list(rows)
print(rows[0]["text"])      # this string is ALL the model ever sees - no magic""")

md("""\
## 3. Inject LoRA

`target_modules` = the attention projections from Lesson 2 (q, k, v, o — sound familiar?). r=16 is the rank of the injected matrices: bigger = more expressive, more overfit risk.""")

code("""\
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()      # 0.37% - the LoRA pitch in one line""")

md("""\
## 4. Train (TRL's SFTTrainer)

TRL = Transformers Reinforcement Learning, the tuning library on top of the Trainer you already know. Same five-line loop under the hood — loss is cross-entropy on the training text. Note the starting loss around **7**: pirate dialect is genuinely alien to this model. Watch it fall.

> **CPU tier note:** the cell below automatically runs **20 steps instead of the full 12 epochs**. That's a smoke test — it proves the whole pipeline (dataset → LoRA → Trainer → adapter file) works end to end, and the loss will start to fall, but 20 steps is not enough to teach the dialect. For the real transformation, run the same notebook on a GPU (a free Colab T4 works; set `TIER` aside and it scales itself back up).""")

code("""\
args = SFTConfig(
    output_dir=str(ROOT / "runs" / "lesson4"),
    max_length=384,
    per_device_train_batch_size=1 if TIER == "gpu-low" else 2,
    gradient_accumulation_steps=8 if TIER == "gpu-low" else 4,  # effective batch 8 without more VRAM
    num_train_epochs=12,
    max_steps=20 if TIER == "cpu" else -1,   # CPU: smoke test only - see the note above
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    logging_steps=10,
    report_to="none",
    bf16=(DEVICE == "cuda" and DTYPE == torch.bfloat16),
    seed=42,
    save_strategy="no",
    dataset_text_field="text",
)
if TIER == "cpu":
    print("CPU tier: 20-step smoke test. The pipeline is real; the pirate accent needs a GPU run.")
trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)

t0 = time.time()
trainer.train()
msg = f"\\ntrained in {(time.time()-t0)/60:.1f} min"
if DEVICE == "cuda":
    msg += f" | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB"
print(msg)""")

md("""\
**You should see:** loss fall ~7 → ~0.2 (it memorizes the 72 examples — expected; style is what transfers), in about 2 minutes on a 16 GB GPU, using under a third of its VRAM. On the 20-step CPU smoke test, expect loss to start ~7 and begin falling (roughly into the 5-6 range) — the trajectory is the point, not the destination.

## 5. Talk to your pirate

> **A bug I actually hit while building this lesson:** the tuned model first produced garbage (`Just:::::::::...`) because after `trainer.train()` the model is still in **train mode** — dropout was randomly zeroing LoRA units *during generation*. `model.eval()` before generating, always. Now you've seen it too.""")

code("""\
print("=== BEFORE vs AFTER ===")
AFTER = {q: chat(q) for q in TESTS}
for q in TESTS:
    print(f"Q: {q}")
    print(f"   tuned: {AFTER[q]}\\n")""")

md("""\
**What to look for:**
- tea & dog (in training): full pirate, near-verbatim — memorization
- **flat tire** (never seen): still pirate *style* on a topic we never taught — **generalization**. The LoRA captured a dialect, not a lookup table. That's the whole trick of fine-tuning.

(On a 20-step CPU smoke test the accent won't have formed yet — BEFORE and AFTER will look similar. Expected: the evidence that the machinery works is in the loss curve, not the accent.)

## 6. The payoff: 26 MB of personality""")

code("""\
ADAPTER_DIR = ROOT / "runs" / "lesson4_pirate_adapter"
model.save_pretrained(ADAPTER_DIR)
import os
size = sum(os.path.getsize(os.path.join(r, f))
           for r, _, fs in os.walk(ADAPTER_DIR) for f in fs) / 1e6
print(f"adapter: {size:.0f} MB  vs  base model: 3,400 MB  ({100*size/3400:.1f}%)")

# reload path (how you'd use it in a real app):
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map=DEVICE)
pirate = PeftModel.from_pretrained(base, ADAPTER_DIR)
print("reload works - same behavior, one line to attach/detach")""")

md("""\
## Exercises""")

code("""\
# 1. NEW PERSONALITY: keep TASKS, replace pirateize() with your own filter
#    (caveman? haiku answers? everything in bullet points?). Rerun everything.
#    You now have a repeatable recipe for ANY style transfer.
#
# 2. RANK: r=4 vs r=16 vs r=64 - watch trainability vs memorization of the
#    held-out tire question. Bigger is not automatically better.
#
# 3. LEARNING RATE: 5e-5 (too weak?) and 5e-4 (too hot? watch generation quality).
#
# 4. MERGE: model = model.merge_and_unload() folds LoRA into the base weights -
#    a standalone model with zero extra latency. When would you merge vs keep separate?
#
# 5. QLoRA stretch: quantize the base to 4-bit before LoRA
#    (BitsAndBytesConfig(load_in_4bit=True) in from_pretrained) -> under 5 GB total.
#    How you'd tune 8B+ models on a 16 GB card.
#
# 6. SHARE: model.push_to_hub("your-name/qwen3-pirate-lora") - 26 MB upload,
#    anyone can stack your personality on the public base.

print("Your turn.")""")

md("""\
## Where you are now

The full arc, completed: regression from scratch (L1) → transformer from scratch (L2) → standard fine-tune (L3) → **LoRA personality transplant on a production-size model** (L4). Same five lines every time, scaled.

**Final lesson - number 5:** same LoRA idea, different universe: teach Stable Diffusion to draw in a new art style.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3 (ml-training)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path("lessons/03_llm_lora/lesson4_lora_qwen3.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "with", len(cells), "cells")
