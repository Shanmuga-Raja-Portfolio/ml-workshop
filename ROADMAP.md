# ML Workshop — Course Roadmap

Hands-on learning path. Everything runs locally on your own machine — no subscription,
no accounts, no cloud required.

## How to run things

All Python lives in an isolated uv-managed environment pinned to Python 3.12 — your
system Python stays untouched.

```bash
python tools/bootstrap.py --download   # one-time setup + model downloads
uv run jupyter lab                     # open notebooks in your browser
```

Windows: double-click `START HERE.bat`. macOS/Linux: `./start-here.sh`.
Or open the folder in VS Code and run notebooks with the project `.venv` kernel.

## The course — v1.0

### Stage 1 — Foundations
1. **Lesson 1 — Train a tiny neural net from scratch**
   (`lessons/01_foundations/lesson1_tiny_net_from_scratch.ipynb`)
   Tensors, autograd, and the 5-line training loop by hand. A ~4k-param net learns
   `sin(x)`. *Verified: loss 0.62 → 0.002.* ~2 min.
2. **Lesson 2 — Build a tiny GPT from scratch**
   (`lessons/01_foundations/lesson2_tiny_gpt_from_scratch.ipynb`)
   Tokenizers (char-level by hand + a real BPE), causal attention hand-rolled, a
   2.7M-param transformer learns to write fake Shakespeare. *Verified: loss 4.19
   (= ln 65, exactly random-guess theory) → 1.55.* ~3 min on GPU.

### Stage 2 — Fine-tune a real model
3. **Lesson 3 — Fine-tune DistilBERT (sentiment classifier)**
   (`lessons/02_finetune_classifier/lesson3_finetune_distilbert.ipynb`)
   The standard HF workflow: `datasets` → tokenize → `Trainer` → honest evaluation.
   Includes a deliberate overfitting demo (train loss → 0 while val loss rises) and a
   look at where small classifiers fail. *Verified: 95.3% accuracy.* ~5 min.

### Stage 3 — LLM fine-tuning
4. **Lesson 4 — LoRA-tune Qwen3-1.7B into a pirate**
   (`lessons/03_llm_lora/lesson4_lora_qwen3.ipynb`)
   Dataset engineering, chat templates, PEFT LoRA (0.37% of params), TRL SFTTrainer,
   before/after generations including a held-out question (generalization proof), a
   ~26MB adapter save/reload, and the eval-mode-vs-train-mode dropout gotcha.
   *Verified: full pirate dialect incl. unseen topics; ~2 min train; ~4.5GB VRAM.*

### Stage 4 — Image models
5. **Lesson 5 — Diffusion, from scratch to style LoRA**
   (`lessons/04_diffusion_lora/lesson5_style_lora.ipynb`)
   Three tiers, one idea: **(A)** a tiny diffusion model trained from scratch on
   Fashion-MNIST (runs on CPU); **(B)** LoRA-teach Stable Diffusion 1.5 a public-domain
   ukiyo-e style (trains in ~3GB VRAM); **(C)** the same trick on FLUX.1-schnell
   (Apache-2.0, 4-bit) for bigger GPUs. Pre-computed results included for low-spec
   machines.

**The through-line:** every lesson is the same five lines — data → forward → loss →
backward → step. Only the data format, model size, and *which* parameters move change.

## Downloads (first run only, then offline)

Everything is fetched by `python tools/bootstrap.py --download`, with the license shown
before each download. Full list + manual fallbacks: `downloads/manifest.json`; license
details: `THIRD_PARTY_LICENSES.md`.

- `Qwen/Qwen3-1.7B` (Apache-2.0, ~4GB) — lesson 4
- `distilbert/distilbert-base-uncased` + `stanfordnlp/sst2` — lesson 3
- `stable-diffusion-v1-5/stable-diffusion-v1-5` (OpenRAIL-M) — lesson 5
- `black-forest-labs/FLUX.1-schnell` (Apache-2.0, optional) — lesson 5 bonus
- Fashion-MNIST (MIT) + public-domain ukiyo-e prints — lesson 5
- tiny shakespeare (public domain, vendored in the repo) — lesson 2

## Planned releases (staggered)

- **v1.1 — The next arc:** QLoRA (4-bit) on 7–8B models, DPO/preference tuning, honest
  model evaluation.
- **v1.2 — Inference & deployment:** quantization, GGUF, local serving, running your
  adapters anywhere including CPU-only machines.
- **v1.3 — RAG & agents:** retrieval-augmented generation and a tool-calling agent from
  scratch.
- **v1.4 — More modalities:** audio (whisper-style), vision transformers from scratch.

Community requests welcome via GitHub issues.

## Gotchas learned the hard way (while building this course)

1. After `trainer.train()`, call `model.eval()` before generating — dropout stays on
   otherwise and outputs garbage.
2. Diffusion loss curves are fuzzy bands, not clean slopes — judge by images, not loss.
3. SD1.5's VAE in pure fp16 → black images (range overflow). bf16 is safe.
4. Never pipe a download command through `| tail` — you lose the real exit code.
5. `&&` doesn't work in PowerShell — use the provided `.bat` / `.sh` launchers.

## Notes

- Lesson notebooks write checkpoints/images into `runs/` — safe to delete any time,
  never committed to git.
- `tools/` contains the scripts that generated each notebook (`build_lesson*.py`) —
  rerun one to regenerate its notebook.
- `tools/license_audit.py` verifies every installed dependency is permissively licensed.
