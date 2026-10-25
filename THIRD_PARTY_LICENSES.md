# Third-Party Licenses

This course is MIT-licensed (see `LICENSE`). This document is the complete, honest
accounting of **everything else your machine touches** — what its license is, why it's
OK, and where the sharp edges are. We wrote this because licensing traps in ML are real
and usually undocumented.

**The one-sentence version:** everything committed to this repository is MIT /
BSD / Apache-2.0 / CC0 / public domain. Anything with a more complicated license is
**downloaded to your machine by you at first run** (via `tools/bootstrap.py --download`
or manually — see `downloads/manifest.json`), never redistributed by us.

---

## 1. Python libraries (installed by uv/pip, never committed)

| Package | License | Notes |
|---|---|---|
| torch, torchvision | BSD-3-Clause | See "The CUDA wrinkle" below |
| transformers, datasets, peft, trl, diffusers, accelerate, huggingface-hub | Apache-2.0 | |
| bitsandbytes | MIT | |
| jupyter, ipykernel | BSD-3-Clause | |
| matplotlib | PSF-based (permissive) | |
| pandas | BSD-3-Clause | |
| scikit-learn | BSD-3-Clause | |
| uv (installer/tool) | MIT / Apache-2.0 | |

`tools/license_audit.py` re-verifies this table against your actual installed
environment and **fails loudly** if anything non-permissive appears. Run it any time:

```bash
uv run python tools/license_audit.py
```

### The CUDA wrinkle (the NVIDIA trap, documented)

PyTorch itself is BSD-3-Clause. But the **CUDA-enabled PyTorch wheels** that pip/uv
downloads for NVIDIA GPUs bundle NVIDIA binary libraries (cuBLAS, cuDNN, CUDA runtime,
etc.) which are governed by **separate NVIDIA licenses** (CUDA Toolkit EULA and
related). This is normal and legal — NVIDIA explicitly distributes these through
PyPI for exactly this purpose — but you deserve to know:

- Those NVIDIA binaries are **downloaded to your machine by your package manager**.
  This repository contains none of them.
- Their licenses permit use and redistribution *as part of the wheels*, which is how
  every PyTorch user on earth gets them.
- If you want a setup with zero NVIDIA-licensed binaries: use the CPU build
  (bootstrap auto-selects it on non-NVIDIA machines) or ROCm on AMD/Linux.

## 2. Models & datasets (downloaded at first run, never committed)

Full machine-readable list: `downloads/manifest.json`. Bootstrap prints each license
before downloading.

| Item | License | Used in | Sharp edges |
|---|---|---|---|
| `Qwen/Qwen3-1.7B` | Apache-2.0 | Lesson 4 | None. Adapters you train are yours. |
| `distilbert/distilbert-base-uncased` | Apache-2.0 | Lesson 3 | None. |
| `stanfordnlp/sst2` | Research use | Lesson 3 | Stanford's sentiment treebank; fine for learning/research. Don't build a commercial product on the raw data. |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | **CreativeML OpenRAIL-M** | Lesson 5 | See "The OpenRAIL wrinkle" below. |
| `black-forest-labs/FLUX.1-schnell` | Apache-2.0 | Lesson 5 (bonus) | None. |
| Fashion-MNIST (via torchvision) | MIT | Lesson 5 | None. |
| Ukiyo-e woodblock prints (Wikimedia Commons) | Public domain | Lesson 5 | Artists died 100+ years ago; attribution in `data/ukiyoe/ATTRIBUTION.csv` anyway, because credit is good manners. |
| Tiny Shakespeare (vendored in `lessons/01_foundations/data/`) | Public domain | Lesson 2 | It's Shakespeare. He's fine with it. |

### The OpenRAIL wrinkle (Stable Diffusion 1.5)

OpenRAIL-M is **not** a permissive license — it's an "open responsible AI" license with
**use restrictions** (no illegal/harmful/deceptive uses, etc.) that **flow down to
derivatives**:

- You may download and fine-tune SD1.5 — that's exactly what the license allows.
- **Any adapter/LoRA you train on it inherits the OpenRAIL-M use restrictions.** If you
  share your adapter, you must pass those restrictions along. (In this course we say so
  in the lesson, and the pre-trained adapter we distribute separately is labeled
  OpenRAIL-M — not MIT.)
- By downloading it at first run *yourself* from Hugging Face, you accept those terms
  directly. This repository never redistributes the weights.

If you'd rather stay 100% permissive end-to-end: Lesson 5's bonus track uses
**FLUX.1-schnell (Apache-2.0)** — adapters trained on it carry no use restrictions.

## 3. What this repo deliberately does NOT contain

- No model weights, no datasets, no NVIDIA/CUDA binaries.
- No copyrighted artwork or scraped images. (The course previously used a copyrighted
  anime image dataset during private development; it was removed entirely for release.)
- No API keys, tokens, or personal paths.

Found something that shouldn't be here? Open an issue — that's a bug, and we'll treat
it like one.

---

*This document is practical information for learners, not legal advice.*
