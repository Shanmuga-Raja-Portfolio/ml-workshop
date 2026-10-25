# Architecture — how this repo fits together

Read this before contributing. Five minutes here saves an hour of confusion.

## Repo layout

```
├── START HERE.bat / start-here.sh   # zero-typing entry points (Windows / macOS+Linux)
├── lessons/                          # the course — Jupyter notebooks, in order
│   ├── 01_foundations/               # lessons 1-2 (from scratch; + vendored public-domain data)
│   ├── 02_finetune_classifier/       # lesson 3 (DistilBERT)
│   ├── 03_llm_lora/                  # lesson 4 (Qwen3 LoRA)
│   └── 04_diffusion_lora/            # lesson 5 (DDPM from scratch + SD1.5/FLUX LoRA)
├── tools/
│   ├── bootstrap.py                  # stdlib-only first-run setup (hardware detect, uv, torch index)
│   ├── fetch_data.py                 # manifest-driven downloader (models, datasets, images)
│   ├── license_audit.py              # fails if any installed dep isn't permissively licensed
│   ├── check_repo.py                 # hygiene gate: scrub greps, clean notebooks, generator sync
│   ├── build_lesson2..5.py           # NOTEBOOK GENERATORS — see golden rule below
│   └── download_all.sh               # legacy bash fallback (calls bootstrap --download)
├── downloads/
│   ├── manifest.json                 # every runtime download: ref, license, fallback instructions
│   └── ukiyoe_sources.csv            # curated public-domain image sources (Wikimedia Commons)
├── docs/                             # PORTABILITY.md, TROUBLESHOOTING.md, this file
├── website/                          # drop-in static page linking the repo
├── runs/                             # gitignored scratch: checkpoints, adapters, images
├── data/                             # gitignored: downloaded datasets/images land here
└── workshop_config.json              # gitignored: written by bootstrap, read by every notebook
```

## The golden rule: notebooks are generated

`lessons/01_foundations/lesson2...` through `lesson5...` are **generated artifacts**.
The source of truth is `tools/build_lessonN.py`.

```bash
uv run python tools/build_lesson4.py    # rewrites lessons/03_llm_lora/lesson4_lora_qwen3.ipynb
```

Never hand-edit a generated notebook — CI regenerates all of them and fails the PR if
they drift (`tools/check_repo.py`). Lesson 1 predates the generator system and is the
one exception: edit its JSON directly, carefully, and keep outputs empty.

## The hardware config contract

`tools/bootstrap.py` writes `workshop_config.json` at the repo root:

```json
{"device": "cuda"|"mps"|"cpu",
 "accelerator": "cuda"|"rocm"|"mps"|"cpu",
 "dtype": "bfloat16"|"float16"|"float32",
 "vram_gb": 16.0,
 "tier": "gpu-high"|"gpu-low"|"cpu"}
```

Tier rules: `vram_gb >= 12` → `gpu-high`; `0 < vram_gb < 12` → `gpu-low`; else `cpu`.
dtype: cuda → bfloat16 (float16 pre-Ampere), mps → float16, cpu → float32.

**Every notebook's first code cell** loads this file (walking up from the lesson
directory) and falls back to torch auto-detection if it's missing — so the course
works even without bootstrap. Notebooks then scale themselves:

| Lesson | cpu tier | gpu-low tier |
|---|---|---|
| 1 | unchanged (tiny) | unchanged |
| 2 | 500 steps (~10-15 min) | batch 32 |
| 3 | 2k-train/500-eval subset | halved batch sizes |
| 4 | 20-step smoke test | batch 1, grad-accum 8 |
| 5 | Part A (DDPM/Fashion-MNIST) only + pre-computed results | Part B (SD1.5, ~3GB) but not Part C (FLUX) |

If you change a lesson, you must preserve the `cpu` tier running without error.

## Bootstrap flow

```
START HERE.bat / start-here.sh
        │
        ▼
tools/bootstrap.py  (stdlib only — runs before any deps exist)
  1. find or install uv
  2. detect accelerator (nvidia-smi / rocm / Apple Silicon / cpu)
  3. write workshop_config.json
  4. uv sync with the right torch wheel index for this platform
        (pyproject.toml is deliberately platform-neutral — the index
         override is injected here, never committed)
  5. --download: tools/fetch_data.py walks downloads/manifest.json,
     prints each item's license, retries with resume
        ▼
uv run jupyter lab
```

## The manifest contract

`downloads/manifest.json` is the single source of truth for runtime downloads.
Each entry: `id, type (hf-model|hf-dataset|torchvision-dataset|url-list), ref,
license, license_url, size_hint, required_by, optional, fallback`.

To add a model/dataset: add a manifest entry (with an honest `fallback` string) and
teach `fetch_data.py` if it's a new `type`. If its license isn't permissive, it must
stay a download — see `THIRD_PARTY_LICENSES.md` for what that means and why.

## The gates (all enforced in CI, runnable locally)

```bash
uv run python tools/license_audit.py   # every installed dep's license vs allowlist
uv run python tools/check_repo.py      # scrub greps, empty notebook outputs,
                                       # generators regenerate notebooks byte-identical
```

## Adding a new lesson

1. Write `tools/build_lessonN.py` (copy the structure of an existing one:
   `md()`/`code()` helpers, config cell first, tier scaling, "You should see" notes).
2. Any model/dataset → `downloads/manifest.json` entry, license documented.
3. Regenerate, execute end-to-end on the lowest tier you claim to support, verify the
   "You should see" numbers, clear outputs.
4. Update README table + ROADMAP. PR with DCO sign-off.

## Release process (maintainer)

1. Gates green on `main`.
2. Bump `version` in `pyproject.toml`, add `CHANGELOG.md` entry.
3. `git tag vX.Y.Z`, push, create GitHub Release.
4. Attach any pre-trained adapters to the Release — each labeled with its own license
   (SD1.5-derived adapters are OpenRAIL-M, never MIT; FLUX-schnell-derived are clean).
