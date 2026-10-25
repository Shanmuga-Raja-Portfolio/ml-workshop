# Troubleshooting & FAQ

Stuck? Start here. Still stuck? Open a GitHub issue with the exact red error text,
your OS, and your GPU (or "CPU only").

## Setup

**`python` is not recognized (Windows)**
Install Python 3.12+ from python.org (tick "Add python.exe to PATH"), or use the
`py` launcher: `py tools/bootstrap.py`. Easiest path: double-click `START HERE.bat`.

**`uv` is not recognized / jupyter not found**
You typed a command by hand before setup ran. Run `python tools/bootstrap.py` first —
it finds or installs uv and builds the environment.

**Bootstrap fails partway through**
Just re-run it. It's idempotent and downloads resume where they stopped.

**"No matching distribution" / wheel errors during sync**
Usually a too-old Python or a platform without prebuilt wheels. This course requires
Python 3.12.x. On unusual platforms, bootstrap falls back to the CPU build of PyTorch.

## Jupyter

**Browser didn't open**
Copy the `http://localhost:8888/lab?token=...` line from the console into your browser.

**"Kernel not found" in VS Code**
Top-right of the notebook → Select Kernel → Python Environments → pick the one ending
in `(.venv)`.

**A cell failed with a red error**
Re-run it once (some failures are flaky downloads). If it repeats, read the last line
of the error first — it's usually the real one.

## Hardware

**CUDA out of memory**
Close GPU-hungry apps (browsers with many tabs, games), then re-run. If it persists,
delete `workshop_config.json` and re-run bootstrap with `python tools/bootstrap.py --cpu`
to force smaller runs, or edit the config's tier.

**PyTorch installed but GPU not detected** (`torch.cuda.is_available()` is False)
Your torch wheel doesn't match your GPU. Fix: `python tools/bootstrap.py` — it picks
the right wheel index for your hardware. Very new NVIDIA cards need CUDA ≥ 12.8 builds;
bootstrap handles this.

**Everything is very slow**
You're on the CPU path. Check the first cell's output: `Workshop tier: cpu`. That's
fine — the course scales down deliberately — but lessons 4–5 are smoke-test scale
there. A free cloud GPU (Colab T4) runs everything at full scale.

**macOS: MPS errors or weird results**
The config uses float16 on Apple Silicon. If you hit an MPS-specific bug, force CPU by
deleting `workshop_config.json` and running `python tools/bootstrap.py --cpu`.

## Downloads & models

**A model download failed / rate-limited**
Re-run `python tools/bootstrap.py --download` — it resumes. Hugging Face rate limits
clear with time; a free HF account token (`HF_TOKEN` env var) raises the limit.

**No internet / air-gapped machine**
Download everything once on a connected machine, then copy the Hugging Face cache
(`~/.cache/huggingface` on macOS/Linux, `%USERPROFILE%\.cache\huggingface` on Windows)
and the repo's `data/` folder to the offline machine. `downloads/manifest.json` lists
every item with manual instructions.

**Where do downloads live? How much disk?**
Models go to your Hugging Face cache (~8–30 GB depending on which lessons you run);
images/datasets go to `data/` inside the repo. Everything regenerates — deleting is safe.

## Lesson-specific

**Lesson 3 accuracy differs from 95.3%**
Normal — fine-tuning has run-to-run variance of a point or two. CPU tier uses a smaller
subset and lands lower (~85–90%); that's expected and the lesson says so.

**Lesson 4 pirate speaks garbage after training**
Call `model.eval()` before generating (dropout stays on otherwise). The lesson covers
this — it's the most famous gotcha in the course.

**Lesson 5: black images from SD1.5**
Its VAE overflows in pure fp16. Use bf16 (the config does this automatically on
supported GPUs).

**Lesson 5: FLUX part out of memory**
FLUX needs ≥8–12GB even at 4-bit. On smaller GPUs, skip it — the SD1.5 part (~3GB) is
the main track, and pre-computed FLUX results are included in the lesson.

## FAQ

**Is this really free? What's the catch?**
MIT-licensed, no accounts, no telemetry, no catch. Models download from Hugging Face
at first run — that's the only thing leaving your machine.

**Can I teach a class / paid workshop with this?**
Yes. MIT means you can use it commercially too. Keep the LICENSE file; that's all we ask.

**Do I need a GPU?**
No. Every lesson runs on CPU — some at reduced scale, with pre-computed results so you
never miss the payoff.

**Where's the Vulkan version?**
See `docs/PORTABILITY.md` — short version: PyTorch's Vulkan backend can't train models
yet. We're tracking it.
