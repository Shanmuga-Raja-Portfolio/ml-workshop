# ML Workshop — learn to train models, hands-on, free forever

A complete, hands-on course that takes you from **training a neural net from scratch**
to **LoRA fine-tuning LLMs and image models** — running entirely on your own machine.

- **Free and open — MIT licensed.** No subscriptions, no accounts, no paywalls.
- **Runs on almost anything.** CPU-only laptop → gaming GPU → workstation. The course
  detects your hardware and scales itself (see [docs/PORTABILITY.md](docs/PORTABILITY.md)).
- **License-clean.** Only permissively-licensed material is committed here; models and
  datasets are downloaded by *you* at first run, each with its license shown. Full
  accounting: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
- **Interactive.** Every lesson is a Jupyter notebook with "you should see: ..."
  checkpoints after each step.

## Quickstart

**Windows:** double-click `START HERE.bat` → wait for the browser to open.

**macOS / Linux:** `./start-here.sh`

**Manual (any OS):**

```bash
python tools/bootstrap.py --download   # sets up env, detects hardware, downloads models
uv run jupyter lab                     # open the notebooks
```

Bootstrap installs [uv](https://docs.astral.sh/uv/) if needed, creates an isolated
environment (your system Python is never touched), picks the right PyTorch build for
your hardware, and downloads the course models — printing each item's license as it
goes. No internet after first run? Models are cached locally.

**No bootstrap?** The notebooks also auto-detect hardware on their own. You just need
`uv sync` (or any Python 3.12 env with the deps in `pyproject.toml`).

## The course (do them in order)

| # | Notebook (under `lessons/`) | What you build | Time (GPU) |
|---|---|---|---|
| 1 | `01_foundations/lesson1_tiny_net_from_scratch.ipynb` | First neural net, by hand | ~2 min |
| 2 | `01_foundations/lesson2_tiny_gpt_from_scratch.ipynb` | A tiny GPT that writes fake Shakespeare | ~3 min |
| 3 | `02_finetune_classifier/lesson3_finetune_distilbert.ipynb` | Sentiment classifier at ~95% + overfitting demo | ~5 min |
| 4 | `03_llm_lora/lesson4_lora_qwen3.ipynb` | LoRA-tune a 1.7B chat model into a pirate | ~5 min |
| 5 | `04_diffusion_lora/lesson5_style_lora.ipynb` | Diffusion from scratch + teach image models a public-domain art style | ~15 min |

**The through-line:** every lesson is the same five lines — *data → forward → loss →
backward → step*. Only the data format, model size, and *which* parameters move change.

## Hardware tiers

| Your machine | Your experience |
|---|---|
| GPU ≥12 GB VRAM | Everything, including the FLUX.1-schnell bonus track |
| GPU 4–12 GB | Everything except FLUX training (SD1.5 LoRA trains in ~3 GB) |
| CPU only | Lessons 1–3 in full; lesson 4 at smoke-test scale; lesson 5's from-scratch part + pre-computed results |

Details and the "why not Vulkan?" answer: [docs/PORTABILITY.md](docs/PORTABILITY.md).

## If something goes wrong

- **Browser didn't open** → copy the `http://localhost:8888/lab?token=...` line from the
  console into your browser.
- **Out of memory** → close GPU-hungry apps (browsers, games) and re-run the cell.
- **Download failed** → re-run `python tools/bootstrap.py --download` (it resumes), or
  follow the manual fallback instructions in `downloads/manifest.json`.
- **License sanity check** → `uv run python tools/license_audit.py`.

## What's next

See [ROADMAP.md](ROADMAP.md) for planned releases (QLoRA, preference tuning, inference &
deployment, RAG & agents, more modalities) — shipped incrementally, free as always.

## Contributing & community

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) (DCO sign-off, **no
CLA**), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).
One hard rule: **only permissively-licensed material may be committed** (MIT/BSD/
Apache/CC0/public domain) — `tools/license_audit.py` and `tools/check_repo.py` enforce
it in CI. When in doubt, make it a runtime download with a manifest entry instead.

More docs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (how the repo works),
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) (help & FAQ),
[CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE). Third-party components (all downloaded at runtime, never
committed) are documented in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
