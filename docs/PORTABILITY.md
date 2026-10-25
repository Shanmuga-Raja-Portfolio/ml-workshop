# Portability: how this course runs on anyone's machine

The course is designed so **every lesson works on every machine**, with the experience
tiered by hardware rather than gated by it.

## The accelerator matrix

`tools/bootstrap.py` auto-detects your hardware at first run and picks the right
PyTorch build. Nothing vendor-specific is committed to this repo — the package manager
downloads the right wheel for *your* machine.

| Your hardware | What bootstrap selects | What works |
|---|---|---|
| NVIDIA GPU (any recent) | CUDA wheel (cu12x) | Everything, full speed |
| AMD GPU on Linux | ROCm wheel | Everything (ROCm support varies by card — check PyTorch's list) |
| Apple Silicon Mac | Default wheel (MPS built in) | Everything; MPS used automatically |
| AMD GPU on Windows | CPU wheel | Lessons 1–4 at reduced scale + all low-spec paths |
| Intel/AMD CPU only | CPU wheel | Lessons 1–3 fully; lessons 4–5 at reduced scale + pre-computed results |

Every notebook's first cell reads `workshop_config.json` (written by bootstrap) and
scales itself: batch sizes, step counts, and which optional sections unlock. If the
config is missing, notebooks auto-detect with plain PyTorch — so the course still works
if you skip bootstrap entirely.

### Hardware tiers

| Tier | Meaning | Experience |
|---|---|---|
| `gpu-high` | ≥12 GB VRAM | Everything, including Lesson 5's FLUX.1-schnell bonus track |
| `gpu-low` | 4–12 GB VRAM | Everything except FLUX training (SD1.5 LoRA trains in ~3 GB) |
| `cpu` | No usable GPU | Lessons 1–3 in full; lesson 4 at smoke-test scale; lesson 5's from-scratch diffusion part + pre-computed results for the GPU parts |

## Why not Vulkan? (the honest answer)

We get asked for Vulkan because it's the obvious "any GPU, any OS, one API" dream —
and it *would* be the ideal teaching backend. Here's the reality as of this writing:

- **PyTorch's Vulkan backend is a prototype.** It covers a small set of inference
  operations; it cannot train models (no autograd support on the Vulkan path), and most
  transformer/diffusion ops in these lessons aren't implemented there.
- A course that requires training — the entire point here — would simply be broken for
  everyone if we built it on Vulkan today.
- What actually delivers "works for anyone" **today** is: BSD-3-Clause PyTorch as the
  core, CPU as the universal baseline, and vendor backends (CUDA/ROCm/MPS) as optional
  accelerators downloaded per-machine.

We're tracking PyTorch's Vulkan support. When it can train a transformer end-to-end,
adding a Vulkan tier is a bootstrap change, not a course rewrite — the config contract
(`workshop_config.json`) already has room for it.

## The license angle

Everything above is chosen to keep the licensing story clean too:

- PyTorch core: BSD-3-Clause, always.
- CUDA wheels bundle NVIDIA binaries under separate NVIDIA licenses — downloaded by
  your package manager, never committed here. Full details in
  `THIRD_PARTY_LICENSES.md` ("The CUDA wrinkle").
- CPU/MPS/ROCm builds involve no NVIDIA-licensed binaries at all.
