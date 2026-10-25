"""Builds lesson5 notebook: diffusion from scratch + ukiyo-e style LoRA (SD1.5, FLUX bonus).
License-clean edition: public-domain ukiyo-e prints + Fashion-MNIST (MIT). Verified on RTX 5060 Ti."""
import json
from pathlib import Path

cells = []
def md(src): cells.append({"cell_type": "markdown", "id": f"md{len(cells)}", "metadata": {}, "source": src.splitlines(keepends=True)})
def code(src): cells.append({"cell_type": "code", "id": f"code{len(cells)}", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)})

md("""\
# Lesson 5 — Teach an image model a new style (diffusion LoRA)

The grand finale, and proof the pattern generalizes: **the exact same five lines from Lesson 1 — data, forward, loss, backward, step — now for images.**

This lesson is four parts, one through-line:

| Part | What | Who runs it |
|---|---|---|
| **A** | Build a diffusion model *from scratch*: tiny UNet, hand-rolled noise schedule, Fashion-MNIST | **Everyone** (CPU works) |
| **B** | LoRA-teach Stable Diffusion 1.5 the **ukiyo-e** woodblock style | GPU (≥ 4 GB VRAM) |
| **C** | Bonus: same trick on **FLUX.1-schnell**, quantized to 4-bit | Big GPU (≥ 12 GB) |
| **D** | Pre-computed results + how to load the trained adapters | **Everyone** |

The training data in Part B is 25 **public-domain** ukiyo-e prints (Hokusai, Hiroshige, Utamaro — artists who died well over a century ago), already in `data/ukiyoe/` with full attribution in `ATTRIBUTION.csv`. Fashion-MNIST in Part A is MIT-licensed. No licensing traps anywhere in this lesson — the one sharp edge (Stable Diffusion's model license) is flagged in a box when we get there.

**Time:** Part A ~2 min on GPU / ~5–10 min on CPU. Part B ~3 min on a 16 GB card. Part C ~15 min if your hardware qualifies.""")

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
ROOT = Path("../../").resolve()
print(f"Workshop tier: {TIER} | device: {DEVICE} | dtype: {DTYPE}")

(ROOT / "runs").mkdir(exist_ok=True)   # all training output lands in <repo>/runs""")

md("""\
---
# Part A — Diffusion from scratch (everyone runs this)

Before we LoRA a 1-billion-parameter image model, let's *be* one — small enough to see every moving part.

**The whole idea of diffusion, honestly told:** you take a real image and deliberately add noise to it, a little at a time, until it's pure static. Then you train a network to play the process *backwards* — given a noisy image, predict the noise, subtract it, repeat. Generation = starting from pure static and denoising step by step until a picture condenses out of the fog.

That's it. Everything else is plumbing.

Our dataset: **Fashion-MNIST** (MIT license) — 60,000 tiny grayscale images of shirts, shoes, and bags. Already downloaded in Lesson 1's neighborhood, at `data/FashionMNIST`.""")

code("""\
import time, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

random.seed(42); torch.manual_seed(42)

tfm = transforms.Compose([transforms.ToTensor(), lambda t: t * 2 - 1])   # [0,1] -> [-1,1]
fmnist = datasets.FashionMNIST(root=str(ROOT / "data"), train=True, download=True, transform=tfm)
print(f"{len(fmnist):,} training images, {fmnist[0][0].shape[-1]}x{fmnist[0][0].shape[-1]} grayscale")

fig, axes = plt.subplots(1, 8, figsize=(12, 1.8))
for ax, i in zip(axes, range(8)):
    ax.imshow((fmnist[i][0][0] + 1) / 2, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
plt.suptitle("the raw material: shoes, shirts, bags, trousers")
plt.show()""")

md("""\
**You should see:** eight tiny grayscale fashion items. Nothing learned yet — just data.

## A.1 The denoiser: a tiny UNet

The network's job: input a noisy image + "how noisy is it" (the timestep `t`), output its guess for the noise. The architecture is a **UNet** — shrink the image down to a bottleneck while remembering the details, then expand back up, pasting the remembered details back in via *skip connections* (that's the U shape). Timestep `t` is injected as a sinusoidal embedding — the exact same sinusoidal trick Lesson 2 used for positions — so the net knows which noise level it's looking at.

~1.1M parameters. Fashion-MNIST is 28×28, so the whole U is three levels deep.""")

code("""\
def sinusoidal(t, dim=64):
    half = dim // 2
    f = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * f[None, :]
    return torch.cat([a.sin(), a.cos()], dim=-1)     # same idea as Lesson 2's position encoding

class Block(nn.Module):
    \"\"\"conv -> norm -> +timestep -> conv -> norm, with a residual shortcut.\"\"\"
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.tproj = nn.Linear(tdim, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x, temb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.tproj(F.silu(temb))[:, :, None, None]   # inject "how noisy" as per-channel bias
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)

class TinyUNet(nn.Module):
    \"\"\"28x28 -> 14x14 -> 7x7 -> back up, skip connections across the U.\"\"\"
    def __init__(self, ch=48, tdim=128):
        super().__init__()
        self.tmlp = nn.Sequential(nn.Linear(64, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inc = nn.Conv2d(1, ch, 3, padding=1)
        self.b1 = Block(ch, ch, tdim)            # 28x28
        self.down1 = nn.Conv2d(ch, 2*ch, 3, stride=2, padding=1)
        self.b2 = Block(2*ch, 2*ch, tdim)        # 14x14
        self.down2 = nn.Conv2d(2*ch, 2*ch, 3, stride=2, padding=1)
        self.mid1 = Block(2*ch, 2*ch, tdim)      # 7x7 bottleneck
        self.mid2 = Block(2*ch, 2*ch, tdim)
        self.up2 = Block(4*ch, 2*ch, tdim)       # concat skip from 14x14
        self.up1 = Block(3*ch, ch, tdim)         # concat skip from 28x28
        self.outc = nn.Conv2d(ch, 1, 3, padding=1)
    def forward(self, x, t):
        temb = self.tmlp(sinusoidal(t))
        s1 = self.b1(self.inc(x), temb)
        s2 = self.b2(self.down1(s1), temb)
        h = self.mid2(self.mid1(self.down2(s2), temb), temb)
        h = self.up2(torch.cat([F.interpolate(h, scale_factor=2), s2], dim=1), temb)
        h = self.up1(torch.cat([F.interpolate(h, scale_factor=2), s1], dim=1), temb)
        return self.outc(h)

tiny = TinyUNet().to(DEVICE)
n_params = sum(p.numel() for p in tiny.parameters())
print(f"tiny UNet: {n_params:,} parameters — about 1/750th of Stable Diffusion's UNet")""")

md("""\
**You should see:** `tiny UNet: 1,111,553 parameters`. Small enough to train in minutes, big enough to learn what a boot looks like.

## A.2 The noise schedule — sabotage, on purpose

We define 1,000 noise levels. Level `t` mixes an image with static: mostly image at low `t`, mostly static at high `t`. A **linear schedule** of betas decides how fast the fog rolls in; the math below (the cumulative product `ᾱ_t`) lets us jump straight to *any* noise level in one line, no need to simulate the in-between steps.

This is the **sabotage-and-quiz** method: we damage the data on purpose, then quiz the network on the damage.""")

code("""\
T = 1000                                                     # noise levels
betas = torch.linspace(1e-4, 0.02, T, device=DEVICE)         # linear schedule: how much fog per step
alphas = 1.0 - betas
acum = torch.cumprod(alphas, 0)                              # ᾱ_t = "how much signal survives at level t"

def add_noise(x, noise, t):
    a = acum[t][:, None, None, None]
    return a.sqrt() * x + (1 - a).sqrt() * noise             # √ᾱ·image + √(1-ᾱ)·static

# watch one shoe dissolve into fog
x0 = fmnist[7][0].unsqueeze(0).to(DEVICE)
levels = [0, 100, 300, 500, 700, 900, 999]
fig, axes = plt.subplots(1, len(levels), figsize=(13, 1.9))
for ax, t in zip(axes, levels):
    tt = torch.tensor([t], device=DEVICE)
    foggy = add_noise(x0, torch.randn_like(x0), tt)
    ax.imshow(((foggy[0, 0].clamp(-1, 1) + 1) / 2).cpu(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"t={t}", fontsize=9); ax.axis("off")
plt.suptitle("sabotage in action: same shoe, rising noise level t")
plt.show()""")

md("""\
**You should see:** a shoe that stays readable until about t=300, drowns by t=700, and is indistinguishable from TV static at t=999.

## A.3 THE loop — Lesson 1's five lines, diffusion edition

1. **data**: a batch of fashion images
2. **forward**: sabotage each one at a *random* noise level, ask the UNet "what noise did I add?"
3. **loss**: MSE between guessed and actual noise
4. **backward**: gradients
5. **step**: AdamW

Training on random noise levels at once teaches every level simultaneously — no curriculum needed.

> **Tier note:** GPU tiers run the full **2,000 steps** (~2 min measured on an RTX 5060 Ti, ~0.05 s/it). CPU tier automatically drops to **400 steps** (~3 min on a desktop CPU, up to ~10 on a laptop — be patient, it's real training, and the samples at the end will be blobbier but recognizable).""")

code("""\
STEPS_A = 400 if TIER == "cpu" else 2000
dl_a = DataLoader(fmnist, batch_size=128, shuffle=True, drop_last=True, num_workers=0)
opt_a = torch.optim.AdamW(tiny.parameters(), lr=1e-3)

tiny.train()
t0, step, losses_a = time.time(), 0, []
done = False
while not done:
    for x, _ in dl_a:
        x = x.to(DEVICE)                                       # 1. data
        noise = torch.randn_like(x)
        t = torch.randint(0, T, (x.shape[0],), device=DEVICE)
        pred = tiny(add_noise(x, noise, t), t)                 # 2. forward (sabotage + quiz)
        loss = F.mse_loss(pred, noise)                         # 3. loss
        opt_a.zero_grad(); loss.backward(); opt_a.step()       # 4+5. backward, step
        step += 1; losses_a.append(loss.item())
        if step % 200 == 0:
            print(f"step {step:5d}  loss {sum(losses_a[-200:])/200:.4f}  ({(time.time()-t0)/step:.3f}s/it)")
        if step >= STEPS_A:
            done = True; break
print(f"\\ntrained in {(time.time()-t0)/60:.1f} min")""")

md("""\
**You should see:** loss start near **1.0** (guessing pure static = average squared error of 1) and settle into the **0.04–0.07** band. Like all diffusion losses it's a fuzzy band, not a glide — each step quizzes a different image at a different noise level, and fog is intrinsically harder to predict than near-clean images. **Judge diffusion models by the images, never the loss curve.**

## A.4 Generation: the reverse walk, written out by hand

Now play the film backwards. Start from pure static; at each of the 1,000 steps, ask the UNet which way is "less noisy," take a small step that way, and add back a sprinkle of fresh randomness (the ancestral-sampling term — without it the walk collapses to a blurry average). The formula below is the DDPM update rule, unhidden.""")

code("""\
tiny.eval()
n_show = 16
x = torch.randn(n_show, 1, 28, 28, device=DEVICE)     # pure fog
snaps = {}
with torch.no_grad():
    for tt in range(T - 1, -1, -1):                    # walk t = 999 ... 0
        t = torch.full((n_show,), tt, device=DEVICE, dtype=torch.long)
        eps = tiny(x, t)                               # "which way is less noisy?"
        a_t, ac_t = alphas[tt], acum[tt]
        x = (x - (1 - a_t) / (1 - ac_t).sqrt() * eps) / a_t.sqrt()   # step toward clean
        if tt > 0:
            x = x + betas[tt].sqrt() * torch.randn_like(x)           # sprinkle of fresh randomness
        if tt in (999, 750, 500, 250, 100, 0):
            snaps[tt] = x.clone()

fig, axes = plt.subplots(len(snaps), 8, figsize=(12, 9))
for row, (tt, xs) in enumerate(sorted(snaps.items(), reverse=True)):
    imgs = ((xs.clamp(-1, 1) + 1) / 2).cpu()
    for col in range(8):
        axes[row][col].imshow(imgs[col, 0], cmap="gray", vmin=0, vmax=1)
        axes[row][col].axis("off")
        if col == 0:
            axes[row][col].set_title(f"t={tt}", fontsize=8, loc="left")
plt.suptitle("the reverse walk: fog slowly deciding to be clothes")
plt.tight_layout(); plt.show()""")

md("""\
**You should see:** static at t=999 → ghostly vertical smudges by t=250 → unmistakable **trousers, boots, shirts, and bags** at t=0. (On the 400-step CPU run: recognizable clothing shapes with rougher edges — same phenomenon, shorter training.)

That is a *generative model* — images it has never seen, invented on demand — and you wrote every line of it: the UNet, the schedule, the sabotage, the reverse walk. Hold onto two facts for the rest of the lesson:

1. **Training a denoiser = sabotage-and-quiz.** Parts B and C use the identical loss.
2. **The network conditions on `t`.** Stable Diffusion adds one more condition — *text* — and that's the lever LoRA pulls.

---
# Part B — SD1.5 ukiyo-e style LoRA (GPU track)

Stable Diffusion 1.5 in one paragraph: a **text encoder** (CLIP, Lesson 3's cousin) turns your prompt into vectors; a **UNet** (Part A's idea, 750× bigger, with cross-attention to the text) repeatedly removes noise from random fog *guided by your prompt*; a **VAE** compresses images to 64×64 "latents" so all the denoising happens in a small space, then decodes back to pixels. We freeze all of it, LoRA the UNet's attention layers (yes — `to_q`, `to_k`, `to_v` again), and fine-tune on 25 ukiyo-e prints so the phrase **"in ukiyo-e style"** rewires the output.

> **License box — read this one.**
> - **The training images** are **public domain** (Hokusai, Hiroshige, Utamaro, Sharaku, Kuniyoshi — all died in the 19th century). Full provenance in `data/ukiyoe/ATTRIBUTION.csv`.
> - **The base model**, Stable Diffusion 1.5, is **CreativeML OpenRAIL-M** — *not* a permissive license. You downloaded it yourself (via bootstrap or first run); this repo never redistributes it. OpenRAIL-M carries **use restrictions** (no illegal content, no disinformation, etc.) that **flow down to any adapter you train and share** — your LoRA file inherits them. Full accounting: `THIRD_PARTY_LICENSES.md`.
> - Part C's FLUX.1-schnell is Apache-2.0, for contrast — same code, no use restrictions on adapters.

> **Tier note:** this part needs a GPU with ≥ 4 GB VRAM (~3 GB measured). On CPU tier the cells below print a skip message — Part D has your results.""")

code("""\
import csv, os
from PIL import Image
from torch.utils.data import Dataset
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig

SD, VARIANT = "stable-diffusion-v1-5/stable-diffusion-v1-5", "fp16"
RES, BATCH, ACCUM, LR = 512, 1, 4, 1e-4
STEPS_B = 700                                  # 25 images, 28 epochs worth — style needs reps, not volume
ASSETS = Path("assets"); ASSETS.mkdir(exist_ok=True)

PROMPTS = ["a wizard with a long white beard, in ukiyo-e style",
           "a robot dog playing in a park, in ukiyo-e style"]

# ---- dataset: 25 public-domain prints, captioned from ATTRIBUTION.csv ----
DATA = ROOT / "data" / "ukiyoe"
examples = []
for r in csv.DictReader(open(DATA / "ATTRIBUTION.csv", encoding="utf-8")):
    subject = r["title"].split("(")[0].strip().lower()
    caption = f"a traditional japanese woodblock print of {subject}, in ukiyo-e style"
    examples.append((DATA / r["filename"], caption))
print(f"{len(examples)} prints. Sample captions:")
for _, c in examples[:3]:
    print("  ", c)

class Ukiyoe(Dataset):
    def __len__(self): return len(examples)
    def __getitem__(self, i):
        path, caption = examples[i]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        s = min(w, h)                                   # center-crop to square, then 512x512
        img = img.crop(((w - s)//2, (h - s)//2, (w + s)//2, (h + s)//2)).resize((RES, RES), Image.LANCZOS)
        px = (torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0) * 2 - 1   # [-1,1]
        return {"pixel_values": px, "caption": caption}

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
ds_b = Ukiyoe()
for ax, i in zip(axes, [0, 10, 20, 23]):
    ax.imshow((ds_b[i]["pixel_values"].permute(1, 2, 0) + 1) / 2)
    ax.axis("off"); ax.set_title(ds_b[i]["caption"][:42] + "...", fontsize=7)
plt.suptitle("the whole dataset: 25 woodblock prints")
plt.show()""")

md("""\
**You should see:** 25 prints loaded, captions like *"a traditional japanese woodblock print of the great wave off kanagawa, in ukiyo-e style"*.

Notice the dataset engineering: every caption names its **subject** (so the model doesn't bind "great wave" to *every* output) and ends with the constant **trigger phrase** `", in ukiyo-e style"`. The model learns to bind that phrase to what all 25 images *share* — bold outlines, flat color blocks, visible grain — not to any one subject. Afterwards the phrase is the on/off switch for the style.""")

code("""\
if TIER == "cpu":
    print("Skipped — SD1.5 LoRA needs a GPU (>=4 GB). Your results are pre-computed in Part D.")
else:
    tok = CLIPTokenizer.from_pretrained(SD, subfolder="tokenizer")

    def collate(batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "input_ids": tok([b["caption"] for b in batch], padding="max_length",
                             max_length=tok.model_max_length, truncation=True,
                             return_tensors="pt").input_ids,
        }
    dl_b = DataLoader(ds_b, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)
    print(f"dataloader ready: {len(ds_b)} images, batch {BATCH}, gradient accumulation {ACCUM}")""")

md("""\
## B.1 The frozen factory

Load the pretrained parts. Everything runs in **bfloat16** — same memory as fp16, but fp32's numeric *range*, which matters: the SD1.5 VAE famously produces **black images in pure fp16** (activation range overflow in the decoder). bf16 sidesteps that entirely.""")

code("""\
if TIER != "cpu":
    unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet", variant=VARIANT, dtype=DTYPE).to(DEVICE)
    vae = AutoencoderKL.from_pretrained(SD, subfolder="vae", variant=VARIANT, dtype=DTYPE).to(DEVICE)
    text_enc = CLIPTextModel.from_pretrained(SD, subfolder="text_encoder", variant=VARIANT, dtype=DTYPE).to(DEVICE)
    noise_sched = DDPMScheduler.from_pretrained(SD, subfolder="scheduler")
    vae.requires_grad_(False); text_enc.requires_grad_(False)    # only the LoRA (soon) will train

    def make_pipe(u):
        return StableDiffusionPipeline.from_pretrained(
            SD, variant=VARIANT, unet=u, vae=vae, text_encoder=text_enc, tokenizer=tok,
            safety_checker=None, requires_safety_checker=False, dtype=DTYPE).to(DEVICE)

    def generate(pipe, prompt, seed=42, steps=28):
        g = torch.Generator(DEVICE).manual_seed(seed)
        return pipe(prompt, num_inference_steps=steps, generator=g, guidance_scale=7.5).images[0]

    print("components loaded — UNet, VAE, CLIP text encoder, DDPM scheduler (the grown-up Part A)")""")

md("""\
## B.2 Baseline generations

Same prompts, same seed — this seed stays fixed for the whole lesson so before/after is a fair fight.""")

code("""\
if TIER != "cpu":
    pipe = make_pipe(unet)
    for i, p in enumerate(PROMPTS):
        img = generate(pipe, p, seed=42)
        img.save(ASSETS / f"sd15_before_{i}.png")
        img.save(ROOT / "runs" / f"lesson5_before_{i}.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, i, p in zip(axes, range(2), PROMPTS):
        ax.imshow(plt.imread(ASSETS / f"sd15_before_{i}.png")); ax.axis("off")
        ax.set_title(p[:44], fontsize=8)
    plt.suptitle("BEFORE — base SD1.5's vague idea of ukiyo-e")
    plt.show()""")

md("""\
**You should see:** SD1.5 *has* heard of ukiyo-e — it's one of the most famous art styles on earth, so the base model gestures at it: woodblock-ish colors, maybe a cartouche. But look closer and it's a shaky pastiche — gradients and photographic shading creep in, outlines wobble, and whether you get a print or a painting is luck-of-the-seed. Training turns that vague vibe into a style that's **consistent, deeper, and bound to your trigger phrase**.

## B.3 LoRA, one more time

`to_q/to_k/to_v/to_out.0` = the UNet's attention projections (self-attention *and* the cross-attention that reads your prompt). Same r=4 recipe as the official SD examples — image styles need less rank than you'd guess.""")

code("""\
if TIER != "cpu":
    unet.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian",
                                target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    trainable = [p for p in unet.parameters() if p.requires_grad]
    n_tr, n_all = sum(p.numel() for p in trainable), sum(p.numel() for p in unet.parameters())
    print(f"trainable: {n_tr:,} of {n_all:,} ({100*n_tr/n_all:.2f}%)")
    opt_b = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-2)""")

md("""\
**You should see:** ~0.8M trainable params out of ~860M — about 0.09%.

## B.4 THE loop — you have seen this loop before

Sabotage-and-quiz, exactly as in Part A, with two upgrades: the sabotage happens in **latent space** (the VAE compresses each print 8× first), and the quiz is **conditioned on the caption** (the cross-attention reads it). Everything between the comments is Lesson 1 wearing a trenchcoat.""")

code("""\
if TIER != "cpu":
    unet.train()
    t0, step, losses_b = time.time(), 0, []
    done = False
    while not done:
        for batch in dl_b:
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(DEVICE, DTYPE)).latent_dist.sample().detach() * 0.18215
                emb = text_enc(batch["input_ids"].to(DEVICE))[0].detach()
            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_sched.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
            noisy = noise_sched.add_noise(latents, noise, t)   # sabotage
            pred = unet(noisy, t, emb).sample                  # quiz: which noise? (hint: read the caption)
            loss = F.mse_loss(pred, noise) / ACCUM             # grade
            loss.backward()                                    # learn
            step += 1; losses_b.append(loss.item() * ACCUM)
            if step % ACCUM == 0:
                opt_b.step(); opt_b.zero_grad()
            if step % 100 == 0:
                print(f"step {step:5d}  loss {sum(losses_b[-100:])/100:.4f}  ({(time.time()-t0)/step:.2f}s/it)")
            if step >= STEPS_B:
                done = True; break
    msg = f"\\ntrained in {(time.time()-t0)/60:.1f} min"
    if DEVICE == "cuda":
        msg += f" | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB"
    print(msg)""")

md("""\
**You should see:** ~700 steps at roughly 0.25 s/it (~3 min, ~3 GB VRAM on a 16 GB card), loss wandering a fuzzy band around **0.05–0.15** — *not* a smooth glide. Each step scores a different print at a different noise level; fog is always harder to predict. Diffusion fine-tunes are judged by the **images**, and the images are next.""")

code("""\
if TIER != "cpu":
    unet.eval()                                    # eval before generating, always (Lesson 4's dropout bug)
    pipe = make_pipe(unet)
    for i, p in enumerate(PROMPTS):
        img = generate(pipe, p, seed=42)
        img.save(ASSETS / f"sd15_after_{i}.png")
        img.save(ROOT / "runs" / f"lesson5_after_{i}.png")

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for col in range(2):
        axes[0][col].imshow(plt.imread(ASSETS / f"sd15_before_{col}.png")); axes[0][col].axis("off")
        axes[0][col].set_title(f"BEFORE\\n{PROMPTS[col][:40]}", fontsize=8)
        axes[1][col].imshow(plt.imread(ASSETS / f"sd15_after_{col}.png")); axes[1][col].axis("off")
        axes[1][col].set_title("AFTER (same seed)", fontsize=8)
    plt.tight_layout(); plt.show()""")

md("""\
**You should see:** same composition seed, deepened rendering — flatter color blocks, confident contour lines, woodblock grain, a palette pulled toward indigo/ochre/vermilion. The wizard still has his beard and hat; the robot dog is still a robot dog. The base model's shaky pastiche became the real thing, on command. If your AFTER images look identical to BEFORE, train longer (or check that the trigger phrase is in the prompt).

## B.5 Ship it — 1.6 MB of style""")

code("""\
if TIER != "cpu":
    ADAPTER_B = ROOT / "runs" / "lesson5_ukiyoe_sd15_adapter"
    unet.save_lora_adapter(ADAPTER_B)
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(ADAPTER_B) for f in fs) / 1e6
    print(f"adapter: {size:.1f} MB  vs  full UNet: ~1,700 MB  ->  {ADAPTER_B.relative_to(ROOT)}")

    # reload path (how anyone would use it):
    fresh = UNet2DConditionModel.from_pretrained(SD, subfolder="unet", variant=VARIANT, dtype=DTYPE).to(DEVICE)
    fresh.load_lora_adapter(ADAPTER_B, prefix=None, weight_name="pytorch_lora_weights.safetensors")
    n_lora = sum(1 for n, _ in fresh.named_parameters() if "lora" in n.lower())
    print(f"reload works — {n_lora} LoRA tensors attached to a stock UNet")
    del fresh; torch.cuda.empty_cache()""")

md("""\
**Remember the license box:** this adapter inherits SD1.5's **OpenRAIL-M** use restrictions. Sharing it is fine (it's 1.6 MB and contains none of the base weights); use it within the license terms.

---
# Part C — Bonus: FLUX.1-schnell LoRA (gpu-high only)

Everything above, scaled to a 12-billion-parameter **rectified-flow** transformer. Two differences worth naming:

1. **Architecture:** FLUX replaces the UNet with a pure **transformer** (image patches as tokens, text tokens alongside — Lesson 2's architecture generating images). LoRA still lands on `to_q/to_k/to_v/to_out.0`.
2. **Noise math:** instead of DDPM's 1,000-step schedule, FLUX uses **flow matching** — draw a straight line from image to noise, train the model to predict the *velocity* along it (`target = noise − image`). Same sabotage-and-quiz, simpler geometry.

And the licensing contrast: FLUX.1-schnell is **Apache-2.0**. Adapters you train on it carry **no use restrictions** — compare with the OpenRAIL-M box in Part B.

**Memory trick:** 12B params doesn't fit in 16 GB at full precision, so the transformer and the T5 text encoder are loaded **quantized to 4-bit (NF4)** via bitsandbytes — ~9 GB total instead of ~34 — and the LoRA trains on top in bf16 (this combo is "QLoRA", Lesson 4's exercise 5 made real).

> **Tier note:** guarded to `gpu-high` (≥ 12 GB VRAM); on any other tier the cells print a skip message. Honest measurement from the maintainers' machine: peak VRAM during training hit **21.7 GB** — our 16 GB card only survived because Windows quietly spilled the overflow into shared system memory (and still ran 1.8 s/it). Treat **24 GB as the comfortable bar**; if you OOM, drop `RES` to 384 in Part B's dataset cell and rebuild. First run also downloads ~34 GB of weights (the cell uses `snapshot_download` so it resumes cleanly if interrupted).""")

code("""\
if TIER != "gpu-high":
    print("Skipped — needs >=12 GB VRAM; see pre-computed results below and in Part D.")
else:
    from huggingface_hub import snapshot_download
    FLUX_DIR = snapshot_download(   # no-op if already cached; resumes if interrupted
        "black-forest-labs/FLUX.1-schnell",
        allow_patterns=["model_index.json", "transformer/*", "vae/*", "text_encoder/*",
                        "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*", "scheduler/*"])
    print("weights at:", FLUX_DIR)""")

code("""\
if TIER == "gpu-high":
    from diffusers import (FluxPipeline, FluxTransformer2DModel, AutoencoderKL as FluxVAE,
                           FlowMatchEulerDiscreteScheduler, BitsAndBytesConfig as DiffBnb)
    from transformers import (BitsAndBytesConfig as HFBnb, T5EncoderModel, T5TokenizerFast)

    nf4 = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE)
    transformer = FluxTransformer2DModel.from_pretrained(
        FLUX_DIR, subfolder="transformer", quantization_config=DiffBnb(**nf4), dtype=DTYPE)
    vae_f = FluxVAE.from_pretrained(FLUX_DIR, subfolder="vae", dtype=DTYPE)
    clip = CLIPTextModel.from_pretrained(FLUX_DIR, subfolder="text_encoder", dtype=DTYPE)
    t5 = T5EncoderModel.from_pretrained(FLUX_DIR, subfolder="text_encoder_2",
                                        quantization_config=HFBnb(**nf4), dtype=DTYPE)
    tok_c = CLIPTokenizer.from_pretrained(FLUX_DIR, subfolder="tokenizer")
    tok_t5 = T5TokenizerFast.from_pretrained(FLUX_DIR, subfolder="tokenizer_2")

    pipe_f = FluxPipeline(transformer=transformer, vae=vae_f, text_encoder=clip, text_encoder_2=t5,
                          tokenizer=tok_c, tokenizer_2=tok_t5,
                          scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(FLUX_DIR, subfolder="scheduler"))
    pipe_f.enable_model_cpu_offload()      # stream layers GPU<->RAM: the 12B model fits in ~9 GB VRAM
    print("pipeline assembled — ~9 GB of 4-bit weights (from ~34 GB full-precision) stream through the GPU as needed")""")

md("""\
**You should see:** ~9 GB of quantized weights streaming through the GPU instead of ~34 GB resident. First the BEFORE images (schnell is distilled — it generates in **4 steps**, no guidance):""")

code("""\
if TIER == "gpu-high":
    def gen_flux(prompt, seed=42):
        g = torch.Generator("cpu").manual_seed(seed)
        return pipe_f(prompt, num_inference_steps=4, guidance_scale=0.0,
                      max_sequence_length=256, generator=g).images[0]

    for i, p in enumerate(PROMPTS):
        gen_flux(p).save(ASSETS / f"flux_before_{i}.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, i, p in zip(axes, range(2), PROMPTS):
        ax.imshow(plt.imread(ASSETS / f"flux_before_{i}.png")); ax.axis("off")
        ax.set_title(p[:44], fontsize=8)
    plt.suptitle("FLUX BEFORE — the base model's take at seed 42")
    plt.show()""")

md("""\
## C.1 Pre-compute the conditions, then free the text encoders

Training needs three things per image: its **latent** (VAE), its **caption embeddings** (CLIP pooled + T5 sequence), and the LoRA'd transformer. Compute the first two for all 25 prints up front, then *drop the text encoders out of VRAM* — the transformer gets the whole card for backprop.""")

code("""\
if TIER == "gpu-high":
    pipe_f.remove_all_hooks()          # training drives the transformer directly — offload hooks would fight the optimizer
    transformer.to(DEVICE)
    caps = [c for _, c in examples]
    with torch.no_grad():
        clip.to(DEVICE)
        pooled = clip(tok_c(caps, padding="max_length", max_length=77, truncation=True,
                            return_tensors="pt").input_ids.to(DEVICE)).pooler_output
        t5.to(DEVICE)
        txt_emb = t5(tok_t5(caps, padding="max_length", max_length=256, truncation=True,
                            return_tensors="pt").input_ids.to(DEVICE)).last_hidden_state
        vae_f.to(DEVICE)
        px = torch.stack([ds_b[i]["pixel_values"] for i in range(len(ds_b))]).to(DEVICE, DTYPE)
        lats = vae_f.encode(px).latent_dist.sample()
        lats = (lats - vae_f.config.shift_factor) * vae_f.config.scaling_factor
    del px
    clip.to("cpu"); t5.to("cpu"); vae_f.to("cpu"); torch.cuda.empty_cache()
    print(f"precomputed: latents {tuple(lats.shape)}, t5 embeds {tuple(txt_emb.shape)}, pooled {tuple(pooled.shape)}")

    def pack(x):                                   # B,C,H,W -> B,(H/2·W/2),C·4 : image patches as tokens
        B, C, H, W = x.shape
        return x.view(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, (H//2)*(W//2), C*4)

    img_ids = torch.zeros(32, 32, 3)
    img_ids[..., 1] = torch.arange(32)[:, None]    # each patch token knows its (row, col)
    img_ids[..., 2] = torch.arange(32)[None, :]
    img_ids = img_ids.reshape(32*32, 3).to(DEVICE)
    txt_ids = torch.zeros(256, 3).to(DEVICE)       # text tokens: no spatial position
    print("packed coordinates ready — 1,024 image tokens + 256 text tokens per example")""")

md("""\
## C.2 LoRA on the transformer + the flow-matching loop

Sabotage: pick a point `σ` on the straight line from print to noise. Quiz: *"what's the velocity?"* (`noise − image`). Grade, backward, step — the five lines, third time.""")

code("""\
if TIER == "gpu-high":
    transformer.add_adapter(LoraConfig(r=8, lora_alpha=8, init_lora_weights="gaussian",
                                       target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    trainable_f = [p for p in transformer.parameters() if p.requires_grad]
    print(f"trainable: {sum(p.numel() for p in trainable_f):,} of "
          f"{sum(p.numel() for p in transformer.parameters()):,}")
    opt_f = torch.optim.AdamW(trainable_f, lr=1e-4, weight_decay=1e-2)
    transformer.enable_gradient_checkpointing()
    transformer.train()

    STEPS_C = 300
    shift = 3.0                                    # sample more mid/high noise levels (FLUX convention)
    t0, step, losses_c = time.time(), 0, []
    rng = random.Random(42)
    while step < STEPS_C:
        i = rng.randrange(len(lats))
        lat = lats[i:i+1].to(DEVICE)
        noise = torch.randn_like(lat)
        u = torch.rand(1, device=DEVICE)
        sigma = shift * u / (1 + (shift - 1) * u)                 # shifted flow-matching timestep
        noisy = (1 - sigma[0]) * lat + sigma[0] * noise           # sabotage: point on the line
        pred = transformer(hidden_states=pack(noisy),
                           encoder_hidden_states=txt_emb[i:i+1].to(DEVICE),
                           pooled_projections=pooled[i:i+1].to(DEVICE),
                           timestep=sigma, img_ids=img_ids, txt_ids=txt_ids,
                           guidance=None).sample                  # quiz: which velocity?
        loss = F.mse_loss(pred.float(), pack(noise - lat).float()) / ACCUM   # grade
        loss.backward()                                           # learn
        step += 1; losses_c.append(loss.item() * ACCUM)
        if step % ACCUM == 0:
            opt_f.step(); opt_f.zero_grad()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {sum(losses_c[-50:])/50:.4f}  ({(time.time()-t0)/step:.2f}s/it)")
    msg = f"\\ntrained in {(time.time()-t0)/60:.1f} min"
    if DEVICE == "cuda":
        msg += f" | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB"
    print(msg)""")

md("""\
**You should see:** ~1.8 s/it through a 12B model — 4-bit weights mean most math runs at LoRA speed. Loss fell 0.64 → 0.44 on the maintainers' run; same fuzzy-band disclaimer as Part B: judge by images.""")

code("""\
if TIER == "gpu-high":
    transformer.eval()
    clip.to(DEVICE); t5.to(DEVICE); vae_f.to(DEVICE)   # back from exile for the AFTER images
    for i, p in enumerate(PROMPTS):
        gen_flux(p).save(ASSETS / f"flux_after_{i}.png")

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for col in range(2):
        axes[0][col].imshow(plt.imread(ASSETS / f"flux_before_{col}.png")); axes[0][col].axis("off")
        axes[0][col].set_title(f"BEFORE\\n{PROMPTS[col][:40]}", fontsize=8)
        axes[1][col].imshow(plt.imread(ASSETS / f"flux_after_{col}.png")); axes[1][col].axis("off")
        axes[1][col].set_title("AFTER (same seed)", fontsize=8)
    plt.tight_layout(); plt.show()

    ADAPTER_C = ROOT / "runs" / "lesson5_ukiyoe_flux_adapter"
    # note: transformer.save_lora_adapter() trips on 4-bit models in this diffusers version
    # (it tries to .to() the quantized base) — so we save the peft state dict directly
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file
    ADAPTER_C.mkdir(parents=True, exist_ok=True)
    sd_f = get_peft_model_state_dict(transformer, adapter_name="default")
    save_file(sd_f, ADAPTER_C / "pytorch_lora_weights.safetensors")
    size_c = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(ADAPTER_C) for f in fs) / 1e6
    print(f"\\nadapter: {size_c:.1f} MB -> {ADAPTER_C.relative_to(ROOT)}  (Apache-2.0 — yours, no strings)")""")

md("""\
---
# Part D — Pre-computed results & using the adapters (everyone)

Whether or not your hardware ran Parts B–C, the payoff is committed to this repo. These four images were generated by the exact cells above, on the maintainers' machine, at seed 42:""")

code("""\
fig, axes = plt.subplots(2, 2, figsize=(9, 9))
for col in range(2):
    for row, stage in enumerate(["before", "after"]):
        f = ASSETS / f"sd15_{stage}_{col}.png"
        axes[row][col].imshow(plt.imread(f)); axes[row][col].axis("off")
        axes[row][col].set_title(f"{stage.upper()} — {PROMPTS[col][:38]}", fontsize=8)
plt.suptitle("SD1.5 + 25 public-domain prints + 700 LoRA steps")
plt.tight_layout(); plt.show()
print("FLUX before/after pairs are committed in the same assets/ folder (flux_before_*, flux_after_*).")""")

md("""\
**What you're looking at:** left column, a wizard; right column, a robot dog. Top row is stock SD1.5's *approximation* of ukiyo-e — the vibe is there, but so are photographic shading and wobbly outlines. Bottom row is the *same seed, same prompt* after 700 LoRA steps on 25 prints: flatter color blocks, confident contour lines, woodblock grain, cartouches on cue. Composition comes from the seed; the deepened, consistent style comes from the adapter.

## Loading the pre-trained adapters yourself

The trained adapters are published on the course's GitHub Releases page (1.6 MB — vs 4 GB for the base model). On **any** hardware, including CPU-tier laptops, this is the full recipe:

```python
# one-time: download the adapter from the Releases page
import urllib.request, zipfile
url = "https://github.com/Shanmuga-Raja-Portfolio/ml-workshop/releases/download/lesson5/lesson5_ukiyoe_sd15_adapter.zip"
urllib.request.urlretrieve(url, "sd15_adapter.zip")
zipfile.ZipFile("sd15_adapter.zip").extractall("runs/lesson5_ukiyoe_sd15_adapter")

# then: stock SD1.5 + your 1.6 MB style, streamed to fit even small GPUs
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
import torch
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", variant="fp16",
    dtype=torch.bfloat16, safety_checker=None, requires_safety_checker=False)
# the adapter file stores bare UNet keys; the pipeline router wants them under "unet."
sd = load_file("runs/lesson5_ukiyoe_sd15_adapter/pytorch_lora_weights.safetensors")
pipe.load_lora_weights({"unet." + k: v for k, v in sd.items()})
pipe.enable_model_cpu_offload()          # layers stream GPU<->RAM; works on 4 GB cards
img = pipe("a wizard with a long white beard, in ukiyo-e style",
           num_inference_steps=28, guidance_scale=7.5).images[0]
```

(OpenRAIL-M use restrictions travel with the adapter — Part B's license box.) If you ran Part B on this machine, the cell below does the same thing against your *own* freshly trained adapter:""")

code("""\
ADAPTER_B = ROOT / "runs" / "lesson5_ukiyoe_sd15_adapter"
if not ADAPTER_B.exists():
    print("No local adapter — either run Part B on a GPU, or download the pre-trained one (cell above).")
else:
    pipe_x = StableDiffusionPipeline.from_pretrained(
        SD, variant=VARIANT, dtype=DTYPE,
        safety_checker=None, requires_safety_checker=False)
    from safetensors.torch import load_file
    sd_keys = load_file(ADAPTER_B / "pytorch_lora_weights.safetensors")
    pipe_x.load_lora_weights({"unet." + k: v for k, v in sd_keys.items()})   # the one-line attach
    if DEVICE == "cuda":
        pipe_x.enable_model_cpu_offload()
    else:
        pipe_x = pipe_x.to(DEVICE)
    img = pipe_x("a lighthouse at sunrise, in ukiyo-e style",
                 num_inference_steps=28, guidance_scale=7.5,
                 generator=torch.Generator("cpu").manual_seed(7)).images[0]
    plt.figure(figsize=(5, 5)); plt.imshow(img); plt.axis("off")
    plt.title("a lighthouse at sunrise, in ukiyo-e style — from the reloaded adapter")
    plt.show()""")

md("""\
**You should see:** a brand-new subject (never in training) wearing the woodblock style — proof the LoRA learned *style*, not pictures.

## Exercises""")

code("""\
# 1. SWITCH: generate with prompts that OMIT ', in ukiyo-e style' - how much style
#    bleeds in anyway? Trigger phrases are a dial, not a switch.
#
# 2. YOUR OWN STYLE: any folder of 25-200 consistent images (your sketches, an
#    artist you like who is comfortably public domain) + the caption pattern from
#    Part B = a style LoRA pipeline you own end to end.
#
# 3. KNOBS: r=8 / lora_alpha=8, or LR 5e-5 vs 2e-4. Retrain 300 steps each and
#    compare the wizard at fixed seed. Develop your eye.
#
# 4. STRENGTH: pipe(prompt, cross_attention_kwargs={'scale': 0.5}) tones the
#    adapter down at inference - LoRA effects are dial-able AFTER training.
#
# 5. PART A DEEPER: class-condition the tiny UNet (add the Fashion-MNIST label to
#    the time embedding) so you can ask for 'a boot' specifically. That's
#    classifier-free conditioning - the same trick the text encoder plays in Part B.

print("Your turn.")""")

md("""\
# Course complete

Five lessons, one loop:

| Lesson | Model | Params you trained | Method |
|---|---|---|---|
| 1 | tiny MLP | 4,353 | from scratch, by hand |
| 2 | tiny GPT | 2.7M | from scratch, by hand |
| 3 | DistilBERT | 67M | full fine-tune (HF Trainer) |
| 4 | Qwen3-1.7B | 6.4M (0.37%) | LoRA via PEFT/TRL |
| 5A | tiny UNet | 1.1M | diffusion from scratch, by hand |
| 5B | Stable Diffusion 1.5 | 0.8M (0.09%) | LoRA, manual loop |
| 5C | FLUX.1-schnell (4-bit) | 9.3M (0.16%) | QLoRA, flow matching |

Every single one was: **data → forward → loss → backward → step.** The differences were only *what* the data was, *how big* the model was, and *which* parameters moved. That's the whole field, honestly told.

Where to go next: `ROADMAP.md` has the pointers (unsloth for faster/larger LoRA runs, DPO for preference tuning, the HF courses, r/LocalLLaMA scene). Go break things.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3 (ml-training)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path("lessons/04_diffusion_lora/lesson5_style_lora.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "with", len(cells), "cells")
