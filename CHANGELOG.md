# Changelog

All notable changes to this course are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows SemVer.

## [1.0.0] - 2026-10-24

Initial public release.

### Added
- Five-lesson course: from-scratch foundations (tiny net, tiny GPT), DistilBERT
  fine-tuning, Qwen3 LoRA, diffusion from scratch + style LoRA (SD1.5 main track,
  FLUX.1-schnell bonus track).
- One-command setup: `tools/bootstrap.py` (hardware detection, per-platform PyTorch
  builds, manifest-driven downloads with license notices).
- Hardware tiers: every lesson runs on CPU-only machines, scaling automatically.
- License infrastructure: MIT for the course; `THIRD_PARTY_LICENSES.md`;
  `tools/license_audit.py`; runtime-download-only policy for non-permissive models.
- Community layer: CONTRIBUTING (DCO sign-off, no CLA), CODE_OF_CONDUCT, SECURITY,
  issue/PR templates, CI gates.
- Docs: PORTABILITY (incl. the Vulkan explanation), TROUBLESHOOTING/FAQ, ARCHITECTURE.

### Removed (pre-release scrub)
- All references to the copyrighted development dataset; replaced with public-domain
  ukiyo-e artwork (with attribution).
- All personal paths, usernames, and machine-specific assumptions.
