# Contributing to ML Workshop

First: thank you. This course exists so anyone can learn ML for free — contributions
that make it clearer, more correct, or more accessible are the whole point.

## The hard rules (non-negotiable)

1. **License cleanliness.** Only MIT / BSD / Apache-2.0 / CC0 / public-domain material
   may be committed to this repo. No exceptions, no "it's probably fine".
   - New Python dependency? `uv run python tools/license_audit.py` must pass.
   - New model or dataset? It goes in `downloads/manifest.json` (downloaded by the user
     at first run), **never** committed — unless it's public domain or CC0 *and* small.
   - Use-restricted licenses (OpenRAIL, non-commercial, research-only) are allowed
     **only** as user-downloaded manifest entries with the license clearly documented.
2. **No CLA, ever.** We will never ask you to sign a Contributor License Agreement or
   assign your copyright. Your contribution stays yours. Instead we use the
   [Developer Certificate of Origin](https://developercertificate.org/) — sign off your
   commits with `git commit -s` (adds `Signed-off-by: Your Name <email>`), which simply
   certifies you wrote it or have the right to submit it under this project's license.
3. **No personal traces.** No absolute paths from your machine, no usernames, no API
   keys. CI greps for these and fails.
4. **Notebooks ship clean.** Zero outputs, `execution_count: null`. CI enforces this.

## How the repo works (5-minute version)

Full details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The short version:

- **Lessons 2–5 are GENERATED.** Edit `tools/build_lessonN.py`, then regenerate:
  `uv run python tools/build_lessonN.py`. Never edit those `.ipynb` files directly —
  CI regenerates them and fails if they drift.
- Lesson 1 has no generator (historical reasons) — edit its JSON carefully by hand.
- Hardware adaptation goes through `workshop_config.json` (written by
  `tools/bootstrap.py`). Every notebook must run on the `cpu` tier without error, even
  if at reduced scale. See the config contract in ARCHITECTURE.md.

## Making a change

1. Fork, branch, hack.
2. Run the gates locally (same as CI):
   ```bash
   uv run python tools/license_audit.py          # dependency licenses
   uv run python tools/check_repo.py             # scrub greps + clean notebooks
   uv run python tools/check_repo.py --regen-check   # + generator sync (CI parity)
   ```
3. If you changed lesson behavior, run the affected notebook(s) end-to-end
   (`uv run jupyter nbconvert --to notebook --execute --inplace <path>`), confirm the
   "You should see" claims still hold, then **restore the notebook from its generator**
   (`uv run python tools/build_lessonN.py`) before committing — never nbconvert-clear,
   that reformats the JSON and breaks generator sync.
4. Commit with `-s` (DCO sign-off), open a PR, fill in the template.

### The pyproject.toml wart (read before your first commit)

`tools/bootstrap.py` appends a marked machine-local PyTorch block to your local
`pyproject.toml` (that's how each machine gets the right torch wheel). **Never commit
that block** — CI hard-fails on it. When staging changes, either restore the file
(`git checkout pyproject.toml`) before committing, or strip everything between the
`--- BEGIN/END machine-local PyTorch config ---` markers. Bootstrap re-adds it
automatically the next time you run anything.

## What makes a good contribution here

- **Clarity over cleverness.** The audience is a beginner at 11pm. If a sentence needs
  jargon, teach the jargon or drop it.
- **Every claim is verified.** "You should see loss ≈ X" must be a number someone
  actually observed. If you can't verify it, say so in the text.
- **Low-spec first.** A change that only works on a $2000 GPU is a regression. Every
  lesson needs a CPU or ≤4GB-VRAM path, or shipped pre-computed results.
- **Small PRs.** One lesson, one fix, one idea.

## Releases

Maintainer cuts releases from `main`: version bump in `pyproject.toml`, entry in
`CHANGELOG.md`, git tag `vX.Y.Z`, GitHub Release with any pre-trained adapter artifacts
(each adapter carries its own license label — see `THIRD_PARTY_LICENSES.md`).

## Code of Conduct

Applies to all project spaces: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Be kind;
everyone here is learning.

## Questions?

Open an issue with the `question` label, or see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
