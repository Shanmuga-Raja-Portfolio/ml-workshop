# Security Policy

## Scope

This is an educational repository: notebooks, Python tooling, and documentation.
The realistic risks here are:

- A committed secret (API key, token) — ours or a contributor's.
- A malicious or tampered download URL in `downloads/manifest.json`.
- A dependency with a known vulnerability.
- Code in a lesson that does something beyond what its text claims.

## Reporting a vulnerability

**Please do not open a public issue for security reports.**

Use GitHub's private reporting: this repository's **Security tab →
"Report a vulnerability"**. You should receive an acknowledgment within a few days.

If the issue is in a third-party model or dataset we download (not our code),
report it to the upstream project (e.g. Hugging Face hub) and feel free to also
tell us so we can update the manifest.

## Supported versions

Only the latest release and `main` receive fixes. This is a course, not a
service — pull the latest release and you're current.

## What we commit to

- Secrets committed to this repo are treated as compromised: revoked first,
  investigated second.
- The license audit (`tools/license_audit.py`) and repo hygiene checks
  (`tools/check_repo.py`) run in CI on every PR and release.
