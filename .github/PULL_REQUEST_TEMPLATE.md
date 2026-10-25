## What does this PR do?


## Checklist (CI enforces most of these)

- [ ] My commits are signed off (`git commit -s` — DCO, no CLA; see CONTRIBUTING.md)
- [ ] `uv run python tools/license_audit.py` passes (any new dependency is permissively licensed)
- [ ] `uv run python tools/check_repo.py --regen-check` passes (notebooks regenerated, clean outputs, no forbidden strings)
- [ ] New models/datasets are manifest entries, not committed files
- [ ] Lessons I touched still run on the `cpu` tier (or I explained why not)
- [ ] "You should see" claims in any lesson I touched are numbers I actually observed
- [ ] Docs updated (README / ROADMAP / docs/*) if behavior changed

## Anything reviewers should know?
