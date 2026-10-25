#!/usr/bin/env python3
"""Fetch all runtime downloads for the ML workshop.

Driven by downloads/manifest.json. Requires project deps (huggingface_hub,
requests, torchvision) - run `python tools/bootstrap.py` first, or invoke via
`uv run python tools/fetch_data.py ...`.

Usage:
    uv run python tools/fetch_data.py [all|ukiyoe|models] [--with-optional]

Behavior:
  - Every item prints its license line from the manifest before downloading.
  - OpenRAIL-licensed items (e.g. Stable Diffusion 1.5) print a prominent
    use-restriction notice BEFORE any bytes are fetched.
  - Per-item retry: up to 5 attempts, 30s backoff. huggingface_hub resumes
    partial downloads, so retries are cheap.
  - Optional manifest items (e.g. FLUX.1-schnell, ~24GB) are skipped unless
    --with-optional is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "downloads" / "manifest.json"
UKIYOE_CSV = REPO_ROOT / "downloads" / "ukiyoe_sources.csv"
UKIYOE_DIR = REPO_ROOT / "data" / "ukiyoe"
DATA_DIR = REPO_ROOT / "data"

MAX_ATTEMPTS = 5
BACKOFF_S = 30
USER_AGENT = "ml-workshop-course/1.0 (educational; github.com/Shanmuga-Raja-Portfolio)"

OPENRAIL_NOTICE = """\
========================================================================
  LICENSE NOTICE - CreativeML OpenRAIL-M
========================================================================
  The next download ({ref}) is NOT under a plain permissive license.
  CreativeML OpenRAIL-M attaches USE RESTRICTIONS (a list of prohibited
  uses) that legally FLOW DOWN to anything derived from the model -
  including the LoRA adapters you train in lesson 5 and share.

  Full license text:
  https://huggingface.co/spaces/CompVis/stable-diffusion-license

  Continuing means you accept those terms.
========================================================================"""


def log(msg: str) -> None:
    print(f"[fetch_data] {msg}", flush=True)


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def print_license_line(item: dict) -> None:
    log(f"{item['id']}: {item['ref']} - license: {item['license']} ({item['license_url']})")


def maybe_openrail_notice(item: dict) -> None:
    if "openrail" in item.get("license", "").lower():
        print(OPENRAIL_NOTICE.format(ref=item["ref"]), flush=True)


def run_with_retry(name: str, fn) -> bool:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            fn()
            log(f"{name}: DONE")
            return True
        except Exception as exc:  # noqa: BLE001 - report and retry anything
            log(f"{name}: attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}")
            if attempt < MAX_ATTEMPTS:
                log(f"{name}: sleeping {BACKOFF_S}s, then retrying "
                    "(partial progress is kept)...")
                time.sleep(BACKOFF_S)
    log(f"{name}: FAILED after {MAX_ATTEMPTS} attempts.")
    log(f"{name}: manual fallback - see manifest: {name}")
    return False


# ------------------------------------------------------------ item handlers


def fetch_hf(item: dict) -> None:
    from huggingface_hub import snapshot_download

    kwargs: dict = {"repo_id": item["ref"], "max_workers": 4}
    if item["type"] == "hf-dataset":
        kwargs["repo_type"] = "dataset"
    if item["id"] == "stable-diffusion-v1-5":
        # Lesson 5 loads everything with variant="fp16"; grab configs plus
        # fp16 safetensors only and skip the fp32 .ckpt/.bin duplicates.
        kwargs["allow_patterns"] = ["model_index.json", "**/*.json", "**/*.txt",
                                    "**/*.fp16.safetensors"]
    path = snapshot_download(**kwargs)
    log(f"  cached at {path}")


def fetch_torchvision(item: dict) -> None:
    from torchvision import datasets

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for train in (True, False):
        datasets.FashionMNIST(root=str(DATA_DIR), train=train, download=True)
    log(f"  stored under {DATA_DIR / 'FashionMNIST'}")


def _thumb_url(file_url: str, width: int = 1280) -> str | None:
    """Wikimedia thumbnail URL served via commons.wikimedia.org/w/thumb.php.

    Originals are museum scans up to ~50MB; for training (SD1.5 runs at 512px)
    a ~1280px render is plenty, and pulling 25 originals is exactly what trips
    upload.wikimedia.org's Varnish rate limiter. Direct thumb URLs on
    upload.wikimedia.org only allow fixed widths (https://w.wiki/GHai) and are
    aggressively rate-limited; thumb.php on the API host is far more tolerant
    and serves the rendered bytes directly.
    """
    from urllib.parse import quote, unquote

    marker = "/wikipedia/commons/"
    if marker not in file_url:
        return None
    name = unquote(file_url.rsplit("/", 1)[1])
    return f"https://commons.wikimedia.org/w/thumb.php?f={quote(name)}&width={width}"


def _download_one_image(row: dict, dest: Path) -> None:
    """Single image download with Retry-After-aware 429 handling.

    Tries the 1280px thumbnail first, falls back to the original file on any
    4xx (thumbs wider than the original, or non-thumbable files, error out).
    """
    import requests

    candidates = [u for u in (_thumb_url(row["file_url"]), row["file_url"]) if u]
    url_idx = 0
    delays = [0, 15, 30, 60, 120]
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            time.sleep(delay)
        resp = requests.get(candidates[url_idx], headers={"User-Agent": USER_AGENT},
                            stream=True, timeout=180)
        if resp.status_code in (400, 403, 404) and url_idx == 0:
            url_idx = len(candidates) - 1  # switch to the original file
            continue
        if resp.status_code == 429:
            # Wikimedia sends Retry-After (typically 600s once tripped);
            # honor it instead of hammering with short backoffs.
            retry_after = resp.headers.get("Retry-After")
            wait = min(int(retry_after), 300) if retry_after and retry_after.isdigit() else 60
            log(f"  {row['filename']}: HTTP 429 (rate limited), waiting {wait}s "
                f"(attempt {attempt}/5)")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if not ctype.startswith("image/"):
            raise RuntimeError(f"{row['filename']}: unexpected content-type {ctype!r}")
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
        return
    raise RuntimeError(f"{row['filename']}: still rate-limited after 5 attempts")


def fetch_ukiyoe(item: dict | None = None) -> bool:
    """Download public-domain ukiyo-e prints and write ATTRIBUTION.csv."""
    UKIYOE_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(UKIYOE_CSV.open(encoding="utf-8")))
    log(f"ukiyoe: {len(rows)} prints listed in {UKIYOE_CSV.name}")

    attribution_rows = []
    for row in rows:
        dest = UKIYOE_DIR / row["filename"]
        if dest.exists() and dest.stat().st_size > 0:
            log(f"  {row['filename']}: already present, skipping")
        else:
            _download_one_image(row, dest)
            log(f"  {row['filename']}: {dest.stat().st_size // 1024} KB")
            time.sleep(5.0)  # politeness: Wikimedia asks bots to go slow
        attribution_rows.append({
            "filename": row["filename"],
            "title": row["title"],
            "artist": row["artist"],
            "source_url": row["commons_url"],
            "license": row["license"],
        })

    attr_path = UKIYOE_DIR / "ATTRIBUTION.csv"
    with attr_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "title", "artist",
                                                "source_url", "license"])
        writer.writeheader()
        writer.writerows(attribution_rows)
    log(f"  wrote {attr_path.relative_to(REPO_ROOT)} ({len(attribution_rows)} rows)")
    return True


# ------------------------------------------------------------------ driver


def selected_items(manifest: list[dict], subset: str, with_optional: bool) -> list[dict]:
    if subset == "ukiyoe":
        kinds = {"url-list"}
    elif subset == "models":
        kinds = {"hf-model", "hf-dataset", "torchvision-dataset"}
    else:  # all
        kinds = {"hf-model", "hf-dataset", "torchvision-dataset", "url-list"}
    items = [it for it in manifest if it["type"] in kinds]
    skipped = [it for it in items if it.get("optional") and not with_optional]
    for it in skipped:
        log(f"skipping optional item {it['id']} ({it['size_hint']}) - pass --with-optional to include")
    return [it for it in items if not it.get("optional") or with_optional]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subset", nargs="?", default="all",
                        choices=["all", "ukiyoe", "models"],
                        help="what to download (default: all)")
    parser.add_argument("--with-optional", action="store_true",
                        help="include optional manifest items (e.g. FLUX.1-schnell, ~24GB)")
    args = parser.parse_args()

    manifest = load_manifest()
    items = selected_items(manifest, args.subset, args.with_optional)
    log(f"{len(items)} item(s) to fetch (subset={args.subset})")

    failures = []
    for item in items:
        print_license_line(item)
        maybe_openrail_notice(item)
        if item["type"] in ("hf-model", "hf-dataset"):
            ok = run_with_retry(item["id"], lambda it=item: fetch_hf(it))
        elif item["type"] == "torchvision-dataset":
            ok = run_with_retry(item["id"], lambda it=item: fetch_torchvision(it))
        elif item["type"] == "url-list":
            ok = run_with_retry(item["id"], lambda it=item: fetch_ukiyoe(it))
        else:
            log(f"{item['id']}: unknown type {item['type']!r}, skipping")
            ok = False
        if not ok:
            failures.append(item)

    if failures:
        log("FAILED items - manual instructions:")
        for it in failures:
            print(f"  * {it['id']}: {it['fallback']}", flush=True)
        sys.exit(1)
    log("all downloads complete.")


if __name__ == "__main__":
    main()
