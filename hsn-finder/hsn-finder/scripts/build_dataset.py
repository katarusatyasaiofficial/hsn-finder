"""Build the cleaned HSN dataset used by the app.

Input : raw/hsn_codes_source.json  (12,604 HSN codes + descriptions, taken from
        the open-source `hsn-code-package` npm package, ISC licence, which
        bundles the CBIC / WCO Harmonized System nomenclature)
Output: data/hsn_codes.csv

What this script does
---------------------
1. Left-pads codes to 8 digits (the source drops the leading zero for
   chapters 01-09, e.g. "1011010" -> "01011010").
2. Derives the hierarchy: chapter (2 digit), heading (4 digit), subheading
   (6 digit).
3. Splits each official description into its hierarchy levels ("path") so the
   UI can show a breadcrumb of why a code matches.
4. Normalises whitespace in descriptions.

NOTE: GST rates are deliberately NOT carried over. The source's rates are
flat chapter-level approximations and still contain the pre-Sept-2025 slabs
(12%, 28%), so they are not reliable per-code rates.

Run:  python scripts/build_dataset.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsn_finder.chapters import chapter_title  # noqa: E402

SRC = ROOT / "raw" / "hsn_codes_source.json"
OUT = ROOT / "data" / "hsn_codes.csv"


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    # source sometimes has "ANIMAL S" style broken words; keep as is, but fix
    # spacing around separators for readability.
    s = re.sub(r"\s*:\s*", " : ", s)
    s = re.sub(r"\s*;\s*", "; ", s)
    return re.sub(r"\s+", " ", s).strip()


LEVEL_SPLIT = re.compile(r"\s+-+\s*|\s*-{2,}\s*|\s+:\s+")


def to_path(desc: str) -> str:
    """Split an official description into its hierarchy levels ("A > B > C")."""
    parts = [p.strip(" -:") for p in LEVEL_SPLIT.split(desc)]
    parts = [p for p in parts if p]
    # drop immediately repeated levels
    out = []
    for p in parts:
        if not out or out[-1].lower() != p.lower():
            out.append(p)
    return " > ".join(out)


def main() -> None:
    raw = json.loads(SRC.read_text())
    rows = []
    for r in raw:
        code = str(r["code"]).strip().zfill(8)
        desc = clean_text(r["description"])
        rows.append({"code": code, "description": desc})
    df = pd.DataFrame(rows).drop_duplicates("code").sort_values("code").reset_index(drop=True)

    df["chapter"] = df["code"].str[:2]
    df["heading"] = df["code"].str[:4]
    df["subheading"] = df["code"].str[:6]
    df["chapter_title"] = df["chapter"].map(chapter_title)

    df["path"] = df["description"].map(to_path)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} codes -> {OUT.relative_to(ROOT)}")
    print(f"Chapters: {df['chapter'].nunique()}  Headings: {df['heading'].nunique()}  "
          f"Subheadings: {df['subheading'].nunique()}")


if __name__ == "__main__":
    main()
