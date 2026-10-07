"""Evaluate the matcher on the hand-labelled query sets.

Ground truth is at the 4-digit HS *heading* level (the level the HS
nomenclature defines most firmly). A query counts as correct if a returned
code falls inside any accepted heading.

Metrics
-------
top1_heading   : rank-1 code is in an accepted heading
top5_heading   : any of the top-5 codes is in an accepted heading
top5_distinct  : any of the 5 best *distinct headings* is accepted
top1_chapter   : rank-1 code is in an accepted heading's 2-digit chapter

Usage
-----
    python eval/evaluate.py --ablation         # dev + test table for all configs
    python eval/evaluate.py --set test         # final config on the test set
    python eval/evaluate.py --set dev --show-misses
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsn_finder.matcher import Config, HSNMatcher  # noqa: E402

EVAL_DIR = ROOT / "eval"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(EVAL_DIR / f"{name}_queries.csv", dtype=str)


def evaluate(matcher: HSNMatcher, qdf: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    for r in qdf.itertuples():
        accepted = set(r.accepted_headings.split("|"))
        acc_ch = {h[:2] for h in accepted}
        res = matcher.search(r.query, k=5)
        heads = [m.heading for m in res]
        distinct = [h for h, _ in matcher.top_headings(r.query, 5)]
        rows.append(
            {
                "query": r.query,
                "accepted": r.accepted_headings,
                "top1_heading": bool(heads[:1]) and heads[0] in accepted,
                "top5_heading": any(h in accepted for h in heads),
                "top5_distinct": any(h in accepted for h in distinct),
                "top1_chapter": bool(heads[:1]) and heads[0][:2] in acc_ch,
                "got_top1": heads[0] if heads else "-",
                "got_distinct": ",".join(distinct),
                "score1": res[0].score if res else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    metrics = {
        "n": len(out),
        "top1_heading": out["top1_heading"].mean(),
        "top5_heading": out["top5_heading"].mean(),
        "top5_distinct": out["top5_distinct"].mean(),
        "top1_chapter": out["top1_chapter"].mean(),
    }
    return metrics, out


CONFIGS = {
    "A. word TF-IDF only": Config(char_weight=0.0, heading_weight=0.0, use_synonyms=False, spell_fix=False),
    "B. + char n-grams": Config(char_weight=0.3, heading_weight=0.0, use_synonyms=False, spell_fix=False),
    "C. + heading rollup": Config(char_weight=0.3, heading_weight=0.3, use_synonyms=False, spell_fix=False),
    "D. + synonyms": Config(char_weight=0.3, heading_weight=0.3, use_synonyms=True, spell_fix=False),
    "E. + spelling correction (FINAL)": Config(char_weight=0.3, heading_weight=0.3, use_synonyms=True, spell_fix=True),
    "F. E + head-noun boost": Config(char_weight=0.3, heading_weight=0.3, use_synonyms=True, spell_fix=True, head_boost=1),
}
FINAL = "E. + spelling correction (FINAL)"


def add_typo(q: str, rng) -> str:
    """Corrupt the longest word of a query with one random edit."""
    toks = q.split()
    idx = max(range(len(toks)), key=lambda i: len(toks[i]))
    w = toks[idx]
    if len(w) < 5:
        return q
    op = rng.choice(["del", "swap", "dup", "sub"])
    i = rng.randrange(1, len(w) - 1)
    if op == "del":
        w = w[:i] + w[i + 1:]
    elif op == "swap":
        w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    elif op == "dup":
        w = w[:i] + w[i] + w[i:]
    else:
        w = w[:i] + rng.choice("aeiou") + w[i + 1:]
    toks[idx] = w
    return " ".join(toks)


def typo_eval(matcher: HSNMatcher, qdf: pd.DataFrame, seeds: int = 5) -> dict:
    """Average accuracy over several random typo corruptions of every query."""
    import random

    t1, t5 = [], []
    for seed in range(seeds):
        rng = random.Random(seed)
        noisy = qdf.copy()
        noisy["query"] = [add_typo(q, rng) for q in qdf["query"]]
        m, _ = evaluate(matcher, noisy)
        t1.append(m["top1_heading"])
        t5.append(m["top5_heading"])
    return {"top1_heading": sum(t1) / seeds, "top5_heading": sum(t5) / seeds}


def fmt(m: dict) -> str:
    return (
        f"{m['top1_heading']*100:5.1f}% | {m['top5_heading']*100:5.1f}% | "
        f"{m['top5_distinct']*100:5.1f}% | {m['top1_chapter']*100:5.1f}%"
    )


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    import math

    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d, (c + a) / d


def write_report() -> None:
    """Regenerate eval/RESULTS.md from scratch (fully reproducible)."""
    import io
    from contextlib import redirect_stdout

    from hsn_finder.synonyms import expand_query

    dev, test, holdout = load("dev"), load("test"), load("holdout")
    sets = {"dev": dev, "test": test, "holdout": holdout}
    final = HSNMatcher.from_csv(config=CONFIGS[FINAL])
    base = HSNMatcher.from_csv(config=CONFIGS["C. + heading rollup"])

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("# Evaluation results\n")
        print("Auto-generated by `python eval/evaluate.py --report`. See README for methodology.\n")
        print("## 1. Ablation (clean queries)\n")
        print("Top-1 heading = rank-1 code is in an accepted 4-digit heading; Top-5 codes = any of the 5 returned codes is;")
        print("Top-5 distinct headings = any of the 5 best distinct headings is; Top-1 chapter = rank-1 is in the right 2-digit chapter.\n")
        evaluate_ablation(sets)
        print("\n## 2. Typo robustness\n")
        print("Every query gets one random edit (delete / swap / duplicate / substitute a letter) in its longest word; averaged over 5 seeds.\n")
        evaluate_typos(sets)
        print("\n## 3. Does the synonym table cover the queries?\n")
        print("Accuracy split by whether the query contains at least one term from the synonym table.\n")
        print("| Set | Covered by table | n | Final top-1 | Final top-5 headings | No-synonym top-1 | No-synonym top-5 headings |")
        print("|---|---|---|---|---|---|---|")
        for sname, q in sets.items():
            cov = [expand_query(x) != x for x in q["query"]]
            _, d = evaluate(final, q)
            _, db = evaluate(base, q)
            d["cov"], db["cov"] = cov, cov
            for lab, flag in (("yes", True), ("no", False)):
                sub, subb = d[d["cov"] == flag], db[db["cov"] == flag]
                if len(sub):
                    print(f"| {sname} | {lab} | {len(sub)} | {sub.top1_heading.mean()*100:.1f}% | "
                          f"{sub.top5_distinct.mean()*100:.1f}% | {subb.top1_heading.mean()*100:.1f}% | "
                          f"{subb.top5_distinct.mean()*100:.1f}% |")
        print("\n## 4. Final config on the holdout set\n")
        m, d = evaluate(final, holdout)
        k1, k5 = int(d.top1_heading.sum()), int(d.top5_distinct.sum())
        lo1, hi1 = wilson(k1, len(d))
        lo5, hi5 = wilson(k5, len(d))
        print(f"- n = {len(d)}")
        print(f"- Top-1 heading: **{m['top1_heading']*100:.1f}%** (95% CI {lo1*100:.0f}-{hi1*100:.0f}%)")
        print(f"- Top-5 codes: **{m['top5_heading']*100:.1f}%**")
        print(f"- Top-5 distinct headings: **{m['top5_distinct']*100:.1f}%** (95% CI {lo5*100:.0f}-{hi5*100:.0f}%)")
        print(f"- Top-1 chapter: **{m['top1_chapter']*100:.1f}%**\n")
        print("Rank-1 misses on holdout:\n")
        print("| Query | Accepted | Got (top-5 distinct headings) |")
        print("|---|---|---|")
        for r in d[~d.top1_heading].itertuples():
            print(f"| {r.query} | {r.accepted.replace(chr(124), chr(47))} | {r.got_distinct} |")
        print("\n## 5. Score calibration (all three sets pooled, final config)\n")
        rows = []
        for q in sets.values():
            _, dd = evaluate(final, q)
            rows += list(zip(dd["score1"], dd["top1_heading"], dd["top5_distinct"]))
        import pandas as pd

        a = pd.DataFrame(rows, columns=["s1", "ok1", "ok5"])
        print("| Top score | n | Rank-1 correct | Right heading in top-5 |")
        print("|---|---|---|---|")
        for lo, hi, lab in [(0, 0.45, "< 0.45 (Low)"), (0.45, 0.6, "0.45-0.60 (Medium)"), (0.6, 9, ">= 0.60 (High)")]:
            sub = a[(a.s1 >= lo) & (a.s1 < hi)]
            if len(sub):
                print(f"| {lab} | {len(sub)} | {sub.ok1.mean()*100:.1f}% | {sub.ok5.mean()*100:.1f}% |")
    (EVAL_DIR / "RESULTS.md").write_text(buf.getvalue())
    print(f"Wrote {EVAL_DIR / 'RESULTS.md'}")


def evaluate_ablation(sets: dict) -> None:
    print("| Config | Set | n | Top-1 heading | Top-5 codes | Top-5 distinct headings | Top-1 chapter |")
    print("|---|---|---|---|---|---|---|")
    for name, cfg in CONFIGS.items():
        matcher = HSNMatcher.from_csv(config=cfg)
        for sname, q in sets.items():
            m, _ = evaluate(matcher, q)
            print(f"| {name} | {sname} | {m['n']} | {m['top1_heading']*100:.1f}% | {m['top5_heading']*100:.1f}% | "
                  f"{m['top5_distinct']*100:.1f}% | {m['top1_chapter']*100:.1f}% |")


def evaluate_typos(sets: dict) -> None:
    print("| Config | Set | Clean top-1 | Typo top-1 | Typo top-5 codes |")
    print("|---|---|---|---|---|")
    for name, cfg in list(CONFIGS.items())[:5]:
        matcher = HSNMatcher.from_csv(config=cfg)
        for sname, q in sets.items():
            clean, _ = evaluate(matcher, q)
            noisy = typo_eval(matcher, q)
            print(f"| {name} | {sname} | {clean['top1_heading']*100:.1f}% | "
                  f"{noisy['top1_heading']*100:.1f}% | {noisy['top5_heading']*100:.1f}% |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["dev", "test", "holdout"], default="dev")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("--report", action="store_true", help="regenerate eval/RESULTS.md")
    ap.add_argument("--typos", action="store_true", help="typo-robustness table (no synonyms bias: all configs)")
    args = ap.parse_args()

    if args.report:
        write_report()
        return

    if args.report:
        write_report()
        return

    dev, test, holdout = load("dev"), load("test"), load("holdout")
    sets = {"dev": dev, "test": test, "holdout": holdout}

    if args.ablation:
        print("| Config | Set | n | Top-1 heading | Top-5 codes | Top-5 distinct headings | Top-1 chapter |")
        print("|---|---|---|---|---|---|---|")
        for name, cfg in CONFIGS.items():
            matcher = HSNMatcher.from_csv(config=cfg)
            for sname, q in sets.items():
                m, _ = evaluate(matcher, q)
                print(
                    f"| {name} | {sname} | {m['n']} | {m['top1_heading']*100:.1f}% | "
                    f"{m['top5_heading']*100:.1f}% | {m['top5_distinct']*100:.1f}% | "
                    f"{m['top1_chapter']*100:.1f}% |"
                )
        return

    if args.typos:
        print("| Config | Set | Clean top-1 | Typo top-1 (avg of 5 seeds) | Typo top-5 |")
        print("|---|---|---|---|---|")
        for name, cfg in list(CONFIGS.items())[:5]:
            matcher = HSNMatcher.from_csv(config=cfg)
            for sname, q in sets.items():
                clean, _ = evaluate(matcher, q)
                noisy = typo_eval(matcher, q)
                print(
                    f"| {name} | {sname} | {clean['top1_heading']*100:.1f}% | "
                    f"{noisy['top1_heading']*100:.1f}% | {noisy['top5_heading']*100:.1f}% |"
                )
        return

    final = CONFIGS[FINAL]
    matcher = HSNMatcher.from_csv(config=final)
    q = sets[args.set]
    m, detail = evaluate(matcher, q)
    print(f"Set={args.set}  n={m['n']}")
    print("top1_heading | top5_codes | top5_distinct | top1_chapter")
    print(fmt(m))
    if args.show_misses:
        miss = detail[~detail["top5_distinct"] | ~detail["top1_heading"]]
        print("\nQueries not correct at rank 1 (accepted vs got):")
        print(
            miss[["query", "accepted", "got_top1", "got_distinct", "score1"]]
            .to_string(index=False, float_format=lambda x: f"{x:.2f}")
        )


if __name__ == "__main__":
    main()
