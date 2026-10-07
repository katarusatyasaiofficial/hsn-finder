# 🔎 HSN Code Finder

**Describe a product in plain English → get the most likely HSN codes, with the chapter and heading they sit under.**

> **Live demo:** _add your Streamlit link here after deploying (steps below)_
> **Author:** Pranjal Dixit · MBA (International Business), IIFT Delhi

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![tests](https://img.shields.io/badge/tests-30%20passing-brightgreen) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

## Why this exists

Small sellers and exporters on B2B marketplaces have to put an **HSN code** on invoices, GST returns and export/customs paperwork. The official nomenclature has **12,600+ eight-digit codes** written in tariff language ("telephones for cellular networks or for other wireless networks"), while sellers think in product language ("mobile phone"). Picking the wrong code means rework, notices or customs friction.

This project bridges that vocabulary gap with a small, transparent, dependency-light NLP search, and — just as importantly — **measures honestly how well it works**.

## What it does

- Takes a free-text product description (typos allowed).
- Returns the **top 5 HSN codes**, a **match strength** (High / Medium / Low), the **chapter title**, and a **breadcrumb of the official description** showing *why* it matched.
- Shows the **best distinct 4-digit headings**, because the heading is often the safest level to confirm.
- Says **"no confident match"** for gibberish or vague input instead of returning junk.
- Exports results to CSV. Also usable from the command line: `python -m hsn_finder "stainless steel sink"`.

## Quick start

```bash
git clone <your-repo-url> && cd hsn-finder
pip install -r requirements.txt
streamlit run app.py          # web app
python -m hsn_finder "cotton t-shirt for men"   # CLI
pip install pytest && pytest -q                 # 30 tests
python eval/evaluate.py --report                # regenerate eval/RESULTS.md
```

## How it works

```
query ──► spelling correction ──► synonym expansion ──► TF-IDF scoring ──► top-k codes
          (closest known word)    (seller → tariff        code-level  = 70% word 1-2-grams
                                   vocabulary, ~300                    + 30% char 3-5-grams
                                   terms)                 final score = 70% code-level
                                                                      + 30% heading-level index
```

| Component | Why it's there | Measured effect |
|---|---|---|
| Word TF-IDF (unigrams + bigrams) over all 12,604 official descriptions | Baseline lexical retrieval | Top-1 ≈ 59% on holdout |
| Character n-grams | Robust to spelling variants and run-together words | Mainly helps typos |
| Heading-level index | A strong 4-digit heading match lifts its sub-codes | Small gain on clean queries |
| Seller→tariff synonym table | Closes the vocabulary gap ("TMT", "whisky", "chilli") | **Largest gain: +21 pts top-1** |
| Spelling correction | Fixes unknown words against the tariff + seller vocabulary | **Typo top-1 40% → 72%**, no change on clean queries |
| Head-noun boosting | Tried and rejected | No consistent gain, left off |

No neural models and no external APIs: the app loads in ~4 s, answers in <0.1 s, and every ranking can be explained from the matched words.

## Evaluation (please read: this is the part I care about most)

**Protocol.** Ground truth is at the **4-digit HS heading** level, hand-labelled from HS knowledge. I used three query sets:

| Set | n | Role |
|---|---|---|
| `dev` | 40 | Used to design and tune the system |
| `test` | 48 | Used for failure analysis, so **no longer a clean test** |
| `holdout` | 66 | Written and **committed to git before** the synonym table was expanded (commit `cbe34c1`); never used to tune |

**Headline result (holdout, n = 66, final configuration):**

| Metric | Final | Baseline (config C: no synonyms, no spelling fix) |
|---|---|---|
| Right heading at **rank 1** | **80.3%** (95% CI 69–88%) | 59.1% |
| Right heading in **top-5 codes** | 87.9% | 75.8% |
| Right heading in **top-5 distinct headings** | **100%** (95% CI 94–100%) | 83.3% |
| Right **chapter** at rank 1 | 95.5% | 72.7% |
| Rank-1 accuracy with one random typo in the longest word of each query | 71.5% | 35.8% |

**Calibration.** Rank-1 was right 89% of the time when the match strength was High, 74% at Medium and 67% at Low (all sets pooled), which is how the High/Medium/Low labels were set.

### Caveats, stated plainly

1. **The holdout mostly tests vocabulary I covered.** 63 of the 66 holdout queries contain at least one term from the synonym table. I wrote both the table and the queries from the same domain knowledge, so the headline number measures *"the table plus the matcher"*, **not** performance on arbitrary seller vocabulary. For words outside the table, the honest expectation is the no-synonym column: roughly **59% top-1 and 80–83% top-5 headings**.
2. **The synonym table overfit dev before I fixed the process.** My first table (built only from dev failures) moved dev by +30 points and the test set by **0**. That is why I went back, built a broad table systematically by walking the HS sections, and wrote a fresh holdout first. The full history is in `eval/RESULTS.md`.
3. **Small samples.** 66 queries gives roughly ±10 points on rank-1 accuracy.
4. **Labels are mine.** They come from HS knowledge, not from a customs expert or an official concordance. Please spot-check them in `eval/*_queries.csv` before leaning on the numbers.
5. **Later tweaks were made after seeing results** — spelling correction (after the typo evaluation) and the no-match score floor. Neither changed any clean-query metric (verified), but they were not pre-registered.

## Data and its limits

- **Source:** the open-source [`hsn-code-package`](https://github.com/karthi-21/HSN-Code-Package) (ISC), which bundles the CBIC / WCO nomenclature: **12,604 codes**. Rebuild with `python scripts/build_dataset.py`. See `NOTICE.md`.
- **No GST rates are shown — on purpose.** The same package ships a GST rate per code, but I found it is a *chapter-level approximation* (one flat rate for every code in a chapter) and still contains the **12% and 28% slabs that were abolished by GST 2.0 on 22 Sep 2025** (rates are now 5% / 18% / 40%). Showing a confident-looking number that can be wrong is worse than showing none, so the app links to the official [CBIC GST rates page](https://cbic-gst.gov.in/gst-goods-services-rates.html) instead.
- **Known data gap:** heading **0407** (birds' eggs in shell) is missing from the source data, so egg queries cannot succeed. I excluded that one query from evaluation rather than blame the model.
- It is a **suggestion tool**, not a binding classification. Always confirm with the official notifications / your CA or customs broker.

## Known limitations

- It matches the **product itself, not accessories or parts**: "mobile phone cover" returns phones.
- Confusions between **knitted vs woven** garments (chapters 61 vs 62) and between close sibling headings (e.g. 8208 vs 8211 for knives) remain; the top-5 headings panel exists for this reason.
- Lexical matching can't know that "petrol" ≈ "motor spirit" unless the table says so. Coverage grows only as the table grows.
- Only 8-digit goods codes; **services (SAC codes)** are out of scope.

## Ideas for v2

- Replace/augment TF-IDF with **sentence embeddings** (e.g. a multilingual MiniLM) for vocabulary outside the table, evaluated on this same harness.
- **Hinglish / Hindi** product names ("jeera", "dal" are already handled; a transliteration layer is not).
- A **clarifying-question** step ("knitted or woven?") when top candidates span sibling headings.
- Learn the synonym table from **real listing titles and the codes sellers confirm**, which is the natural extension on a marketplace.

## Project structure

```
app.py                     Streamlit web app
hsn_finder/
  matcher.py               TF-IDF + char n-gram + heading-level scoring, spelling fix
  synonyms.py              seller → tariff vocabulary (organised by HS section)
  text.py                  normalisation / light stemmer
  chapters.py              HS chapter titles
  __main__.py              CLI
data/hsn_codes.csv         cleaned dataset (12,604 rows, hierarchy columns)
raw/hsn_codes_source.json  original source data
scripts/build_dataset.py   reproducible data build
eval/                      query sets, evaluate.py, RESULTS.md (auto-generated)
tests/                     30 pytest tests
```

## Deploy (free, ~5 minutes)

1. Push this folder to a **public GitHub repo**.
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**, pick the repo, branch `main`, main file `app.py`, and deploy.
3. Paste the resulting `https://<name>.streamlit.app` link into the *Live demo* line at the top of this README.

## License

MIT for the code (see `LICENSE`). Data attribution in `NOTICE.md`.
