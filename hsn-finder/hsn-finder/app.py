"""HSN Code Finder - Streamlit app.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from hsn_finder import HSNMatcher, strength_label

CBIC_RATES_URL = "https://cbic-gst.gov.in/gst-goods-services-rates.html"
EXAMPLES = [
    "stainless steel kitchen sink",
    "cotton t-shirt for men",
    "basmati rice",
    "LED bulb 9 watt",
    "corrugated box",
    "mobile phone",
    "TMT steel bars",
    "turmeric powder",
]
STRENGTH_ICON = {"High": "🟢", "Medium": "🟡", "Low": "🟠"}

st.set_page_config(page_title="HSN Code Finder", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading the HSN index...")
def get_matcher() -> HSNMatcher:
    return HSNMatcher.from_csv()


def shorten(text: str, n: int = 110) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def breadcrumb(path: str) -> str:
    """'A > B > C' -> 'A › B › C' with each level shortened and title-cased."""
    levels = [shorten(p.strip().capitalize(), 110) for p in path.split(" > ")]
    return " › ".join(levels)


def set_query(text: str) -> None:
    st.session_state["query"] = text


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "Type what you sell in plain English and get the most likely **HSN codes** "
        "(8-digit Indian tariff codes), with the chapter and heading they sit under."
    )
    st.subheader("How it works")
    st.markdown(
        "- TF-IDF over **12,604** official HSN descriptions\n"
        "- Character n-grams + spelling correction for typos\n"
        "- Heading-level matching so a strong 4-digit heading lifts its sub-codes\n"
        "- A trade-vocabulary table (~300 terms) that maps seller words like "
        "*“mobile”* or *“TMT”* to tariff wording"
    )
    st.subheader("Measured accuracy")
    st.markdown(
        "On 66 fresh hand-labelled products (never used for tuning):\n"
        "- right 4-digit heading at **rank 1: ~80%**\n"
        "- right heading within the **top 5: 100%**\n\n"
        "For product words *outside* the vocabulary table, expect closer to ~60% / ~80%. "
        "Details in `eval/RESULTS.md`."
    )
    st.subheader("What it does not do")
    st.markdown(
        "- **No GST rates.** The open datasets I found have unreliable per-code rates "
        "(and GST 2.0 changed the slabs on 22 Sep 2025), so I chose not to show a number "
        "that could be wrong.\n"
        "- It matches the *product itself*, not accessories (a phone cover is not a phone).\n"
        "- Heading 0407 (eggs in shell) is missing from the source data."
    )

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("🔎 HSN Code Finder")
st.caption("Describe a product in plain English → get likely HSN codes. Built for MSME sellers and exporters.")

matcher = get_matcher()

st.text_input(
    "What are you selling?",
    key="query",
    placeholder="e.g. stainless steel kitchen sink",
)

st.write("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, key=f"ex_{ex}", on_click=set_query, args=(ex,), width="stretch")

query = st.session_state.get("query", "").strip()

if not query:
    st.info("Enter a product description above, or click an example.")
else:
    results = matcher.search(query, k=5)
    if not results:
        st.warning(
            "No confident match. Try the product's name plus its material, type or use "
            "(e.g. *“cotton”*, *“kitchen”*, *“industrial”*)."
        )
    else:
        corrected = matcher.fix_spelling(query)
        if corrected.lower() != query.lower():
            st.caption(f"Showing results for **{corrected}**")

        best = results[0]
        label = strength_label(best.score)

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.subheader("Best match")
            with st.container(border=True):
                st.markdown(f"## `{best.code_formatted}`")
                st.markdown(f"{STRENGTH_ICON[label]} **{label} match** · similarity {best.score:.2f}")
                st.markdown(f"**Chapter {best.chapter}** – {best.chapter_title}")
                st.markdown(f"**Heading {best.heading}** · Subheading {best.subheading}")
                st.markdown(breadcrumb(best.path))
                with st.expander("Full official description"):
                    st.write(best.description)

            if label == "Low":
                st.warning(
                    "Lower-confidence match (in testing, rank 1 was right about two times in three "
                    "at this level). Check the other candidates and headings, or add detail "
                    "(material, type, use)."
                )

            st.subheader("Other candidates")
            for i, m in enumerate(results[1:], start=2):
                ml = strength_label(m.score)
                with st.container(border=True):
                    st.markdown(
                        f"**{i}. `{m.code_formatted}`** · {STRENGTH_ICON[ml]} {ml} · {m.score:.2f}  \n"
                        f"Chapter {m.chapter}: {m.chapter_title}  \n"
                        f"{breadcrumb(m.path)}"
                    )

        with right:
            st.subheader("Which 4-digit heading?")
            st.caption("Best distinct headings; the heading is often the safest level to confirm.")
            rows = []
            by_heading = {m.heading: m for m in matcher.search(query, k=60)}
            for h, sc in matcher.top_headings(query, n=5):
                m = by_heading.get(h)
                if m is None:
                    r = matcher.df[matcher.df["heading"] == h].iloc[0]
                    ct, first = r["chapter_title"], r["path"].split(" > ")[0]
                else:
                    ct, first = m.chapter_title, m.path.split(" > ")[0]
                rows.append(
                    {
                        "Heading": h,
                        "Score": round(sc, 2),
                        "Chapter": shorten(ct, 60),
                        "Heading wording": shorten(first.capitalize(), 80),
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

            export = pd.DataFrame(
                [
                    {
                        "query": query,
                        "rank": i,
                        "hsn_code": m.code,
                        "chapter": m.chapter,
                        "chapter_title": m.chapter_title,
                        "description": m.description,
                        "similarity": round(m.score, 3),
                        "match_strength": strength_label(m.score),
                    }
                    for i, m in enumerate(results, 1)
                ]
            )
            st.download_button(
                "⬇️ Download results (CSV)",
                export.to_csv(index=False).encode("utf-8"),
                file_name="hsn_suggestions.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "⚠️ Suggestions only, not a binding classification. Always confirm the HSN code and the "
    f"applicable GST rate on the official [CBIC GST rate page]({CBIC_RATES_URL}) or with your "
    "CA / customs broker before invoicing or filing. Data: CBIC / WCO Harmonized System via the "
    "open-source `hsn-code-package` (ISC)."
)
