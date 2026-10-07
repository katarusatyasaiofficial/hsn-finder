"""Command-line interface:  python -m hsn_finder "stainless steel kitchen sink" """
import argparse

from .matcher import HSNMatcher, strength_label


def main() -> None:
    ap = argparse.ArgumentParser(description="Find likely HSN codes for a product description.")
    ap.add_argument("query", nargs="+", help="product description in plain English")
    ap.add_argument("-k", type=int, default=5, help="number of codes to show (default 5)")
    args = ap.parse_args()

    query = " ".join(args.query)
    matcher = HSNMatcher.from_csv()
    results = matcher.search(query, k=args.k)
    if not results:
        print("No match found - try more descriptive words (material, type, use).")
        return
    print(f'\nTop {len(results)} HSN suggestions for "{query}"\n')
    for i, m in enumerate(results, 1):
        print(f"{i}. {m.code_formatted}  [{strength_label(m.score)} match, {m.score:.2f}]")
        print(f"   Chapter {m.chapter}: {m.chapter_title}")
        print(f"   {m.path}\n")
    print("Suggestions only - verify the code and the GST rate on the official CBIC / GST portal.")


if __name__ == "__main__":
    main()
