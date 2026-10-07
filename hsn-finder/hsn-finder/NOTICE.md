# Data attribution

`data/hsn_codes.csv` is derived from `raw/hsn_codes_source.json`, taken from the
open-source npm package **hsn-code-package** (ISC licence)
<https://github.com/karthi-21/HSN-Code-Package>, which bundles the CBIC / WCO
Harmonized System nomenclature (12,604 8-digit HSN codes and descriptions).

Changes made here: codes left-padded to 8 digits, hierarchy columns derived,
descriptions whitespace-normalised. GST-rate data from the same package was
**intentionally not used** (see README, "Data and its limits").

The HS chapter titles in `hsn_finder/chapters.py` are shortened versions of the
standard WCO Harmonized System chapter names.
