"""
verify_candidate_tickers.py

RUN THIS BEFORE adding any replacement ticker to 02a_get_fundamentals.py.

WHY: SJW passed a casual eyeball test (regulated water utility, long
history, files 10-Ks) and still turned out to be unusable, because it
never tags an aggregate capex figure. We only discovered that after
scraping and cleaning it. This script front-loads that check.

It scores each candidate on the tags that actually feed the model:
  roe             <- NetIncomeLoss, StockholdersEquity
  debt_to_equity  <- Liabilities (or Assets - StockholdersEquity)
  shares          <- one of three share-count tags
  free_cash_flow  <- OperatingCashFlow AND a real capex tag

A candidate is only safe if it clears ALL of them with good coverage
over your study window. Anything that fails the capex check is
another SJW.

Output: a PASS/FAIL table per candidate, plus the tag names to add to
02a if a candidate uses one we don't already pull.
"""

import os
import sys
import time
from collections import defaultdict

import requests

# SEC blocks generic/unidentified User-Agents, so a real contact string is
# required. Read from the environment rather than hardcoded so this script
# doesn't need editing (or leak a personal email) to run.
_user_agent = os.environ.get("SEC_USER_AGENT")
if not _user_agent:
    sys.exit(
        "SEC_USER_AGENT environment variable is not set.\n"
        'Set it to a real contact string, e.g.:\n'
        '  export SEC_USER_AGENT="Your Name your.email@example.com"'
    )
HEADERS = {"User-Agent": _user_agent}

# Candidate replacements for SJW. Deliberately large-cap, per feedback
# that sub-billion water utilities tend to have thinner, less reliable
# XBRL tagging (which is exactly what happened with SJW's capex gap).
# All three CIKs below were confirmed via SEC's own filing-index pages
# (not guessed) and each is a SINGLE continuous CIK spanning well
# before 2013 — no ticker/entity restructuring gap like VRT/VLTO carry.
CANDIDATES = {
    "PNR": "0000077360",   # Pentair — ~$10B, continuous CIK since 1993
    "WTS": "0000795403",   # Watts Water — ~$6.7B, continuous CIK since 1985 filings
    "BMI": "0000009092",   # Badger Meter — ~$4.3B, continuous CIK since 2004+ filings
}
# Note: WTS runs a 52-week fiscal year where quarters (except Q4) end
# on a Sunday, not a calendar quarter-end — the same kind of
# non-calendar fiscal pattern we already handle correctly for JCI via
# infer_fiscal_year_starts() in 02b. Not a new risk, just flagging it.

# Only quarters at/after this matter for the study.
STUDY_START_YEAR = 2013

# What each model feature needs. Lists are alternatives (any one works).
REQUIREMENTS = {
    "net_income": ["NetIncomeLoss"],
    "equity": ["StockholdersEquity"],
    "assets": ["Assets"],
    "liabilities_or_derivable": ["Liabilities", "Assets"],  # Assets-Equity fallback
    "shares": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesIssued",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    # THE SJW KILLER. A candidate failing this is unusable for FCF.
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForProceedsFromProductiveAssets",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireWaterAndWasteWaterSystems",
        "PaymentsForConstructionInProcess",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
        "PaymentsToAcquireMachineryAndEquipment",
        "PaymentsToAcquireBuildings",
        "PaymentsToAcquireOilAndGasPropertyAndEquipment",
        "PaymentsToAcquireEquipmentOnLease",
        # Extra candidates checked here but NOT yet in the scraper —
        # if one of these is what a candidate uses, add it to 02a.
        "PaymentsToAcquireWaterSystems",
        "PaymentsToAcquireUtilityPlant",
        "UtilitiesOperatingExpenseMaintenanceOperations",
        "PaymentsToAcquireRegulatedAssets",
    ],
}

MIN_QUARTERS = 30  # below this, coverage is too thin to be useful


def fetch_facts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def tag_coverage(facts, tag):
    """Distinct quarter-end dates this tag covers in 10-Q/10-K, post-2013."""
    ends = set()
    for ns_tags in facts.get("facts", {}).values():
        if tag not in ns_tags:
            continue
        for entries in ns_tags[tag].get("units", {}).values():
            for e in entries:
                if e.get("form") not in ("10-Q", "10-K"):
                    continue
                end = e.get("end")
                if end and int(end[:4]) >= STUDY_START_YEAR:
                    ends.add(end)
    return ends


def evaluate(ticker, cik):
    print("=" * 72)
    try:
        facts = fetch_facts(cik)
    except requests.HTTPError as err:
        print(f"{ticker}: FETCH FAILED ({err}) — check the CIK is correct")
        return None

    name = facts.get("entityName", "?")
    print(f"{ticker}  |  {name}  |  CIK {cik}")
    print("=" * 72)

    results = {}
    for requirement, tags in REQUIREMENTS.items():
        best_tag, best_n = None, 0
        found = {}
        for tag in tags:
            n = len(tag_coverage(facts, tag))
            if n:
                found[tag] = n
            if n > best_n:
                best_tag, best_n = tag, n
        results[requirement] = (best_tag, best_n, found)

    print(f"\n{'REQUIREMENT':<26} {'STATUS':<7} {'QTRS':>5}  BEST TAG")
    print("-" * 100)
    all_pass = True
    for requirement, (tag, n, _found) in results.items():
        ok = n >= MIN_QUARTERS
        all_pass &= ok
        print(f"{requirement:<26} {'PASS' if ok else 'FAIL':<7} {n:>5}  {tag or '(none found)'}")

    # Surface any capex tag that isn't already in the scraper.
    scraper_has = set(REQUIREMENTS["capex"][:11])
    capex_found = results["capex"][2]
    new_tags = [t for t in capex_found if t not in scraper_has]
    if new_tags:
        print(f"\n  ACTION: add these capex tags to 02a TAGS + CAPEX_PRIORITY:")
        for t in new_tags:
            print(f"    {t}  ({capex_found[t]} quarters)")

    print(f"\n  VERDICT: {'USABLE' if all_pass else 'NOT USABLE — do not add'}")
    return all_pass


def main():
    verdicts = {}
    for ticker, cik in CANDIDATES.items():
        verdicts[ticker] = evaluate(ticker, cik)
        time.sleep(0.2)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for ticker, ok in verdicts.items():
        label = {True: "USABLE", False: "NOT USABLE", None: "FETCH FAILED"}[ok]
        print(f"  {ticker:<8} {label}")
    print("\nPick a USABLE candidate, add it to TICKERS in 02a, add any")
    print("new capex tags flagged above, then re-run 02a and 02b.")


if __name__ == "__main__":
    main()