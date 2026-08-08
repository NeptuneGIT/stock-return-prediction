"""
verify_candidate_tickers.py

RUN THIS BEFORE adding any replacement/additional ticker to
02_fundamentals.py (aka 02a_get_fundamentals.py).

WHY: SJW passed a casual eyeball test (regulated water utility, long
history, files 10-Ks) and still turned out to be unusable, because it
never tags an aggregate capex figure. We only discovered that after
scraping and cleaning it. This script front-loads that check.

It scores each candidate on the tags that actually feed the model:
  roe             <- NetIncomeLoss, StockholdersEquity (or NCI-inclusive)
  debt_to_equity  <- Liabilities (or Assets - StockholdersEquity)
  shares          <- one of three share-count tags
  free_cash_flow  <- OperatingCashFlow AND a real capex tag

A candidate is only safe if it clears ALL of them with good coverage
over your study window. Anything that fails the capex check is
another SJW.

Output: a PASS/FAIL table per candidate printed to the console, plus a
persisted CSV at output/tables/candidate_verification_<YYYYMMDD>.csv so
a run's results survive to be cited in EXCLUDED_TICKERS/CAPEX_PRIORITY
comments later (the console table alone doesn't survive past the
terminal scrollback).

CANDIDATE_TICKERS is a flat list, resolved to CIKs automatically via
SEC's company_tickers.json (mirroring get_cik_map() in
02_fundamentals.py) rather than a hand-maintained ticker->CIK dict --
that stops scaling once there are more than a couple of candidates.
MANUAL_CIK_OVERRIDES still exists for the cases auto-lookup gets wrong
(renamed/restructured filers).

REQUIREMENTS["equity"] accepts both plain StockholdersEquity and
StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest,
matching 02_clean_fundamentals.py's EQUITY_PRIORITY fallback -- a
candidate that exclusively tags the NCI-inclusive variant should not be
marked NOT USABLE when the cleaner would handle it fine.
REQUIREMENTS["net_income"] similarly accepts "ProfitLoss" as a fallback
for "NetIncomeLoss", matching NET_INCOME_PRIORITY. REQUIREMENTS["capex"]
includes "PaymentsToAcquireOtherProductiveAssets" as a candidate tag not
yet in the scraper, since some filers split capex reporting across more
than one tag with only partial overlap.

evaluate() scores each requirement by the UNION of quarter-end coverage
across every alternative tag in its list, not just the single
best-covering tag -- this matches what the production cleaner actually
does for shares/capex/equity/net_income (priority-fallback fills gaps
across tags, which is exactly a coverage union), so a candidate whose
best single tag looks thin but whose tags collectively cover the study
window is scored correctly rather than undercounted.

Results are persisted to output/tables/ as well as printed, so a run's
findings survive past the terminal scrollback to be cited later.
"""

import csv
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests


def get_sec_headers():
    """Build the User-Agent header SEC EDGAR requires on every request.

    SEC EDGAR's fair-access policy requires a real name/contact string in
    the User-Agent, not a generic client string, and will reject
    unidentified traffic -- so this fails loudly instead of silently
    sending a placeholder.
    """
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        sys.exit(
            "SEC_USER_AGENT environment variable is not set.\n"
            "SEC EDGAR requires a real contact string in the User-Agent "
            "header, e.g.:\n"
            '  export SEC_USER_AGENT="Your Name your.email@example.com"\n'
            "then re-run this script."
        )
    return {"User-Agent": user_agent}


HEADERS = get_sec_headers()

# Candidates for the 8->25-30 ticker expansion. The pure regulated-water-
# utility universe is exhausted (YORW/ARTNA/CWCO/GWRS already rejected for
# thin XBRL tagging -- the same failure mode that killed SJW; MSEX/SJW
# themselves excluded -- see 02_clean_fundamentals.py EXCLUDED_TICKERS), so
# this list leans into water-adjacent equipment/infrastructure/industrial
# names, the same character as the already-included XYL/VRT/JCI/BMI.
# PNR and WTS were checked once before (old hardcoded CANDIDATES dict) but
# never added -- re-verified here alongside the new names.
CANDIDATE_TICKERS = [
    "PNR",   # Pentair -- water treatment/pool equipment
    "WTS",   # Watts Water Technologies -- water safety/flow control
    "MWA",   # Mueller Water Products -- pipes, valves, meters
    "AOS",   # A.O. Smith -- water heaters/water treatment
    "ITT",   # ITT Inc -- pumps, fluid technology
    "IEX",   # IDEX Corp -- fluidics, pumps
    "FELE",  # Franklin Electric -- water & fuel pumping systems
    "FLS",   # Flowserve Corp -- pumps/valves/seals, flow control
    "DOV",   # Dover Corp -- pumps, fluid transfer
    "ECL",   # Ecolab -- water treatment chemicals
    "GRC",   # Gorman-Rupp -- pump manufacturer
    "CR",    # Crane Company -- fluid handling
    "ROP",   # Roper Technologies -- water metering (Neptune) + diversified
    "EMR",   # Emerson Electric -- automation incl. water/wastewater
    "PH",    # Parker Hannifin -- fluid power/motion control
    "HON",   # Honeywell -- process/building/water solutions
    "WMS",   # Advanced Drainage Systems -- stormwater/drainage infrastructure
    "GVA",   # Granite Construction -- infrastructure incl. water/wastewater
    # Added for buffer after the first verification pass landed at 24
    # tickers (8 existing + 16 USABLE), one short of the 25-30 target.
    "MAS",   # Masco -- Delta Faucet, plumbing products
    "ITRI",  # Itron -- smart metering, incl. water meters (competes w/ BMI)
    "PWR",   # Quanta Services -- infrastructure construction incl. water/electric
    "MTZ",   # MasTec -- infrastructure construction incl. water/sewer
]

# SEC's ticker->CIK file keys on current registered name, so a recent
# rename can break the automatic lookup (this is what happened with SJW,
# which now files as "H2O America"). Add any ticker this script warns
# about here, verified at sec.gov. Keep in sync with the same-named dict
# in 02_fundamentals.py once a candidate is actually added to TICKERS.
MANUAL_CIK_OVERRIDES = {}

# Only quarters at/after this matter for the study.
STUDY_START_YEAR = 2013

# What each model feature needs. Lists are alternatives (any one works).
REQUIREMENTS = {
    # ProfitLoss = consolidated net income including NCI; NetIncomeLoss =
    # parent-only. Same split as "equity" below (some filers, e.g. ECL,
    # tag only one of the two consistently).
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    # Widened to match EQUITY_PRIORITY in 02_clean_fundamentals.py: a
    # filer that only ever tags the NCI-inclusive variant (e.g. CWT from
    # 2021-Q2 onward) is still usable -- the cleaner resolves either by
    # priority, never a blind alias.
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
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
        # Extra candidates checked here but NOT yet in the scraper --
        # if one of these is what a candidate uses, add it to 02_fundamentals.py.
        "PaymentsToAcquireWaterSystems",
        "PaymentsToAcquireUtilityPlant",
        "UtilitiesOperatingExpenseMaintenanceOperations",
        "PaymentsToAcquireRegulatedAssets",
        "PaymentsToAcquireOtherProductiveAssets",
    ],
}

MIN_QUARTERS = 30  # below this, coverage is too thin to be useful

OUT_DIR = Path("output/tables")


def get_cik_map(tickers):
    """
    Map ticker -> 10-digit zero-padded CIK, via SEC's bulk ticker file
    plus MANUAL_CIK_OVERRIDES. Mirrors get_cik_map() in
    02_fundamentals.py:149-172 -- duplicated rather than imported so this
    script stays runnable standalone; keep the two in sync if either
    changes.
    """
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json", headers=HEADERS
    )
    resp.raise_for_status()

    lookup = {
        row["ticker"]: str(row["cik_str"]).zfill(10) for row in resp.json().values()
    }
    lookup.update(
        {t: str(c).strip().zfill(10) for t, c in MANUAL_CIK_OVERRIDES.items()}
    )

    missing = [t for t in tickers if t not in lookup]
    if missing:
        print(
            f"  WARNING: no CIK found for {missing}. Look them up at "
            f"sec.gov and add to MANUAL_CIK_OVERRIDES."
        )
    return {t: lookup[t] for t in tickers if t in lookup}


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
        return None, {}

    name = facts.get("entityName", "?")
    print(f"{ticker}  |  {name}  |  CIK {cik}")
    print("=" * 72)

    # Score by the UNION of quarter-end coverage across every alternative
    # tag in the requirement's list, not just the single best-covering
    # tag. Shares/capex/equity/net_income are all resolved in production
    # by priority-fallback (.fillna() across tags), which fills gaps from
    # every tag in the list -- exactly a coverage union. Scoring by best-
    # single-tag alone undercounts a candidate that spreads coverage
    # across two or more tags.
    results = {}
    for requirement, tags in REQUIREMENTS.items():
        found = {}
        union = set()
        for tag in tags:
            cov = tag_coverage(facts, tag)
            if cov:
                found[tag] = len(cov)
            union |= cov
        best_tag = max(found, key=found.get) if found else None
        results[requirement] = (best_tag, len(union), found)

    print(f"\n{'REQUIREMENT':<26} {'STATUS':<7} {'QTRS':>5}  BEST TAG (individual coverage; QTRS is the combined union)")
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
        print(f"\n  ACTION: add these capex tags to 02_fundamentals.py TAGS + CAPEX_PRIORITY:")
        for t in new_tags:
            print(f"    {t}  ({capex_found[t]} quarters)")

    print(f"\n  VERDICT: {'USABLE' if all_pass else 'NOT USABLE — do not add'}")
    return all_pass, {req: n for req, (_tag, n, _found) in results.items()}


def main():
    cik_map = get_cik_map(CANDIDATE_TICKERS)
    verdicts = {}
    rows = []

    for ticker in CANDIDATE_TICKERS:
        cik = cik_map.get(ticker)
        if cik is None:
            verdicts[ticker] = None
            rows.append({"ticker": ticker, "cik": "", "verdict": "NO CIK FOUND"})
            continue

        ok, quarter_counts = evaluate(ticker, cik)
        verdicts[ticker] = ok
        label = {True: "USABLE", False: "NOT USABLE", None: "FETCH FAILED"}[ok]
        row = {"ticker": ticker, "cik": cik, "verdict": label}
        row.update(quarter_counts)
        rows.append(row)
        time.sleep(0.2)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for ticker, ok in verdicts.items():
        label = {True: "USABLE", False: "NOT USABLE", None: "FETCH FAILED"}[ok]
        print(f"  {ticker:<8} {label}")
    print("\nAdd USABLE candidates to TICKERS in 02_fundamentals.py / STOCKS in")
    print("01_get_data.R, add any new capex tags flagged above, then re-run")
    print("02_fundamentals.py and 02_clean_fundamentals.py.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"candidate_verification_{date.today():%Y%m%d}.csv"
    fieldnames = ["ticker", "cik", "verdict"] + list(REQUIREMENTS.keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
