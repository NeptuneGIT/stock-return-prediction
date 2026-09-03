"""
verify_candidate_tickers.py -- run before adding any ticker to
02_fundamentals.py.

A candidate can look fine on the surface (long history, regulated
industry, files 10-Ks) and still be unusable if it never tags an
aggregate capex figure or another field the model needs. This front-loads
that check instead of finding out after scraping and cleaning.

Scores each candidate on the tags that feed the model: roe, debt_to_equity,
shares, free_cash_flow. Safe only if it clears ALL of them with good
coverage over the study window (>= MIN_QUARTERS from STUDY_START_YEAR).

Output: PASS/FAIL table to console, plus a persisted CSV at
output/tables/candidate_verification_<YYYYMMDD>.csv. USABLE candidates go
into TICKERS/STOCKS; NOT USABLE ones get appended to EXCLUDED_TICKERS.
See docs/METHODOLOGY.md for the full ticker-screening workflow.

Requires the SEC_USER_AGENT environment variable (see get_sec_headers()
below) -- SEC EDGAR requires a real contact string in the User-Agent header
on every request and will reject unidentified traffic.
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

    SEC EDGAR's fair-access policy requires a real name/contact string, not a
    generic client string -- see https://www.sec.gov/os/webmaster-faq#developers.
    Failing loudly here beats silently sending a placeholder that SEC could
    start blocking without warning.
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

CANDIDATE_TICKERS = [
    # --- Information Technology (35) ---
    "MSFT", "AAPL", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "ACN",
    "AMD", "QCOM", "TXN", "IBM", "INTU", "AMAT", "ADI", "LRCX", "KLAC",
    "MU", "SNPS", "CDNS", "ADSK", "NXPI", "MCHP", "ON", "TER", "JNPR",
    "WDC", "STX", "NTAP", "TYL", "PTC", "SWKS", "GRMN", "ZBRA",
    # --- Health Care (30) ---
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "MDT", "GILD", "CVS", "CI", "ELV", "HUM", "SYK", "BSX", "ISRG",
    "ZBH", "BDX", "BAX", "REGN", "VRTX", "BIIB", "MCK", "COR", "HCA", "DVA",
    # --- Financials (30) ---
    "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "BK",
    "SCHW", "BLK", "SPGI", "MCO", "CME", "ICE", "AON", "MMC", "AJG", "TRV",
    "ALL", "AIG", "PGR", "MET", "PRU", "AFL", "CB", "HIG", "STT", "FITB",
    # --- Consumer Discretionary (29) ---
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG",
    "ORLY", "AZO", "ROST", "YUM", "MAR", "HLT", "GM", "F", "APTV", "BBY",
    "DHI", "LEN", "PHM", "NVR", "WHR", "TSCO", "ULTA", "GPC", "CMG", "EXPE",
    # --- Communication Services (15) ---
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    "EA", "TTWO", "WBD", "OMC", "IPG", "LYV",
    # --- Industrials(15) ---
    "CAT", "DE", "UNP", "UPS", "FDX", "BA", "LMT", "RTX", "GD", "NOC",
    "CSX", "NSC", "WM", "RSG", "GE",
    # --- Consumer Staples (23) ---
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "KHC", "HSY", "STZ", "SYY", "ADM", "TSN", "TAP", "CAG", "CLX",
    "MKC", "K", "CHD",
    # --- Energy (18) ---
    "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "VLO", "OXY", "WMB",
    "KMI", "OKE", "HES", "DVN", "FANG", "BKR", "HAL", "TRGP",
    # --- Utilities (15) ---
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "ED", "WEC", "PEG",
    "ES", "FE", "ETR", "EIX", "PPL",
    # --- Real Estate (15) ---
    "PLD", "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "SPG", "DLR", "AVB",
    "EQR", "VTR", "ESS", "MAA", "IRM",
    # --- Materials, beyond the existing Ecolab (15) ---
    "LIN", "APD", "SHW", "FCX", "NEM", "DOW", "DD", "PPG", "NUE", "ALB",
    "CE", "IFF", "MLM", "VMC", "IP",
]


MANUAL_CIK_OVERRIDES = {
    "AEP": "0000004904",
}

# Only quarters at/after this matter for the study.
STUDY_START_YEAR = 2003

REQUIREMENTS = {

    "net_income": ["NetIncomeLoss", "ProfitLoss"],
  
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

    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",

        "PaymentsToAcquireOtherProductiveAssets",
        "PaymentsForProceedsFromProductiveAssets",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireWaterAndWasteWaterSystems",
        "PaymentsForConstructionInProcess",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
        "PaymentsToAcquireMachineryAndEquipment",
        "PaymentsToAcquireBuildings",
        "PaymentsToAcquireOilAndGasPropertyAndEquipment",
        "PaymentsToAcquireEquipmentOnLease",

        "PaymentsToAcquireWaterSystems",
        "PaymentsToAcquireUtilityPlant",
        "UtilitiesOperatingExpenseMaintenanceOperations",
        "PaymentsToAcquireRegulatedAssets",
    ],
}


CAPEX_ALREADY_IN_SCRAPER = 11

MIN_QUARTERS = 48 

OUT_DIR = Path("output/tables")


def get_cik_map(tickers):
    """Map ticker -> 10-digit zero-padded CIK, via SEC's bulk file plus
    MANUAL_CIK_OVERRIDES. Duplicated from 02_fundamentals.py rather than
    imported so this stays runnable standalone -- keep in sync."""
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
    scraper_has = set(REQUIREMENTS["capex"][:CAPEX_ALREADY_IN_SCRAPER])
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
    suffix = 2
    while out_path.exists():
        out_path = OUT_DIR / f"candidate_verification_{date.today():%Y%m%d}-{suffix}.csv"
        suffix += 1
    fieldnames = ["ticker", "cik", "verdict"] + list(REQUIREMENTS.keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
