"""
verify_candidate_tickers.py

RUN THIS BEFORE adding any replacement/additional ticker to
02_fundamentals.py (aka 02a_get_fundamentals.py).

WHY: SJW passed a casual eyeball test (regulated water utility, long
history, files 10-Ks) and still turned out to be unusable, because it
never tags an aggregate capex figure. That was only discovered after
scraping and cleaning it. This script front-loads that check.

It scores each candidate on the tags that actually feed the model:
  roe             <- NetIncomeLoss, StockholdersEquity (or NCI-inclusive)
  debt_to_equity  <- Liabilities (or Assets - StockholdersEquity)
  shares          <- one of three share-count tags
  free_cash_flow  <- OperatingCashFlow AND a real capex tag

A candidate is only safe if it clears ALL of them with good coverage
over the study window. Anything that fails the capex check is another
SJW. Candidates are scored by the UNION of quarter-end coverage across
every alternative tag in a requirement's list, not just the single
best-covering tag, matching what the production cleaner
(02_clean_fundamentals.py) actually does -- its priority-fallback fills
gaps across tags via .fillna(), which is exactly a coverage union, so
scoring by the single best tag alone would undercount a candidate whose
coverage is split across several tags.

Output: a PASS/FAIL table per candidate printed to the console, plus a
persisted CSV at output/tables/candidate_verification_<YYYYMMDD>.csv so
a run's results survive to be cited in EXCLUDED_TICKERS/CAPEX_PRIORITY
comments later.

CIK lookup is automatic via SEC's company_tickers.json
(mirroring get_cik_map() in 02_fundamentals.py), with
MANUAL_CIK_OVERRIDES for the cases auto-lookup gets wrong (renamed or
restructured filers, the same failure mode that hit SJW/"H2O America").
STUDY_START_YEAR and MIN_QUARTERS are set in lockstep with
01_get_data.R's START_DATE and 01a_ratios.R's STUDY_START -- if those
move, this needs to move with them or a candidate's coverage window here
won't match what the actual pipeline needs. MIN_QUARTERS = 48 (12 years)
is a judgment call balancing "enough clean tickers survive the screen"
against "history is actually long enough to be worth the project's
15-20yr target," not a value derived mechanically.

SCREENING RESULTS FOR THE CURRENT UNIVERSE
This universe grew from an initial 8-ticker water-utility set through a
28-ticker water/infrastructure-adjacent expansion to the current
222-ticker S&P 500 universe, screening roughly 270 candidates for the
pivot to the full S&P 500 (240 in the main batch below, plus a
30-candidate supplemental round targeting the thinnest sectors).
Overall reject rate for the pivot was ~28%, meaningfully higher than the
8->28 expansion's, concentrated in Financials and Communication
Services: most bank/insurer rejects failed specifically on `capex`,
since banks and insurers structurally don't tag a PP&E-style capex
figure the way an industrial or utility does (payment networks,
exchanges, and asset managers like V, MA, BX, KKR, SYF failed for the
same reason, while NDAQ/MSCI/FDS/CBOE/AXP/COF/NTRS/WTW passed).

A handful of "casual eyeball test" failures are worth calling out
explicitly, since they extend the SJW/CR lesson this script exists to
catch: XOM and BLK both resolve to newly-registered holding-company CIKs
with only a handful of quarters of history under that exact CIK (real,
current SEC ticker->CIK mappings, verified directly against SEC's bulk
company_tickers.json, reflecting recent holding-company restructurings,
not a lookup bug); GOOGL (Alphabet's 2015 reorg), MDT (Medtronic plc's
2015 Ireland redomiciliation), and DIS (2019 registrant change) are the
same pattern. A ticker being an obvious, decades-old household name is
not evidence its current SEC registrant has matching history. AEP
required a MANUAL_CIK_OVERRIDES entry: present and active in SEC's own
submissions API but absent from the bulk company_tickers.json used for
lookup, a bulk-file gap rather than a corporate event. BK and MMC
returned no CIK under those symbols because both tickers changed (Bank
of New York Mellon to BNY, Marsh & McLennan to MRSH); re-verified and
added under their new symbols. JNPR, IPG, HES, K, DFS, and PARA all
returned no CIK with no renamed successor findable, consistent with each
having been acquired or taken private and no longer filing as
independent registrants -- dropped from the candidate pool entirely
rather than guessed at.
"""

import csv
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests


def get_sec_headers():
    """Build the User-Agent header SEC EDGAR requires on every request."""
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

# Candidates for the 28 -> ~200-250 ticker S&P-500-universe pivot. The
# prior water-adjacent CANDIDATE_TICKERS list from the 8->28 expansion is
# gone from here -- every ticker on it is now either already live in
# TICKERS (02_fundamentals.py) or already a confirmed reject in
# EXCLUDED_TICKERS (02_clean_fundamentals.py; WTS/CR), so none of them
# need re-screening.
#
# This list is ~240 NEW candidates spanning the 10 GICS sectors the
# existing 28 barely touch (that list is almost entirely Industrials +
# 4 water Utilities + 1 Materials + 1 Information Technology name), sized
# so that 240 new + 28 already-verified = ~268 total candidate pool,
# inside the plan's 260-300 target. Selection favors long-listed
# (pre-2008 IPO/spinoff where practical) large/mid-cap names for the
# 15-20yr history target, spans market-cap tiers within each sector
# rather than only mega-caps, and deliberately includes a small minority
# of newer spinoffs/mergers/renames (e.g. RTX, KHC, TFC, DD, WBD, K, GE)
# as calculated-risk candidates likely to fail the MIN_QUARTERS=48 (12yr)
# or CIK-history check -- same posture as the CR (Crane) precedent from
# the 8->28 expansion: let this script's own screen decide, don't
# hand-filter borderline cases out in advance. Smaller sectors
# (Materials, Real Estate, Utilities, Communication Services) are
# oversampled relative to their S&P 500 market-cap weight so all 11 GICS
# sectors end up meaningfully represented in the final universe, not just
# proportionally represented.
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
    # --- Industrials, beyond the existing 22 water/fluid-equipment names (15) ---
    "CAT", "DE", "UNP", "UPS", "FDX", "BA", "LMT", "RTX", "GD", "NOC",
    "CSX", "NSC", "WM", "RSG", "GE",
    # --- Consumer Staples (23) ---
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "KHC", "HSY", "STZ", "SYY", "ADM", "TSN", "TAP", "CAG", "CLX",
    "MKC", "K", "CHD",
    # --- Energy (18) ---
    "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "VLO", "OXY", "WMB",
    "KMI", "OKE", "HES", "DVN", "FANG", "BKR", "HAL", "TRGP",
    # --- Utilities, beyond the existing 4 water utilities (15) ---
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "ED", "WEC", "PEG",
    "ES", "FE", "ETR", "EIX", "PPL",
    # --- Real Estate (15) ---
    "PLD", "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "SPG", "DLR", "AVB",
    "EQR", "VTR", "ESS", "MAA", "IRM",
    # --- Materials, beyond the existing Ecolab (15) ---
    "LIN", "APD", "SHW", "FCX", "NEM", "DOW", "DD", "PPG", "NUE", "ALB",
    "CE", "IFF", "MLM", "VMC", "IP",
]

# SEC's ticker->CIK file keys on current registered name, so a recent
# rename can break the automatic lookup (this is what happened with SJW,
# which now files as "H2O America"). Add any ticker this script warns
# about here, verified at sec.gov. Keep in sync with the same-named dict
# in 02_fundamentals.py once a candidate is actually added to TICKERS.
MANUAL_CIK_OVERRIDES = {
    # American Electric Power: has an active CIK and current SEC filings
    # (confirmed directly via data.sec.gov/submissions/CIK0000004904.json,
    # ticker AEP, name "AMERICAN ELECTRIC POWER CO INC"), but is simply
    # absent from SEC's bulk company_tickers.json -- confirmed by
    # exact-match lookup returning nothing, not a rename/restructuring
    # like the SJW/H2O America case. A bulk-file gap, not a real
    # corporate event; verified USABLE once given this CIK.
    "AEP": "0000004904",
}

# Only quarters at/after this matter for the study.
STUDY_START_YEAR = 2003

# What each model feature needs. Lists are alternatives (any one works).
REQUIREMENTS = {
    # ProfitLoss = consolidated net income including NCI; NetIncomeLoss =
    # parent-only. Same split as "equity" below -- ECL tags NetIncomeLoss
    # for only 22/54 quarters but ProfitLoss for all 54.
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
        # Added to 02_fundamentals.py TAGS / CAPEX_PRIORITY during the
        # 8->28 expansion's ROP fix -- belongs in the "already in the
        # scraper" portion of this list, NOT the "extra candidates" tail
        # below. It used to sit in that tail, which made CAPEX_ALREADY_IN_SCRAPER
        # stale and caused this script to print a spurious "ACTION: add
        # PaymentsToAcquireOtherProductiveAssets" hint for every single
        # candidate that uses it -- every one of those hints turned out
        # to be this one already-added tag, not a real gap.
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
        # Extra candidates checked here but NOT yet in the scraper --
        # if one of these is what a candidate uses, add it to 02_fundamentals.py.
        "PaymentsToAcquireWaterSystems",
        "PaymentsToAcquireUtilityPlant",
        "UtilitiesOperatingExpenseMaintenanceOperations",
        "PaymentsToAcquireRegulatedAssets",
    ],
}

# The first CAPEX_ALREADY_IN_SCRAPER tags in REQUIREMENTS["capex"] mirror
# what's already live in 02_fundamentals.py's TAGS/CAPEX_PRIORITY; the
# rest are candidates this checker tests but the scraper doesn't pull
# yet. Named explicitly (rather than a magic-number slice like
# REQUIREMENTS["capex"][:11]) so reordering/extending the list above
# can't silently desync it from what evaluate() treats as "already
# handled" -- exactly the bug that caused the stale-hint issue fixed
# just above.
CAPEX_ALREADY_IN_SCRAPER = 11

MIN_QUARTERS = 48  # below this, coverage is too thin to be useful (12yr, raised from 30/7.5yr for the ~20yr history target)

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
    # across two or more tags (e.g. AOS, ROP, ECL).
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
    # Suffix with -2, -3, ... on a same-day re-run instead of silently
    # overwriting an earlier run's archived results -- found the hard way
    # when the 28->~230 pivot's screen landed on the same calendar date as
    # the still-cited 8->28 expansion's results file and clobbered it (the
    # old file was only recoverable via git history afterward).
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
