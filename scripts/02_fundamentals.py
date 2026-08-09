"""
02a_get_fundamentals.py

Pulls raw fundamental facts from SEC EDGAR's free company-facts API for
the 222-ticker S&P 500 universe (TICKERS below).

This script ONLY downloads. All cleaning happens in 02b_clean_fundamentals.py,
so you can re-clean without re-hitting the SEC API.

Key design point: every fact is stored with its `filed` date -- the day
the number actually became public. That is what makes point-in-time
correctness possible downstream and prevents look-ahead bias.

The universe has grown from an initial 8-ticker water-utility set through
a 28-ticker water/infrastructure-adjacent expansion to the current
222-ticker S&P 500 universe spanning all 11 GICS sectors below. Every
addition was verified via verify_candidate_tickers.py
(scripts/verifications.py) before being added -- see that script and
EXCLUDED_TICKERS in 02_clean_fundamentals.py (defense-in-depth reject
list) for the full per-ticker screening rationale, including the
recurring structural gaps found along the way: Financials/REITs
structurally not tagging a PP&E-style capex figure, filers whose current
CIK is too recent (a 2023 spinoff, for example) to clear the trading-
history screen, and a handful of tickers needing a MANUAL_CIK_OVERRIDES
entry because SEC's bulk company_tickers.json doesn't list them despite
having active filings under that CIK.

TAGS accumulated a few additions along the way as individual filers'
tagging quirks surfaced: AssetsCurrent/LiabilitiesCurrent for
current_ratio (instant balance-sheet facts, no duration filtering
needed); StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest
as a fallback for filers (e.g. CWT) that stop tagging plain
StockholdersEquity once they carry a noncontrolling interest; ProfitLoss
as a fallback for filers (e.g. ECL) that tag NetIncomeLoss only
sporadically but ProfitLoss (the NCI-inclusive variant) consistently; and
PaymentsToAcquireOtherProductiveAssets for filers (e.g. ROP) that split
capex reporting across two component tags with little overlap. All of
these are resolved by priority downstream in 02b, never blindly aliased
or summed, since the NCI-inclusive and parent-only variants of a figure
can genuinely differ once a filer actually carries a noncontrolling
interest.

Requires the SEC_USER_AGENT environment variable to be set (see
get_sec_headers() below) -- SEC EDGAR requires a real contact string in
the User-Agent header on every request and will reject unidentified
traffic.

Output: data/raw/fundamentals_raw.csv
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
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

TICKERS = [
    # =====================================================================
    # Original 28-ticker water-sector universe (unchanged by the pivot).
    # =====================================================================
    # Original 8-ticker universe (regulated water utilities + water tech/infra)
    "AWK", "WTRG", "CWT", "AWR", "XYL", "VRT", "JCI", "BMI",
    # Added in the 8 -> 28 expansion, all verified via
    # verify_candidate_tickers.py (output/tables/candidate_verification_20260808_8to28expansion.csv).
    # Water/fluid equipment manufacturers:
    "PNR", "MWA", "AOS", "ITT", "IEX", "FELE", "FLS", "DOV", "ECL", "GRC",
    # Water metering / plumbing:
    "ITRI", "MAS",
    # Diversified industrials with meaningful water/fluid segments:
    "ROP", "EMR", "PH", "HON",
    # Infrastructure construction incl. water/wastewater/drainage:
    "WMS", "GVA", "PWR", "MTZ",
    # =====================================================================
    # 194 tickers added in the 28 -> 222 S&P-500-universe pivot, all
    # verified via verify_candidate_tickers.py (see that script's CHANGE
    # LOG and output/tables/candidate_verification_20260808_sp500pivot.csv
    # for full per-ticker screening results and rationale). Grouped by
    # GICS sector for readability; sector totals include the original 28
    # above (e.g. Industrials = 22 original + 14 new = 36).
    # =====================================================================
    # --- Information Technology (32 total: 1 original + 31 new) ---
    "MSFT", "AAPL", "NVDA", "ORCL", "CRM", "ADBE", "CSCO", "AMD", "QCOM",
    "TXN", "IBM", "INTU", "AMAT", "ADI", "LRCX", "KLAC", "MU", "SNPS",
    "CDNS", "ADSK", "MCHP", "ON", "TER", "WDC", "STX", "NTAP", "TYL",
    "PTC", "SWKS", "GRMN", "ZBRA",
    # --- Health Care (26 total, all new) ---
    "JNJ", "UNH", "LLY", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "GILD", "CVS", "ELV", "HUM", "SYK", "BSX", "ISRG", "ZBH", "BDX", "BAX",
    "VRTX", "BIIB", "MCK", "COR", "HCA", "DVA",
    # --- Financials (24 total, all new) ---
    "C", "GS", "MS", "BNY", "SCHW", "SPGI", "MCO", "ICE", "AON", "MRSH",
    "AJG", "ALL", "PGR", "HIG", "STT", "FITB", "AXP", "COF", "NDAQ",
    "MSCI", "FDS", "CBOE", "NTRS", "WTW",
    # --- Consumer Discretionary (24 total, all new) ---
    "AMZN", "TSLA", "HD", "MCD", "LOW", "SBUX", "TJX", "ORLY", "AZO",
    "ROST", "YUM", "MAR", "HLT", "GM", "APTV", "BBY", "DHI", "PHM", "NVR",
    "WHR", "TSCO", "ULTA", "GPC", "CMG",
    # --- Communication Services (10 total, all new) ---
    "NFLX", "T", "VZ", "TMUS", "EA", "TTWO", "OMC", "LYV", "MTCH", "SIRI",
    # --- Industrials (36 total: 22 original + 14 new) ---
    "CAT", "DE", "UNP", "FDX", "BA", "LMT", "RTX", "GD", "NOC", "CSX",
    "NSC", "WM", "RSG", "GE",
    # --- Consumer Staples (15 total, all new) ---
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "ADM", "CAG", "CLX", "CHD",
    # --- Energy (12 total, all new) ---
    "CVX", "EOG", "SLB", "MPC", "VLO", "OXY", "WMB", "KMI", "OKE", "DVN",
    "HAL", "TRGP",
    # --- Utilities (17 total: 4 original water utilities + 13 new) ---
    "AEP", "DUK", "SO", "D", "EXC", "XEL", "ED", "PEG", "ES", "FE", "ETR",
    "EIX", "PPL",
    # --- Real Estate (13 total, all new) ---
    "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "AVB", "EQR", "VTR", "IRM",
    "UDR", "HST", "BXP",
    # --- Materials (13 total: 1 original + 12 new) ---
    "APD", "SHW", "FCX", "NEM", "PPG", "NUE", "ALB", "CE", "IFF", "MLM",
    "VMC", "IP",
]
# MSEX excluded: >50% of its ROE history had to be forward-filled from
# stale quarters (a 25-consecutive-quarter flat run at worst) because
# its own SEC filings didn't tag granular quarterly figures for long
# stretches. Not a scraper issue -- the data was never filed at that
# level of detail.
# SJW excluded: never tags an aggregate capital-expenditure figure in
# XBRL under any standard us-gaap tag (confirmed by auditing every tag
# it has ever filed), so free_cash_flow was uncomputable for all 53 of
# its quarters. Its narrow PaymentsToAcquireWaterSystems /
# PaymentsToAcquireRealEstate tags cover small acquisitions, not its
# real capex, and using them would give SJW a different capex
# definition than every other ticker in the panel.
# BMI (Badger Meter) added as SJW's replacement -- verified via
# verify_candidate_tickers.py to report all needed tags cleanly.
# WTS (Watts Water Technologies) considered and excluded: never tags any
# point-in-time common-shares-outstanding figure under any standard tag
# (only weighted-average diluted/basic shares used for EPS, and
# preferred-stock counts) -- confirmed via companyfacts audit. Same
# category of gap as SJW's missing capex tag, just for shares_outstanding
# instead of free_cash_flow.
# CR (Crane Company) considered and excluded: its current CIK
# (0001944013) only dates to the 2023 Crane Holdings / Crane NXT
# spinoff, giving it ~19-27 quarters of history depending on tag --
# well under both MIN_QUARTERS=30 and the project's >=10-year trading
# history screen. Not a data-quality gap, just too new under this CIK.
# The remaining NOT USABLE candidates from the S&P-500-pivot screen are
# not individually re-explained here -- see EXCLUDED_TICKERS in
# 02_clean_fundamentals.py (defense-in-depth reject list) and
# verifications.py for the full per-ticker screening detail.

# SEC's ticker->CIK file keys on current registered name, so a recent
# rename can break the automatic lookup (this is what happened with
# SJW, which now files as "H2O America"). Add any ticker the scraper
# warns about here, verified at sec.gov.
MANUAL_CIK_OVERRIDES = {
    # American Electric Power: active CIK with current SEC filings
    # (confirmed via data.sec.gov/submissions/CIK0000004904.json), but
    # absent from SEC's bulk company_tickers.json -- a bulk-file gap,
    # not a rename/restructuring. See verifications.py for how this
    # was diagnosed.
    "AEP": "0000004904",
}

# XBRL tags to pull. SEC's taxonomy is NOT standardized across
# companies or across time, so several concepts need multiple tags.
# Most live in "us-gaap"; TAG_TAXONOMY lists the exceptions.
TAGS = [
    # Balance sheet (instant facts)
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    # Fallback for filers that stop tagging plain StockholdersEquity once
    # they carry a noncontrolling interest (CWT, from 2021-Q2 onward).
    # Resolved by PRIORITY in 02b -- plain StockholdersEquity wins when
    # present -- never blindly aliased, since NCI-inclusive equity can
    # genuinely differ from parent-only equity.
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    # --- Current portions, for current_ratio = AssetsCurrent / LiabilitiesCurrent ---
    # These are INSTANT facts like the rest of the balance sheet, so
    # they bypass the duration-filtering and cumulative-differencing
    # machinery entirely. There is no aliasing problem here either:
    # unlike revenue, these two tag names have been stable across the
    # whole XBRL era.
    # CAVEAT: not every filer presents a CLASSIFIED balance sheet. Some
    # regulated utilities present an unclassified one, in which case
    # these tags legitimately do not exist and current_ratio will be
    # missing for that ticker. That is a reporting choice at the
    # source, not a scraper bug -- check the per-ticker coverage report
    # printed by 02b before trusting this feature.
    "AssetsCurrent",
    "LiabilitiesCurrent",
    # Income statement (duration facts)
    "NetIncomeLoss",
    # Fallback for filers that report net income mostly under the
    # consolidated-including-NCI tag rather than the parent-only one
    # (ECL/Ecolab: NetIncomeLoss only 22/54 quarters, ProfitLoss 54/54).
    # Resolved by PRIORITY in 02b via NET_INCOME_PRIORITY -- same pattern
    # as StockholdersEquity above, never blindly aliased.
    "ProfitLoss",
    "EarningsPerShareDiluted",  # retained as a diagnostic only; pe_ratio removed
    # Revenue: companies migrated to the ASC 606 tag around 2018, and
    # some never used the generic "Revenues" tag at all (e.g. CWT).
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    # Share count, in order of preference. Note these are NOT strictly
    # interchangeable: SharesIssued can exceed SharesOutstanding when a
    # company holds treasury stock. The cleaner uses them as a
    # priority-ordered fallback, never a blind merge.
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
    "CommonStockSharesIssued",
    # --- Cash flow statement, for Free Cash Flow = OCF - CapEx ---
    # Operating cash flow. The "ContinuingOperations" variant is used
    # by companies that have discontinued operations to report; it's a
    # naming alternative, not an additional component.
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    # CapEx. IMPORTANT: unlike revenue aliases, these are NOT all
    # mutually exclusive names for one concept. Some companies report a
    # single total; others break capex into several line items that
    # co-exist in the same filing. The cleaner therefore resolves these
    # by PRIORITY (first available broad tag wins), never by summing --
    # summing would double-count any company reporting both a total and
    # its components. Ordered broadest-first below.
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    # ROP (Roper Technologies) splits capex across these two tags with
    # only 5 quarters of overlap out of 54 -- unioned they cover its
    # full history, alone neither does. Resolved by PRIORITY like every
    # other capex tag, never summed.
    "PaymentsToAcquireOtherProductiveAssets",
    "PaymentsForProceedsFromProductiveAssets",
    "PaymentsForCapitalImprovements",
    "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    "PaymentsToAcquireMachineryAndEquipment",
    "PaymentsToAcquireBuildings",
    "PaymentsForConstructionInProcess",
    "PaymentsToAcquireWaterAndWasteWaterSystems",  # water-utility specific
    "PaymentsToAcquireOilAndGasPropertyAndEquipment",
    "PaymentsToAcquireEquipmentOnLease",
]

# Tags not in the us-gaap taxonomy. EntityCommonStockSharesOutstanding
# is a filing cover-page fact, filed under "dei" (Document & Entity
# Information) rather than as a financial-statement line item.
TAG_TAXONOMY = {
    "EntityCommonStockSharesOutstanding": "dei",
}

OUT_PATH = Path("data/raw/fundamentals_raw.csv")
# SEC asks for <=10 req/sec; stay well under. At 28 tickers this delay
# alone adds ~4s total (dwarfed by per-request network latency). At the
# ~230-ticker S&P 500 target, expect the enforced delay to add ~35s
# total, with actual wall-clock time for this whole script dominated by
# per-company-facts-request network latency instead -- realistically on
# the order of 10-15 minutes end to end, not something this constant
# controls.
SEC_REQUEST_DELAY = 0.15


def get_cik_map(tickers):
    """Map ticker -> 10-digit zero-padded CIK."""
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json", headers=HEADERS
    )
    resp.raise_for_status()

    lookup = {
        row["ticker"]: str(row["cik_str"]).zfill(10) for row in resp.json().values()
    }
    # .strip().zfill() guards against a stray space or missing leading
    # zeros in a hand-entered override, which would otherwise produce a
    # malformed URL and a confusing 404.
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


def fetch_company_facts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def parse_tag(facts_json, tag, ticker):
    """Extract all reported values for one tag as long-format rows."""
    taxonomy = TAG_TAXONOMY.get(tag, "us-gaap")
    try:
        units = facts_json["facts"][taxonomy][tag]["units"]
    except KeyError:
        return []  # company doesn't report this tag -- expected, not an error

    rows = []
    for unit_type, entries in units.items():
        for e in entries:
            # 10-Q/10-K only. Amendments and 8-Ks would duplicate or
            # restate figures we already capture.
            if e.get("form") not in ("10-Q", "10-K"):
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "tag": tag,
                    "unit": unit_type,
                    "fy": e.get("fy"),
                    "fp": e.get("fp"),
                    "form": e.get("form"),
                    "start": e.get("start"),  # None for instant facts
                    "end": e.get("end"),
                    "filed": e.get("filed"),  # when this became public
                    "val": e.get("val"),
                }
            )
    return rows


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cik_map = get_cik_map(TICKERS)
    all_rows = []

    for ticker, cik in cik_map.items():
        print(f"Fetching {ticker} (CIK {cik})...")
        try:
            facts = fetch_company_facts(cik)
        except requests.HTTPError as err:
            print(f"  FAILED for {ticker}: {err}")
            continue

        n_before = len(all_rows)
        for tag in TAGS:
            all_rows.extend(parse_tag(facts, tag, ticker))
        print(f"  {len(all_rows) - n_before} facts")

        time.sleep(SEC_REQUEST_DELAY)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(df)} rows for {df['ticker'].nunique()} tickers -> {OUT_PATH}")
    got = set(df["ticker"].unique())
    if set(TICKERS) - got:
        print(f"MISSING TICKERS: {sorted(set(TICKERS) - got)}")

    # Explicit visibility on the two newly-added tags, since a ticker
    # with an unclassified balance sheet will silently report zero rows
    # here and that is the single most likely surprise in this run.
    for tag in ("AssetsCurrent", "LiabilitiesCurrent"):
        sub = df[df["tag"] == tag]
        covered = sorted(sub["ticker"].unique())
        absent = sorted(set(got) - set(covered))
        print(f"\n{tag}: {len(sub)} facts across {len(covered)} tickers")
        if absent:
            print(
                f"  NOT REPORTED BY: {absent} -- current_ratio will be NA for "
                f"these. Likely an unclassified balance sheet, not a bug."
            )


if __name__ == "__main__":
    main()