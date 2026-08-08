"""
02a_get_fundamentals.py

Pulls raw fundamental facts from SEC EDGAR's free company-facts API
for the water-sector universe.

This script ONLY downloads. All cleaning happens in 02b_clean_fundamentals.py,
so you can re-clean without re-hitting the SEC API.

Key design point: every fact is stored with its `filed` date -- the day
the number actually became public. That is what makes point-in-time
correctness possible downstream and prevents look-ahead bias.

AssetsCurrent / LiabilitiesCurrent are INSTANT balance-sheet facts (they
support current_ratio downstream) and need no duration filtering or
cumulative differencing in 02b_clean_fundamentals.py.
EarningsPerShareDiluted is pulled but not currently used as a model
input -- kept as a raw diagnostic column.

StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest is
pulled alongside plain StockholdersEquity because at least one filer
(CWT) stops tagging the plain version once it begins carrying a
noncontrolling interest in equity (e.g. from a JV/subsidiary) and
switches to this NCI-inclusive tag instead -- without it, roe/
debt_to_equity would forward-fill a stale value for many consecutive
quarters. 02b resolves between the two by priority (the same pattern
used for CapEx below), never a blind alias/rename, since the
NCI-inclusive figure can genuinely differ from the parent-only figure.

ProfitLoss is pulled alongside NetIncomeLoss for the same reason: some
filers (e.g. Ecolab) tag NetIncomeLoss only intermittently but ProfitLoss
(consolidated, including NCI) every quarter. 02b resolves these by
priority (NetIncomeLoss preferred, ProfitLoss as fallback) rather than
aliasing them, since the two figures can genuinely differ once a filer
carries a noncontrolling interest.

PaymentsToAcquireOtherProductiveAssets is pulled alongside
PaymentsToAcquirePropertyPlantAndEquipment because some filers (e.g.
Roper Technologies) split CapEx reporting across both tags with only
partial overlap in coverage -- resolved by priority in CAPEX_PRIORITY,
same as every other capex tag, never summed.

The universe here is 28 tickers: the original 8 regulated water
utilities plus 20 water-adjacent equipment/infrastructure/industrial
names, each verified via verify_candidate_tickers.py
(scripts/verifications.py) before being added.

Output: data/raw/fundamentals_raw.csv
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
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

TICKERS = [
    # Original 8-ticker universe (regulated water utilities + water tech/infra)
    "AWK", "WTRG", "CWT", "AWR", "XYL", "VRT", "JCI", "BMI",
    # Added in the 8 -> 28 expansion, all verified via
    # verify_candidate_tickers.py (output/tables/candidate_verification_20260808.csv).
    # Water/fluid equipment manufacturers:
    "PNR", "MWA", "AOS", "ITT", "IEX", "FELE", "FLS", "DOV", "ECL", "GRC",
    # Water metering / plumbing:
    "ITRI", "MAS",
    # Diversified industrials with meaningful water/fluid segments:
    "ROP", "EMR", "PH", "HON",
    # Infrastructure construction incl. water/wastewater/drainage:
    "WMS", "GVA", "PWR", "MTZ",
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

# SEC's ticker->CIK file keys on current registered name, so a recent
# rename can break the automatic lookup (this is what happened with
# SJW, which now files as "H2O America"). Add any ticker the scraper
# warns about here, verified at sec.gov.
MANUAL_CIK_OVERRIDES = {}

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
SEC_REQUEST_DELAY = 0.15  # SEC asks for <=10 req/sec; stay well under


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