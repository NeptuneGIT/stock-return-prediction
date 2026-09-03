"""
02_fundamentals.py

Pulls raw fundamental facts from SEC EDGAR's free company-facts API for
the 222-ticker S&P 500 universe (TICKERS below).

This script ONLY downloads. All cleaning happens in 02_clean_fundamentals.py,
so you can re-clean without re-hitting the SEC API.

Key design point: every fact is stored with its `filed` date -- the day
the number actually became public. That is what makes point-in-time
correctness possible downstream and prevents look-ahead bias.

Requires the SEC_USER_AGENT environment variable (see get_sec_headers()
below) -- SEC EDGAR requires a real contact string in the User-Agent header
on every request and will reject unidentified traffic.

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

TICKERS = [

    # Original 28-ticker water-sector universe (unchanged by the pivot).

    # Original 8-ticker universe (regulated water utilities + water tech/infra)
    "AWK", "WTRG", "CWT", "AWR", "XYL", "VRT", "JCI", "BMI",
    # Added in the 8 -> 28 expansion, all verified via
    # verify_candidate_tickers.py (scripts/verifications.py).
    # Water/fluid equipment manufacturers:
    "PNR", "MWA", "AOS", "ITT", "IEX", "FELE", "FLS", "DOV", "ECL", "GRC",
    # Water metering / plumbing:
    "ITRI", "MAS",
    # Diversified industrials with meaningful water/fluid segments:
    "ROP", "EMR", "PH", "HON",
    # Infrastructure construction incl. water/wastewater/drainage:
    "WMS", "GVA", "PWR", "MTZ",

    # --- Information Technology  ---
    "MSFT", "AAPL", "NVDA", "ORCL", "CRM", "ADBE", "CSCO", "AMD", "QCOM",
    "TXN", "IBM", "INTU", "AMAT", "ADI", "LRCX", "KLAC", "MU", "SNPS",
    "CDNS", "ADSK", "MCHP", "ON", "TER", "WDC", "STX", "NTAP", "TYL",
    "PTC", "SWKS", "GRMN", "ZBRA",
    # --- Health Care  ---
    "JNJ", "UNH", "LLY", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "GILD", "CVS", "ELV", "HUM", "SYK", "BSX", "ISRG", "ZBH", "BDX", "BAX",
    "VRTX", "BIIB", "MCK", "COR", "HCA", "DVA",
    # --- Financials  ---
    "C", "GS", "MS", "BNY", "SCHW", "SPGI", "MCO", "ICE", "AON", "MRSH",
    "AJG", "ALL", "PGR", "HIG", "STT", "FITB", "AXP", "COF", "NDAQ",
    "MSCI", "FDS", "CBOE", "NTRS", "WTW",
    # --- Consumer Discretionary  ---
    "AMZN", "TSLA", "HD", "MCD", "LOW", "SBUX", "TJX", "ORLY", "AZO",
    "ROST", "YUM", "MAR", "HLT", "GM", "APTV", "BBY", "DHI", "PHM", "NVR",
    "WHR", "TSCO", "ULTA", "GPC", "CMG",
    # --- Communication Services  ---
    "NFLX", "T", "VZ", "TMUS", "EA", "TTWO", "OMC", "LYV", "MTCH", "SIRI",
    # --- Industrials  ---
    "CAT", "DE", "UNP", "FDX", "BA", "LMT", "RTX", "GD", "NOC", "CSX",
    "NSC", "WM", "RSG", "GE",
    # --- Consumer Staples  ---
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "ADM", "CAG", "CLX", "CHD",
    # --- Energy  ---
    "CVX", "EOG", "SLB", "MPC", "VLO", "OXY", "WMB", "KMI", "OKE", "DVN",
    "HAL", "TRGP",
    # --- Utilities  ---
    "AEP", "DUK", "SO", "D", "EXC", "XEL", "ED", "PEG", "ES", "FE", "ETR",
    "EIX", "PPL",
    # --- Real Estate  ---
    "AMT", "EQIX", "CCI", "PSA", "O", "WELL", "AVB", "EQR", "VTR", "IRM",
    "UDR", "HST", "BXP",
    # --- Materials (13 total: 1 original + 12 new) ---
    "APD", "SHW", "FCX", "NEM", "PPG", "NUE", "ALB", "CE", "IFF", "MLM",
    "VMC", "IP",
]


MANUAL_CIK_OVERRIDES = {
    "AEP": "0000004904",
}


TAGS = [

    "Assets",
    "Liabilities",
    "StockholdersEquity",

    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",

    "AssetsCurrent",
    "LiabilitiesCurrent",
    "NetIncomeLoss",

    "ProfitLoss",
    "EarningsPerShareDiluted",  
   
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",

    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
    "CommonStockSharesIssued",

    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",

    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",

    "PaymentsToAcquireOtherProductiveAssets",
    "PaymentsForProceedsFromProductiveAssets",
    "PaymentsForCapitalImprovements",
    "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    "PaymentsToAcquireMachineryAndEquipment",
    "PaymentsToAcquireBuildings",
    "PaymentsForConstructionInProcess",
    "PaymentsToAcquireWaterAndWasteWaterSystems",  
    "PaymentsToAcquireOilAndGasPropertyAndEquipment",
    "PaymentsToAcquireEquipmentOnLease",
]


TAG_TAXONOMY = {
    "EntityCommonStockSharesOutstanding": "dei",
}

OUT_PATH = Path("data/raw/fundamentals_raw.csv")

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