"""
get_dividends_fallback.py -- fallback dividend fetcher, only needed if
01_get_data.R's dividend step fails.

R's jsonlite::fromJSON() segfaults the R process (exit 139, not
catchable) parsing Yahoo's dividend JSON in some environments -- not
rate-limiting, not network, not the ticker list, confirmed a jsonlite
bug. This hits the same Yahoo endpoint via requests/pandas instead,
same output schema: symbol,date,dividend, sorted, dividend > 0 only.
Same source, same semantics -- real ex-dividend cash events, never SEC's
declared dividends.

Output: data/raw/dividends_raw.csv
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research script)"}

STOCKS = [
    # Original 8-ticker universe
    "AWK", "WTRG", "CWT", "AWR", "XYL", "VRT", "JCI", "BMI",
    # Added in the 8 -> 28 expansion
    "PNR", "MWA", "AOS", "ITT", "IEX", "FELE", "FLS", "DOV", "ECL", "GRC",
    "ITRI", "MAS", "ROP", "EMR", "PH", "HON", "WMS", "GVA", "PWR", "MTZ",
    # --- Information Technology ---
    "MSFT", "AAPL", "NVDA", "ORCL", "CRM", "ADBE", "CSCO", "AMD", "QCOM",
    "TXN", "IBM", "INTU", "AMAT", "ADI", "LRCX", "KLAC", "MU", "SNPS",
    "CDNS", "ADSK", "MCHP", "ON", "TER", "WDC", "STX", "NTAP", "TYL",
    "PTC", "SWKS", "GRMN", "ZBRA",
    # --- Health Care ---
    "JNJ", "UNH", "LLY", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "GILD", "CVS", "ELV", "HUM", "SYK", "BSX", "ISRG", "ZBH", "BDX", "BAX",
    "VRTX", "BIIB", "MCK", "COR", "HCA", "DVA",
    # --- Financials ---
    "C", "GS", "MS", "BNY", "SCHW", "SPGI", "MCO", "ICE", "AON", "MRSH",
    "AJG", "ALL", "PGR", "HIG", "STT", "FITB", "AXP", "COF", "NDAQ",
    "MSCI", "FDS", "CBOE", "NTRS", "WTW",
    # --- Consumer Discretionary ---
    "AMZN", "TSLA", "HD", "MCD", "LOW", "SBUX", "TJX", "ORLY", "AZO",
    "ROST", "YUM", "MAR", "HLT", "GM", "APTV", "BBY", "DHI", "PHM", "NVR",
    "WHR", "TSCO", "ULTA", "GPC", "CMG",
    # --- Communication Services ---
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
    # --- Materials  ---
    "APD", "SHW", "FCX", "NEM", "PPG", "NUE", "ALB", "CE", "IFF", "MLM",
    "VMC", "IP",
]

START_DATE = "2003-01-01"
OUT_PATH = Path("data/raw/dividends_raw.csv")
REQUEST_DELAY = 0.3


def fetch_dividends(ticker):
    p1 = int(datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    p2 = int(datetime.now(timezone.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div"
    )
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    js = resp.json()

    result = js.get("chart", {}).get("result")
    if not result:
        return []
    divs = result[0].get("events", {}).get("dividends", {})

    rows = []
    for entry in divs.values():
        amount = entry.get("amount")
        ts = entry.get("date")
        if amount is None or ts is None or amount <= 0:
            continue
        d = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        rows.append({"symbol": ticker, "date": d.isoformat(), "dividend": amount})
    return rows


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    missing = []

    for ticker in STOCKS:
        try:
            rows = fetch_dividends(ticker)
        except requests.HTTPError as err:
            print(f"  FAILED for {ticker}: {err}")
            missing.append(ticker)
            time.sleep(REQUEST_DELAY)
            continue
        print(f"{ticker}: {len(rows)} dividend events")
        if not rows:
            missing.append(ticker)
        all_rows.extend(rows)
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_rows).sort_values(["symbol", "date"])
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} dividend rows for {df['symbol'].nunique()} tickers -> {OUT_PATH}")


    if missing:
        print(f"NO DIVIDEND DATA for: {sorted(missing)} -- verify manually "
              f"before treating their yield as zero or as a fetch failure.")


if __name__ == "__main__":
    main()
