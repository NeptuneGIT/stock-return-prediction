"""
get_dividends_fallback.py

WORKAROUND, not a permanent architectural change. Prefer the dividend
step inside 01_get_data.R (tq_get(STOCKS, get = "dividends", ...)) --
only use this script if that step fails.

WHY THIS EXISTS: while running 01_get_data.R for the 8 -> 28 ticker
expansion, the R-native dividend fetch (tidyquant -> quantmod::getDividends,
which hits Yahoo's legacy v7/finance/download endpoint) either returned
zero rows or crashed the R process outright (segfault, exit code 139 --
not a catchable R-level error). This was root-caused, not just worked
around blindly:
  - Confirmed NOT a rate-limit or network issue: a plain price-only
    request to Yahoo's modern v8/finance/chart endpoint succeeded in the
    same session.
  - Confirmed NOT a malformed-data issue: the exact same dividend JSON
    payload, saved to a local file and parsed with Python's stdlib json
    module, parses cleanly (e.g. AWK's 2013-05-22 $0.28 dividend --
    timestamp 1369229400 -- matches the pre-existing, git-committed
    dividends_raw.csv exactly).
  - Confirmed the crash reproduces even with NO network call at all:
    R's jsonlite::fromJSON() segfaults parsing that same locally-saved
    JSON file, with both default settings and simplifyVector = FALSE.
This isolates the bug to R's jsonlite (or its interaction with R/curl on
this machine) choking on the nested, timestamp-keyed "dividends" object
Yahoo's v8 chart API returns -- independent of quantmod, of tidyquant, of
network conditions, and of which ticker is requested. Not something to
fix by retrying, by changing the ticker list, or by patching this
project's R scripts; fixing it for real would mean upgrading R packages
in the user's environment, which is out of scope for a data-pipeline
change and wasn't requested.

SAME SOURCE, SAME SEMANTICS, SAME OUTPUT SCHEMA as the R path it
replaces -- this changes HOW the data is fetched, not WHAT is fetched.
Still Yahoo Finance actual ex-dividend CASH events (never SEC's
*declared* dividends -- see the rationale in 01_get_data.R's docstring
for why that distinction matters and must not be blurred across
tickers). Output is byte-for-byte the same shape as 01_get_data.R would
have written: data/raw/dividends_raw.csv with columns
symbol,date,dividend, sorted by symbol then date, dividend > 0 only.

If a future session's R environment no longer segfaults on this (e.g.
after a jsonlite/R upgrade), delete this script and rely on
01_get_data.R alone again -- don't maintain two dividend-fetch paths
longer than necessary.

Output: data/raw/dividends_raw.csv
"""

import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research script)"}

# Must stay in sync with STOCKS in 01_get_data.R / TICKERS in
# 02_fundamentals.py. Benchmarks (PHO, SPY) deliberately excluded, same
# as the R script -- only the 28 tradeable tickers need dividend history.
STOCKS = [
    "AWK", "WTRG", "CWT", "AWR", "XYL", "VRT", "JCI", "BMI",
    "PNR", "MWA", "AOS", "ITT", "IEX", "FELE", "FLS", "DOV", "ECL", "GRC",
    "ITRI", "MAS", "ROP", "EMR", "PH", "HON", "WMS", "GVA", "PWR", "MTZ",
]

START_DATE = "2013-01-01"
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

    # A ticker absent here means either a fetch failure OR a company that
    # genuinely never paid a dividend in the queried window -- this
    # script can't tell those apart from an empty response alone (a real
    # HTTP failure raises and is caught above; a confirmed-empty
    # "dividends" object looks identical whether Yahoo has no data
    # because none exists, or because of some silent gap on their end).
    # ITRI (Itron) and MTZ (MasTec) are both well-known non-dividend-
    # payers (growth-reinvestment industrials), which is almost
    # certainly why they show 0 events here, not a fetch problem -- but
    # 01a_ratios.R's paying_tickers mechanism can only mark a ticker as
    # a confirmed 0%-yield payer if it has AT LEAST ONE payment row, so
    # a true never-payer is indistinguishable from a fetch gap under
    # that mechanism and is left as NA (not 0) for div_yield, per the
    # project's "missing != zero" default. This slightly understates
    # data completeness for these two tickers but never fabricates a
    # false zero -- flagged here rather than silently accepted.
    if missing:
        print(f"NO DIVIDEND DATA for: {sorted(missing)} -- verify manually "
              f"before treating their yield as zero or as a fetch failure.")


if __name__ == "__main__":
    main()
