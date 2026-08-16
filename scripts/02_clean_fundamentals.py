"""
02_clean_fundamentals.py -- turns raw SEC XBRL facts into a model-ready
quarterly panel. A number only belongs here if a trader could've known
it at the time.

"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/fundamentals_raw.csv")
OUT_PATH = Path("data/processed/fundamentals_clean.csv")

# Starts earlier than 01a_ratios.R's panel STUDY_START (2010-01-01) so
# TTM calcs and forward-fills have history to draw on first.
STUDY_START = "2003-01-01"
STUDY_END = "2026-12-31"

_SP500_PIVOT_CAPEX_OR_STRUCTURAL_REJECTS = {
    "ABBV", "ACN", "AFL", "AIG", "AVGO", "BAC", "BKNG", "BKR", "BLK",
    "CB", "CHTR", "CI", "CMCSA", "CME", "COP", "DD", "DIS", "DLR",
    "DOW", "ESS", "EXPE", "F", "FANG", "GOOGL", "HSY", "JPM", "KHC",
    "LEN", "LIN", "MAA", "MDT", "MET", "META", "MKC", "NEE", "NKE",
    "NXPI", "PLD", "PNC", "PRU", "PSX", "REGN", "SPG", "STZ", "SYY",
    "TAP", "TFC", "TRV", "TSN", "UPS", "USB", "WBD", "WEC", "WFC", "XOM",
    # Supplemental round (targeting the thin Financials/Real Estate/
    # Communication Services sectors): same two failure patterns.
    "V", "MA", "PYPL", "SYF", "BX", "KKR", "REG", "ARE", "KIM", "FRT",
    "CPT", "INVH", "VICI", "FOXA", "NWSA",
}

_SP500_PIVOT_DELISTED_OR_ACQUIRED = {"JNPR", "IPG", "HES", "K", "DFS", "PARA"}

EXCLUDED_TICKERS = (
    {"MSEX", "SJW", "WTS", "CR"}
    | _SP500_PIVOT_CAPEX_OR_STRUCTURAL_REJECTS
    | _SP500_PIVOT_DELISTED_OR_ACQUIRED
)

FLOW_TAGS = {
    "NetIncomeLoss",
    "ProfitLoss",  # fallback for NetIncomeLoss -- see NET_INCOME_PRIORITY
    "Revenues",
    "EarningsPerShareDiluted",
    "OperatingCashFlow",
    "CapEx",
}

REVENUE_ALIASES = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenues",
    "SalesRevenueNet": "Revenues",
    "SalesRevenueGoodsNet": "Revenues",
}

OCF_ALIASES = {
    "NetCashProvidedByUsedInOperatingActivities": "OperatingCashFlow",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": "OperatingCashFlow",
}

CAPEX_PRIORITY = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    # ROP splits capex across these two tags, only 5/54 quarters overlap
    # -- unioned covers full history, alone neither does. Same priority
    # resolution, never summed.
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
]

EQUITY_PRIORITY = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]

NET_INCOME_PRIORITY = [
    "NetIncomeLoss",
    "ProfitLoss",
]

COVERPAGE_TAGS = {"EntityCommonStockSharesOutstanding"}

QUARTER_DAYS = (80, 100)
HALF_DAYS = (170, 190)
NINEMO_DAYS = (260, 280)
ANNUAL_DAYS = (355, 375)


def load_raw():
    df = pd.read_csv(RAW_PATH)
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["tag"] = df["tag"].replace(REVENUE_ALIASES).replace(OCF_ALIASES)
    # CapEx tags keep their own names through the pivot but are still
    # period-flow facts, so they need duration filtering too.
    FLOW_TAGS.update(CAPEX_PRIORITY)
    df = df[~df["ticker"].isin(EXCLUDED_TICKERS)]
    # A ticker may report the same period under two aliased tags; keep one.
    df = df.drop_duplicates(subset=["ticker", "tag", "start", "end", "filed"])
    print(
        f"Loaded {len(df)} facts | {df['ticker'].nunique()} tickers: "
        f"{sorted(df['ticker'].unique())}"
    )
    return df


def infer_fiscal_year_starts(df):
    fy_starts = {}
    for ticker, grp in df.groupby("ticker"):
        dur = (grp["end"] - grp["start"]).dt.days
        annual = grp[dur.between(*ANNUAL_DAYS)]
        fy_starts[ticker] = (
            int(annual["start"].dt.month.mode().iloc[0]) if len(annual) else 1
        )
    print(f"Fiscal-year start months: {fy_starts}")
    return fy_starts


def derive_quarters_from_cumulative(df, fy_starts):
    diff_tags = FLOW_TAGS - {"EarningsPerShareDiluted"}
    flow = df[df["tag"].isin(diff_tags)].copy()
    flow = flow.sort_values("filed").drop_duplicates(
        subset=["ticker", "tag", "start", "end"], keep="first"
    )

    derived = []
    for (ticker, tag), grp in flow.groupby(["ticker", "tag"]):
        grp = grp.copy()
        grp["dur"] = (grp["end"] - grp["start"]).dt.days

        already_have = set(grp.loc[grp["dur"].between(*QUARTER_DAYS), "end"])
        cumulative = grp[grp["start"].dt.month == fy_starts.get(ticker, 1)]

        for _, fy_rows in cumulative.groupby("start"):

            def pick(window):
                match = fy_rows[fy_rows["dur"].between(*window)]
                return match.iloc[0] if len(match) else None

            q1, h1, m9, fy = (
                pick(QUARTER_DAYS),
                pick(HALF_DAYS),
                pick(NINEMO_DAYS),
                pick(ANNUAL_DAYS),
            )

            for minuend, subtrahend, label in (
                (h1, q1, "H1-Q1"),
                (m9, h1, "9mo-H1"),
                (fy, m9, "FY-9mo"),
            ):
                if minuend is None or subtrahend is None:
                    continue
                # Reported end date, not a synthetic calendar quarter-end
                # -- 52/53-week fiscal calendars drift.
                if minuend["end"] in already_have:
                    continue
                derived.append(
                    {
                        "ticker": ticker,
                        "tag": tag,
                        "end": minuend["end"],
                        "start": subtrahend["end"],
                        "filed": minuend["filed"],
                        "val": minuend["val"] - subtrahend["val"],
                    }
                )

    derived_df = pd.DataFrame(derived)
    print(f"Derived {len(derived_df)} quarterly values from cumulative figures")
    return derived_df


def keep_single_quarter_flows(df):

    is_flow = df["tag"].isin(FLOW_TAGS)
    dur = (df["end"] - df["start"]).dt.days
    keep = (~is_flow) | dur.between(*QUARTER_DAYS)
    print(f"Dropped {(~keep).sum()} cumulative/annual flow rows")
    return df[keep].copy()


def dedupe_point_in_time(df):

    before = len(df)
    df = df.sort_values("filed").drop_duplicates(
        subset=["ticker", "tag", "end"], keep="first"
    )
    print(f"Dropped {before - len(df)} restated/comparative duplicates")
    return df


def pivot_wide(df):
    main = df[~df["tag"].isin(COVERPAGE_TAGS)]
    cover = df[df["tag"].isin(COVERPAGE_TAGS)]

    wide = main.pivot_table(
        index=["ticker", "end"], columns="tag", values="val", aggfunc="first"
    ).reset_index()

    feature_tags = [
        "Assets", "Liabilities",
    ] + NET_INCOME_PRIORITY + EQUITY_PRIORITY
    core = main[main["tag"].isin(feature_tags)]
    filed = core.groupby(["ticker", "end"])["filed"].max().reset_index()
    # Fall back to the all-tag max for any quarter with no core tags.
    fallback = main.groupby(["ticker", "end"])["filed"].max().reset_index()
    filed = fallback.merge(
        filed, on=["ticker", "end"], how="left", suffixes=("_all", "")
    )
    filed["filed"] = filed["filed"].fillna(filed["filed_all"])
    wide = wide.merge(filed[["ticker", "end", "filed"]], on=["ticker", "end"], how="left")


    wide["filed_is_approximated"] = wide["filed"].isna()
    n_approx = int(wide["filed_is_approximated"].sum())
    if n_approx:
        print(f"WARNING: {n_approx} rows had no filing date at all -- "
              f"approximated as end + 45 days. Review these before use.")
        wide["filed"] = wide["filed"].fillna(wide["end"] + pd.Timedelta(days=45))

    if not cover.empty:
        cp = cover.pivot_table(
            index=["ticker", "end"], columns="tag", values="val", aggfunc="first"
        ).reset_index()

        wide = pd.merge_asof(
            wide.sort_values("end"),
            cp.sort_values("end"),
            on="end",
            by="ticker",
            direction="forward",
            tolerance=pd.Timedelta("60D"),
        )

    return wide.sort_values(["ticker", "end"]).reset_index(drop=True)



FCF_EXCLUDED_TICKERS = {"SJW"}


def add_quarter_index(df):
    df = df.reset_index(drop=True).copy()
    df["_qidx"] = df["end"].dt.year * 4 + df["end"].dt.quarter
    dupes = df.duplicated(subset=["ticker", "_qidx"]).sum()
    if dupes:
 
        print(f"NOTE: {dupes} ticker rows share a calendar quarter (fiscal drift)")
    return df


def lag_quarters(df, col, n):

    lookup = df[["ticker", "_qidx", col]].drop_duplicates(
        subset=["ticker", "_qidx"], keep="first"
    ).copy()
    lookup["_qidx"] = lookup["_qidx"] + n
    lookup = lookup.rename(columns={col: "_lagged"})
    merged = df[["ticker", "_qidx"]].merge(lookup, on=["ticker", "_qidx"], how="left")
    return pd.to_numeric(merged["_lagged"], errors="coerce").values


def ttm_sum(df, col):

    total = pd.to_numeric(df[col], errors="coerce").copy()
    for k in (1, 2, 3):
        total = total + pd.Series(lag_quarters(df, col, k), index=df.index)
    return total


def local_median_scale(values, tickers, ends, window=13, min_periods=5):

    values = pd.to_numeric(values, errors="coerce")
    frame = pd.DataFrame({"ticker": tickers.values, "end": ends.values, "val": values.abs()},
                          index=values.index)
    ordered = frame.sort_values(["ticker", "end"])
    local_median = ordered.groupby("ticker")["val"].transform(
        lambda s: s.rolling(window, center=True, min_periods=min_periods).median()
    )
    return local_median.reindex(frame.index)


def flag_scale_outliers(values, tickers, ends, label, factor=50, window=13, min_periods=5):
    values = pd.to_numeric(values, errors="coerce")
    local_median = local_median_scale(values, tickers, ends, window=window, min_periods=min_periods)
    ratio = values.abs() / local_median
    bad = (
        values.notna()
        & local_median.notna()
        & (local_median > 0)
        & ((ratio > factor) | (ratio < 1 / factor))
    )
    n_bad = int(bad.sum())
    if n_bad:
        rows = ", ".join(f"{t}@{e}" for t, e in zip(tickers[bad], ends[bad]))
        print(
            f"WARNING: {n_bad} {label} value(s) off by >{factor}x vs that "
            f"ticker's own NEARBY-QUARTER history (likely a filer "
            f"unit-scale/tagging error, not real growth/shrinkage) -- "
            f"nulled rather than guessed at. Rows: {rows}"
        )
    return values.mask(bad)


def compute_features(df):

    df = add_quarter_index(df)

    # Share count fallback, priority-ordered. SharesIssued only fills
    # remaining gaps -- overstates count when treasury stock exists.
    shares = None
    for col in (
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesIssued",
    ):
        if col in df.columns:
            shares = df[col] if shares is None else shares.fillna(df[col])

    # Unit-scale check: cross-check against EntityCommonStockShares
    # Outstanding (cover-page fact). >10x disagreement means one tag is
    # wrong scale -- either could be it, so arbitrate with
    # local_median_scale(): whichever's closer to the ticker's own
    # nearby-quarter scale wins.
    if "EntityCommonStockSharesOutstanding" in df.columns:
        cover = df["EntityCommonStockSharesOutstanding"]
        ratio = shares / cover
        disagree = shares.notna() & cover.notna() & ((ratio > 10) | (ratio < 0.1))
        n_bad = int(disagree.sum())
        if n_bad:
            ticker_scale = local_median_scale(shares, df["ticker"], df["end"])
            prefer_cover = disagree & (
                (cover.abs() - ticker_scale).abs() < (shares.abs() - ticker_scale).abs()
            )
            n_prefer_cover = int(prefer_cover.sum())
            print(
                f"WARNING: {n_bad} shares_outstanding value(s) off by >10x vs "
                f"EntityCommonStockSharesOutstanding (a units error in ONE of "
                f"the two figures, not necessarily the cover page) -- "
                f"substituting whichever is closer to that ticker's own "
                f"nearby-quarter scale: cover-page figure used for "
                f"{n_prefer_cover} row(s), priority-tag figure kept for "
                f"{n_bad - n_prefer_cover} row(s). Rows using the cover page: "
                + ", ".join(
                    f"{t}@{e}" for t, e in
                    zip(df.loc[prefer_cover, "ticker"], df.loc[prefer_cover, "end"])
                )
            )
            shares = shares.mask(prefer_cover, cover)


    shares = flag_scale_outliers(shares, df["ticker"], df["end"], "shares_outstanding")
    implausible = shares.notna() & (shares < 1_000_000)
    n_implausible = int(implausible.sum())
    if n_implausible:
        rows = ", ".join(
            f"{t}@{e}" for t, e in
            zip(df.loc[implausible, "ticker"], df.loc[implausible, "end"])
        )
        print(
            f"WARNING: {n_implausible} shares_outstanding value(s) below the "
            f"absolute 1,000,000-share floor for this S&P-500-only universe "
            f"(likely a sustained thousands/millions-scale filer error, or "
            f"an impossible negative value) -- nulled rather than guessed "
            f"at. Rows: {rows}"
        )
        shares = shares.mask(implausible)

    df["shares_outstanding"] = shares

    equity = None
    equity_source = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in EQUITY_PRIORITY:
        if col not in df.columns:
            continue
        if equity is None:
            equity = df[col].copy()
            equity_source = equity_source.mask(df[col].notna(), col)
        else:
            newly = equity.isna() & df[col].notna()
            equity = equity.fillna(df[col])
            equity_source = equity_source.mask(newly, col)
    if equity is None:
        print("WARNING: no StockholdersEquity tag (plain or NCI-inclusive) found")
        df["StockholdersEquity"] = pd.NA
    else:
        if len(EQUITY_PRIORITY) == 2 and all(c in df.columns for c in EQUITY_PRIORITY):
            tag_a = pd.to_numeric(df[EQUITY_PRIORITY[0]], errors="coerce")
            tag_b = pd.to_numeric(df[EQUITY_PRIORITY[1]], errors="coerce")
            both_present = tag_a.notna() & tag_b.notna()
            tag_ratio = tag_a / tag_b
            disagree = both_present & ((tag_ratio.abs() > 10) | (tag_ratio.abs() < 0.1))
            if disagree.any():
                ticker_scale = local_median_scale(equity, df["ticker"], df["end"])
                prefer_b = disagree & (
                    (tag_b.abs() - ticker_scale).abs()
                    < (tag_a.abs() - ticker_scale).abs()
                )
                n_swapped = int(prefer_b.sum())
                if n_swapped:
                    rows = ", ".join(
                        f"{t}@{e}" for t, e in
                        zip(df.loc[prefer_b, "ticker"], df.loc[prefer_b, "end"])
                    )
                    print(
                        f"WARNING: {n_swapped} {EQUITY_PRIORITY[0]} value(s) disagree "
                        f"with {EQUITY_PRIORITY[1]} by >10x in the same filing -- "
                        f"substituting the tag closer to that ticker's own "
                        f"historical scale. Rows: {rows}"
                    )
                    equity = equity.mask(prefer_b, tag_b)
                    equity_source = equity_source.mask(
                        prefer_b, EQUITY_PRIORITY[1] + "_corrected"
                    )


        equity = flag_scale_outliers(equity, df["ticker"], df["end"], "StockholdersEquity")
        df["StockholdersEquity"] = equity
    df["equity_source_tag"] = equity_source

    if "Assets" in df.columns:
        df["Assets"] = flag_scale_outliers(df["Assets"], df["ticker"], df["end"], "Assets")
    if "Liabilities" in df.columns:
        df["Liabilities"] = flag_scale_outliers(
            df["Liabilities"], df["ticker"], df["end"], "Liabilities (directly tagged)"
        )


    derived_liab = df.get("Assets") - df.get("StockholdersEquity")
    if "Liabilities" in df.columns:
        n = int(df["Liabilities"].isna().sum())
        df["Liabilities"] = df["Liabilities"].fillna(derived_liab)
    else:
        n = len(df)
        df["Liabilities"] = derived_liab
    print(f"Derived {n} Liabilities from Assets - StockholdersEquity")

    net_income = None
    net_income_source = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in NET_INCOME_PRIORITY:
        if col not in df.columns:
            continue
        if net_income is None:
            net_income = df[col].copy()
            net_income_source = net_income_source.mask(df[col].notna(), col)
        else:
            newly = net_income.isna() & df[col].notna()
            net_income = net_income.fillna(df[col])
            net_income_source = net_income_source.mask(newly, col)
    if net_income is None:
        print("WARNING: no NetIncomeLoss tag (parent-only or consolidated) found")
        df["NetIncomeLoss"] = pd.NA
    else:
        df["NetIncomeLoss"] = net_income
    df["net_income_source_tag"] = net_income_source

    df["roe"] = df["NetIncomeLoss"] / df["StockholdersEquity"]
    df["debt_to_equity"] = df["Liabilities"] / df["StockholdersEquity"]

    capex = None
    capex_source = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in CAPEX_PRIORITY:
        if col not in df.columns:
            continue
        if capex is None:
            capex = df[col].copy()
            capex_source = capex_source.mask(df[col].notna(), col)
        else:
            newly = capex.isna() & df[col].notna()
            capex = capex.fillna(df[col])
            capex_source = capex_source.mask(newly, col)

    if capex is None:
        print("WARNING: no CapEx tags found -- free_cash_flow not computed")
        df["free_cash_flow"] = pd.NA
        df["capex"] = pd.NA
        df["capex_source_tag"] = pd.NA
    else:

        excluded = df["ticker"].isin(FCF_EXCLUDED_TICKERS)
        capex = capex.mask(excluded, pd.NA)
        capex_source = capex_source.mask(excluded, "EXCLUDED_no_true_capex_tag")
        if excluded.any():
            print(
                f"Excluded free_cash_flow for {sorted(df.loc[excluded, 'ticker'].unique())} "
                f"({int(excluded.sum())} rows) -- no genuine aggregate capex tag; "
                f"see FCF_EXCLUDED_TICKERS"
            )

        # SEC reports cash outflows as positive under "Payments..." tags,
        # so capex gets subtracted as an absolute value -- otherwise a
        # filer reporting it signed negative would inflate FCF.
        df["capex"] = pd.to_numeric(capex, errors="coerce").abs()
        df["capex_source_tag"] = capex_source

        if "OperatingCashFlow" in df.columns:
            df["free_cash_flow"] = df["OperatingCashFlow"] - df["capex"]
            n_fcf = int(df["free_cash_flow"].notna().sum())
            print(f"Computed free_cash_flow for {n_fcf}/{len(df)} rows")
            print(f"CapEx tag usage:\n{capex_source.value_counts().to_string()}")
        else:
            print("WARNING: no OperatingCashFlow tag found -- free_cash_flow not computed")
            df["free_cash_flow"] = pd.NA

    if "AssetsCurrent" in df.columns and "LiabilitiesCurrent" in df.columns:
        lc = pd.to_numeric(df["LiabilitiesCurrent"], errors="coerce")
        ac = pd.to_numeric(df["AssetsCurrent"], errors="coerce")
        df["current_ratio"] = np.where(lc > 0, ac / lc, np.nan)
    else:
        print(
            "WARNING: AssetsCurrent/LiabilitiesCurrent not in the raw file -- "
            "current_ratio not computed. Did you re-run 02a after adding them?"
        )
        df["current_ratio"] = np.nan
        for c in ("AssetsCurrent", "LiabilitiesCurrent"):
            if c not in df.columns:
                df[c] = np.nan

    assets = pd.to_numeric(df["Assets"], errors="coerce")
    assets_lag4 = pd.Series(lag_quarters(df, "Assets", 4), index=df.index)
    df["asset_growth"] = np.where(assets_lag4 > 0, assets / assets_lag4 - 1, np.nan)

    df["ni_ttm"] = ttm_sum(df, "NetIncomeLoss")
    df["capex_ttm"] = ttm_sum(df, "capex") if "capex" in df.columns else np.nan

    ni_ttm_lag4 = pd.Series(lag_quarters(df, "ni_ttm", 4), index=df.index)
    df["ni_growth"] = np.where(
        ni_ttm_lag4 > 0, df["ni_ttm"] / ni_ttm_lag4 - 1, np.nan
    )


    df["capex_intensity"] = np.where(
        assets > 0, pd.to_numeric(df["capex_ttm"], errors="coerce") / assets, np.nan
    )

    return df.drop(columns=["_qidx"])


def forward_fill_within_ticker(df, cols):
    """Carry the last known value forward per ticker, like an analyst
    would. Never looks forward, so safe for backtesting. Flags filled
    cells for audit later.

    Fills the RATIOS, not raw inputs -- filling raw Assets first would
    make asset_growth compare a real figure against a stale copy of
    itself and report a fake 0% growth."""
    df = df.sort_values(["ticker", "end"])
    for col in cols:
        if col not in df.columns:
            continue
        df[f"{col}_is_filled"] = df[col].isna()
        df[col] = df.groupby("ticker")[col].ffill()
    return df


def flag_outliers(df):
    """Flag, don't drop -- automated dropping can itself bias the panel."""
    df["flag_negative_equity"] = df["StockholdersEquity"] < 0
    df["flag_extreme_de"] = df["debt_to_equity"].abs() > 10
    df["flag_extreme_roe"] = df["roe"].abs() > 1

    df["flag_extreme_asset_growth"] = df["asset_growth"].abs() > 1
    df["flag_extreme_ni_growth"] = df["ni_growth"].abs() > 3
    flags = [
        "flag_negative_equity",
        "flag_extreme_de",
        "flag_extreme_roe",
        "flag_extreme_asset_growth",
        "flag_extreme_ni_growth",
    ]
    print(f"Flagged {int(df[flags].any(axis=1).sum())} rows for manual review")
    return df


NEW_FEATURES = [
    "current_ratio",
    "asset_growth",
    "ni_growth",
    "capex_intensity",
]

FILLED_COLS = [
    "roe",
    "debt_to_equity",
    "shares_outstanding",
    "free_cash_flow",
] + NEW_FEATURES


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    fy_starts = infer_fiscal_year_starts(df)

    derived = derive_quarters_from_cumulative(df, fy_starts)
    if not derived.empty:
        df = pd.concat([df, derived], ignore_index=True)

    df = keep_single_quarter_flows(df)
    df = dedupe_point_in_time(df)

    wide = pivot_wide(df)
    wide = compute_features(wide)


    print("\nMissingness in NEW features, BEFORE forward-fill (true coverage):")
    print(wide[NEW_FEATURES].isna().mean().round(3).to_string())

    wide = forward_fill_within_ticker(wide, FILLED_COLS)
    wide = flag_outliers(wide)

    wide = wide[(wide["end"] >= STUDY_START) & (wide["end"] <= STUDY_END)]

    keep_cols = [
        "ticker", "end", "filed", "filed_is_approximated",
        # --- model features ---
        "roe", "debt_to_equity",
        "current_ratio", "asset_growth", "ni_growth",
        "capex_intensity",
        # --- inputs the R script still needs for price-based ratios ---
        "shares_outstanding", "free_cash_flow",
        # --- raw / diagnostic ---
        "capex", "capex_ttm", "OperatingCashFlow", "capex_source_tag",
        "equity_source_tag", "net_income_source_tag",
        "ni_ttm",
        "roe_is_filled", "debt_to_equity_is_filled", "shares_outstanding_is_filled",
        "free_cash_flow_is_filled", "current_ratio_is_filled",
        "asset_growth_is_filled", "ni_growth_is_filled",
        "capex_intensity_is_filled",
        "Assets", "AssetsCurrent", "Liabilities", "LiabilitiesCurrent",
        "StockholdersEquity",
        "NetIncomeLoss", "EarningsPerShareDiluted", "Revenues",
        "flag_negative_equity", "flag_extreme_de", "flag_extreme_roe",
        "flag_extreme_asset_growth", "flag_extreme_ni_growth",
    ]
    wide = wide[[c for c in keep_cols if c in wide.columns]]
    wide.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(wide)} rows -> {OUT_PATH}")

    model_cols = ["roe", "debt_to_equity"] + NEW_FEATURES
    print("\nMissingness in model features (post forward-fill):")
    print(wide[model_cols].isna().mean().round(3).to_string())


    complete = wide[model_cols].notna().all(axis=1).mean()
    print(f"\nRows with ALL fundamental features present: {complete:.1%}")
    print("  (glmnet/pls will refuse to run on the rest.)")

    print("\nPer-ticker coverage:")
    summary = wide.groupby("ticker").agg(
        quarters=("end", "count"),
        first=("end", "min"),
        last=("end", "max"),
        roe_miss=("roe", lambda s: round(s.isna().mean(), 2)),
        de_miss=("debt_to_equity", lambda s: round(s.isna().mean(), 2)),
        curr_miss=("current_ratio", lambda s: round(s.isna().mean(), 2)),
        agrow_miss=("asset_growth", lambda s: round(s.isna().mean(), 2)),
        nigrow_miss=("ni_growth", lambda s: round(s.isna().mean(), 2)),
        capint_miss=("capex_intensity", lambda s: round(s.isna().mean(), 2)),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()