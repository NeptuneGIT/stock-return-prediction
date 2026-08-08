"""
02b_clean_fundamentals.py

Turns raw SEC XBRL facts into a model-ready quarterly panel.

The whole job is producing numbers a trader could actually have known
at the time. Every step below exists to protect that property.

PIPELINE
  1. Collapse revenue tag aliases      (SEC taxonomy isn't consistent)
  2. Derive missing quarterly figures  (from cumulative YTD rows)
  3. Keep only true single-quarter flows
  4. Dedupe to FIRST filing            (point-in-time, not restated)
  5. Pivot to one row per ticker-quarter
  6. Derive Liabilities from the accounting identity where absent
  7. Compute ratios + growth features, forward-fill within ticker
  8. Flag outliers, trim to study window

Input:  data/raw/fundamentals_raw.csv
Output: data/processed/fundamentals_clean.csv

Several fields are resolved by PRIORITY across more than one SEC tag
rather than a single fixed tag name, because individual filers are
inconsistent about which tag they use:
  - StockholdersEquity (EQUITY_PRIORITY): plain StockholdersEquity wins
    when a filer reports it, falling back to
    StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest
    otherwise. CWT, for example, stops tagging the plain version after
    2021-03-31 -- without this fallback, roe/debt_to_equity (and
    Liabilities, always derived here via Assets - StockholdersEquity
    since CWT never tags Liabilities directly) would forward-fill a
    stale pre-2021 value indefinitely.
  - NetIncomeLoss (NET_INCOME_PRIORITY): plain NetIncomeLoss
    (parent-only) wins when reported, falling back to ProfitLoss
    (consolidated, including NCI) otherwise. Ecolab, for example, tags
    NetIncomeLoss for only a fraction of its quarters but ProfitLoss for
    all of them.
  - CapEx (CAPEX_PRIORITY): several payments-to-acquire tags are tried
    in priority order, never summed, since filers split capex reporting
    across multiple tags with only partial overlap (Roper Technologies,
    for example, needs two different tags to cover its full history).
  - shares_outstanding is cross-checked against
    EntityCommonStockSharesOutstanding for a >10x unit-scale mismatch
    (see compute_features()) -- individual filers have been observed to
    tag a share count in the wrong scale (e.g. thousands instead of raw
    shares) for a single filing that a later restatement corrects. Left
    uncaught, this kind of error can send pb_ratio/fcf_yield off by
    orders of magnitude for that one quarter, which is more than enough
    to distort a model's training fit for the whole panel. This is a
    generic filer-tagging risk, not specific to any one ticker, so the
    check runs panel-wide.

Derived features:
  current_ratio    = AssetsCurrent / LiabilitiesCurrent
  asset_growth     = Assets / Assets(4 quarters ago) - 1
  ni_growth        = NetIncomeLoss TTM YoY growth
  capex_intensity  = CapEx TTM / Assets

(asset_turnover, Revenues TTM / Assets, was tried and dropped: TTM
revenue needs four CONSECUTIVE tagged quarters, and at least one ticker
in this universe rarely has them, leaving asset_turnover mostly a stale
forward-filled value rather than a fresh observation for that name --
removed rather than kept as a near-constant column that couldn't move
the Broker Test's cross-sectional ranking anyway.)

The universe here is 28 tickers: the original 8 regulated water
utilities plus 20 water-adjacent names, each verified via
verify_candidate_tickers.py before being added; a couple of candidates
(WTS, CR) were considered and excluded -- see EXCLUDED_TICKERS below.

WHY THE REMAINING FOUR ARE COMPUTED HERE AND NOT IN 02_features.R
They depend only on fundamentals, so they must be computed on the
FUNDAMENTALS' OWN quarterly grid (keyed on `end`). 02_features.R joins
fundamentals to calendar quarter-ends by FILING date, and that join
repeats the same filing whenever no new one landed in a quarter.
Computing a year-over-year change after that join would compare a
filing against itself and report growth of exactly 0 -- a silent,
plausible-looking bug. Only price-dependent ratios (pb_ratio,
fcf_yield, div_yield) belong on the R side.

WHY LAGS USE A CALENDAR-QUARTER INDEX, NOT .shift(4)
.shift(4) means "four ROWS back", which equals "four quarters back"
only if the ticker has no gaps. VRT and CWT both have gaps. The
lag_quarters() helper below matches on an explicit year*4+quarter key,
so a missing quarter yields NA rather than a silently wrong 5-quarter
comparison.

NOTE ON operating_margin: still deliberately NOT computed. Revenue
coverage is too sparse and uneven across this universe to produce a
feature that means the same thing for every stock. This is the same
reason asset_turnover was removed above.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/fundamentals_raw.csv")
OUT_PATH = Path("data/processed/fundamentals_clean.csv")

# Start earlier than the 2015 study window so trailing-twelve-month
# calculations and forward-fills have history to work with.
STUDY_START = "2013-01-01"
STUDY_END = "2026-12-31"

# Excluded regardless of what's in the raw file: MSEX's own SEC filings
# didn't tag granular quarterly figures for long stretches, forcing
# >50% of its ROE history to be forward-filled (one run of 25
# consecutive flat quarters). This is a real gap in what MSEX reported,
# not something more code can fix -- excluded from the study universe.
# SJW is listed too as defence-in-depth: the scraper no longer fetches
# it, but this guards against re-running the cleaner against a stale
# raw file that still contains SJW rows.
# WTS (Watts Water Technologies) considered during the 8->28 expansion
# and excluded: never tags a point-in-time common-shares-outstanding
# figure under any standard tag (confirmed via companyfacts audit --
# only weighted-average diluted/basic shares and preferred-stock counts
# exist), so shares_outstanding is uncomputable for all its quarters.
# Same category of gap as SJW's missing capex tag.
# CR (Crane Company) considered and excluded: its current CIK
# (0001944013) only dates to the 2023 Crane Holdings / Crane NXT
# spinoff -- well under both the verifier's MIN_QUARTERS=30 and the
# project's >=10-year trading-history screen. Also listed here as
# defence-in-depth in case a stale raw file still contains its rows.
EXCLUDED_TICKERS = {"MSEX", "SJW", "WTS", "CR"}

# Flow facts cover a period (income statement, cash flow statement) and
# need duration filtering. Everything else is an instant balance-sheet
# snapshot. Cash flow items are reported cumulatively year-to-date by
# most filers, so they rely heavily on the differencing step below.
# NOTE: AssetsCurrent and LiabilitiesCurrent are deliberately ABSENT
# here -- they are instant balance-sheet facts and must not be
# duration-filtered or differenced.
FLOW_TAGS = {
    "NetIncomeLoss",
    "ProfitLoss",  # fallback for NetIncomeLoss -- see NET_INCOME_PRIORITY
    "OperatingIncomeLoss",
    "Revenues",
    "EarningsPerShareDiluted",
    "OperatingCashFlow",
    "CapEx",
}

# Collapsed onto "Revenues" before any other processing, so a ticker
# that switched tags mid-history (ASC 606, ~2018) yields one continuous
# series instead of two half-empty ones.
REVENUE_ALIASES = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenues",
    "SalesRevenueNet": "Revenues",
    "SalesRevenueGoodsNet": "Revenues",
}

# Operating cash flow: these two ARE mutually exclusive naming variants
# for the same concept, so they collapse like the revenue aliases.
OCF_ALIASES = {
    "NetCashProvidedByUsedInOperatingActivities": "OperatingCashFlow",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": "OperatingCashFlow",
}

# CapEx is different and must NOT be collapsed the same way. Some
# filers report one total; others report several co-existing component
# line items in the same filing. Blindly renaming them all to "CapEx"
# would make the dedupe step pick one component arbitrarily and call it
# the total -- silently understating capex for those companies.
# Instead these are resolved by PRIORITY after the pivot: the first
# available tag in this order wins, broadest first. Never summed.
CAPEX_PRIORITY = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    # ROP (Roper Technologies) splits capex across these two tags with
    # only 5 quarters of overlap out of 54 -- alone neither covers its
    # full history, unioned they do. Resolved by priority like every
    # other tag here, never summed.
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

# StockholdersEquity has the same "not a blind alias" problem as CapEx,
# just with one filer instead of many: CWT stopped tagging plain
# StockholdersEquity after 2021-03-31 and switched to the NCI-inclusive
# tag below. The NCI-inclusive figure
# can genuinely exceed the parent-only figure once a noncontrolling
# interest exists, so this resolves by PRIORITY -- plain StockholdersEquity
# wins whenever a filer reports it -- never summed or blindly renamed.
# Without this, CWT's roe/debt_to_equity forward-filled the same 2021-Q1
# value for 21 consecutive quarters (the entire 2023+ test window).
EQUITY_PRIORITY = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]

# NetIncomeLoss has the same "not a blind alias" problem as
# StockholdersEquity: ECL (Ecolab) tags plain NetIncomeLoss (attributable
# to parent) for only 22/54 quarters, but tags ProfitLoss (consolidated,
# including any noncontrolling interest) for all 54. The two can
# genuinely differ once a filer carries a noncontrolling interest, so
# this resolves by PRIORITY -- parent-only NetIncomeLoss wins whenever a
# filer reports it -- never summed or blindly renamed. Keeps roe's
# numerator consistent with the parent-only preference already used for
# its denominator (EQUITY_PRIORITY above).
NET_INCOME_PRIORITY = [
    "NetIncomeLoss",
    "ProfitLoss",
]

# Cover-page facts are dated near the FILING date, not the fiscal
# period end, so they can't be pivoted on exact `end` -- that would
# shatter one quarter into several sparse near-duplicate rows.
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
    # CapEx component tags stay under their own names through the
    # pivot (see CAPEX_PRIORITY), but they're period-flow facts, so
    # they must go through duration filtering and cumulative
    # differencing like any other flow item.
    FLOW_TAGS.update(CAPEX_PRIORITY)
    df = df[~df["ticker"].isin(EXCLUDED_TICKERS)]
    # A ticker may report the same period under two aliased tags in the
    # same filing; keep one.
    df = df.drop_duplicates(subset=["ticker", "tag", "start", "end", "filed"])
    print(
        f"Loaded {len(df)} facts | {df['ticker'].nunique()} tickers: "
        f"{sorted(df['ticker'].unique())}"
    )
    return df


def infer_fiscal_year_starts(df):
    """
    Infer each ticker's fiscal-year start month from its own annual
    filings. Not every company runs a calendar year -- JCI's begins in
    late September -- and assuming Jan-Dec silently skips those tickers
    during quarterly derivation.
    """
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
    """
    Companies often tag only cumulative year-to-date figures (H1, 9mo,
    FY) without a standalone quarter. Recover the discrete quarter by
    subtraction:  Q2 = H1 - Q1,  Q3 = 9mo - H1,  Q4 = FY - 9mo.

    Two correctness details:
      - EPS is excluded: diluted share count changes between quarters,
        so EPS isn't additive and subtracting it would be wrong.
      - `filed` is taken from the LATER cumulative figure, since the
        derived quarter isn't knowable until that filing is public.
        Using the earlier date would reintroduce look-ahead bias.
    """
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
                # Use the reported end date, not a synthetic calendar
                # quarter-end -- 52/53-week fiscal calendars drift.
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
                        "is_derived": True,
                    }
                )

    derived_df = pd.DataFrame(derived)
    print(f"Derived {len(derived_df)} quarterly values from cumulative figures")
    return derived_df


def keep_single_quarter_flows(df):
    """
    Drop cumulative rows. Without this, a 9-month total and a true Q3
    figure both sit under the same `end` date, and the pivot picks one
    arbitrarily -- silently mixing scales across the panel.
    """
    is_flow = df["tag"].isin(FLOW_TAGS)
    dur = (df["end"] - df["start"]).dt.days
    keep = (~is_flow) | dur.between(*QUARTER_DAYS)
    print(f"Dropped {(~keep).sum()} cumulative/annual flow rows")
    return df[keep].copy()


def dedupe_point_in_time(df):
    """
    Every filing restates prior periods as comparatives, so one quarter
    can appear a dozen times across later filings. Keep the EARLIEST --
    that's what the market saw. Keeping the latest would import
    restatements made up to two years after the fact.
    """
    before = len(df)
    df = df.sort_values("filed").drop_duplicates(
        subset=["ticker", "tag", "end"], keep="first"
    )
    print(f"Dropped {before - len(df)} restated/comparative duplicates")
    return df


def pivot_wide(df):
    """One row per ticker-quarter, tags as columns."""
    main = df[~df["tag"].isin(COVERPAGE_TAGS)]
    cover = df[df["tag"].isin(COVERPAGE_TAGS)]

    wide = main.pivot_table(
        index=["ticker", "end"], columns="tag", values="val", aggfunc="first"
    ).reset_index()

    # `filed` = when this row's MODEL FEATURES became public: the last
    # filing among the tags that actually feed roe and debt_to_equity.
    # Taking the max across every tag instead lets a rarely-reported
    # column that only ever appears as a late comparative push the
    # row's availability date out by a year or more, needlessly
    # discarding usable quarters from the backtest.
    #
    # DELIBERATELY NOT EXPANDED to include AssetsCurrent /
    # LiabilitiesCurrent / capex. Every one of those originates in the
    # same 10-Q as the core four, so adding them buys no accuracy --
    # but it reopens exactly the failure mode above, where one
    # sparsely-tagged column drags a whole row's availability date out.
    # That bug cost 1156 days on one MSEX quarter; not worth re-risking.
    # NET_INCOME_PRIORITY included alongside EQUITY_PRIORITY for the same
    # reason: ProfitLoss now genuinely feeds roe for filers like ECL
    # (Ecolab) that mostly tag it instead of plain NetIncomeLoss, so its
    # filing date must count toward this row's availability date too.
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

    # Last-resort fallback: if a ticker-quarter has NO tag with a filed
    # date at all (shouldn't happen given the two fallbacks above, but
    # cheap insurance against a future edge case), approximate with
    # quarter-end + 45 days, the observed median filing lag in this
    # dataset. This is a coarse guess, not a real filing date -- flagged
    # so it can be audited or excluded, never mistaken for the real
    # thing.
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
        # Tolerant join keyed on TRUE quarter-ends (left frame), pulling
        # in the nearest cover-page fact dated at or after it, within 60
        # days. Keying the other way round would replace real
        # period-ends with filing-date artifacts.
        wide = pd.merge_asof(
            wide.sort_values("end"),
            cp.sort_values("end"),
            on="end",
            by="ticker",
            direction="forward",
            tolerance=pd.Timedelta("60D"),
        )

    return wide.sort_values(["ticker", "end"]).reset_index(drop=True)


# Tickers with no usable free_cash_flow, and why. This is a DELIBERATE
# exclusion, not an accidental gap: for SJW, none of the tags in
# CAPEX_PRIORITY exist because SJW simply never tags an aggregate
# capex figure in XBRL (confirmed via a company-facts audit -- see
# diagnose_sjw_capex.py). Its narrow real-estate/water-system-
# acquisition tags are NOT its actual capex and are deliberately not
# used, since blending them in would give SJW a different capex
# definition than every other ticker in the panel -- a silent
# methodology inconsistency, not a genuine data recovery.
# roe / debt_to_equity / shares_outstanding are UNAFFECTED for these
# tickers; only free_cash_flow (and its components) are nulled.
FCF_EXCLUDED_TICKERS = {"SJW"}


# -----------------------------------------------------------------------------
# Calendar-quarter-aware lag helpers
# -----------------------------------------------------------------------------
# Everything below keys on year*4 + quarter rather than row position,
# so a ticker with a missing quarter produces NA instead of a silently
# mis-dated comparison. See the module docstring for why this matters.


def add_quarter_index(df):
    df = df.reset_index(drop=True).copy()
    df["_qidx"] = df["end"].dt.year * 4 + df["end"].dt.quarter
    dupes = df.duplicated(subset=["ticker", "_qidx"]).sum()
    if dupes:
        # 52/53-week fiscal calendars can occasionally land two period
        # ends in one calendar quarter. Report it rather than let the
        # lag merge silently fan out into extra rows.
        print(f"NOTE: {dupes} ticker rows share a calendar quarter (fiscal drift)")
    return df


def lag_quarters(df, col, n):
    """
    Value of `col` exactly n calendar quarters earlier for the same
    ticker. NA when that quarter is absent from the panel.
    """
    lookup = df[["ticker", "_qidx", col]].drop_duplicates(
        subset=["ticker", "_qidx"], keep="first"
    ).copy()
    lookup["_qidx"] = lookup["_qidx"] + n
    lookup = lookup.rename(columns={col: "_lagged"})
    merged = df[["ticker", "_qidx"]].merge(lookup, on=["ticker", "_qidx"], how="left")
    return pd.to_numeric(merged["_lagged"], errors="coerce").values


def ttm_sum(df, col):
    """
    Trailing-twelve-month sum: this quarter plus the previous three.
    Requires all four quarters to be present -- returns NA otherwise,
    rather than quietly reporting a 3-quarter total as if it were a
    year. Only valid for FLOW quantities (income, capex, revenue),
    never for balance-sheet levels.
    """
    total = pd.to_numeric(df[col], errors="coerce").copy()
    for k in (1, 2, 3):
        total = total + pd.Series(lag_quarters(df, col, k), index=df.index)
    return total


def compute_features(df):
    """
    Compute the fundamentals-only feature set.

    Deliberately NOT computed here:
      - operating_margin (revenue coverage too uneven -- see docstring)
      - pe_ratio (REMOVED from the project entirely; a negative PE is
        not "cheap", it is a categorically different state, and the
        feature was replaced by the six added in this revision)
    """
    df = add_quarter_index(df)

    # Share count fallback, priority-ordered. SharesIssued only fills
    # remaining gaps -- it overstates count when treasury stock exists.
    shares = None
    for col in (
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesIssued",
    ):
        if col in df.columns:
            shares = df[col] if shares is None else shares.fillna(df[col])

    # UNIT-SCALE SANITY CHECK (found during the 8 -> 28 expansion,
    # 2026-08-08, root-caused while investigating a bagged-MARS train
    # R^2 of -190): ROP's FY2016 10-K (filed 2017-02-27) tags
    # CommonStockSharesOutstanding for end=2016-12-31 as 101672 -- i.e.
    # in THOUSANDS of shares -- while ROP's own later FY2017 10-K (filed
    # 2018-02-23) restates the identical period as 101672000, in raw
    # shares. Same tag name both times; nothing in the XBRL metadata
    # distinguishes them, and dedupe_point_in_time() correctly keeps the
    # earlier (mis-scaled) filing per the no-look-ahead rule, since that
    # really is what the market saw first. Left unchecked, the
    # 1000x-too-small share count sends book_value_per_share (hence
    # pb_ratio) ~1000x too low and market cap (hence fcf_yield) ~1000x
    # too high for that single quarter -- which alone was enough to blow
    # up bagged-MARS's bagged predictions by orders of magnitude via one
    # bootstrap-resampled training row. This is a generic filer-tagging
    # risk (any ticker/quarter could hit it), not a ROP-specific patch,
    # so the check applies panel-wide: cross-check the priority-picked
    # share count against EntityCommonStockSharesOutstanding (a
    # cover-page fact -- a single scalar per filing, not observed to
    # have this scale problem anywhere in this dataset). Where the two
    # differ by more than 10x, trust the cover-page figure instead; it
    # is confirmed correctly scaled for the SAME filing in the case that
    # motivated this check (ROP's cover page reports 101,434,201 shares
    # as of 2017-02-24, filed 2017-02-27 -- the very filing that tagged
    # CommonStockSharesOutstanding as 101672).
    if "EntityCommonStockSharesOutstanding" in df.columns:
        cover = df["EntityCommonStockSharesOutstanding"]
        ratio = shares / cover
        bad_scale = shares.notna() & cover.notna() & ((ratio > 10) | (ratio < 0.1))
        n_bad = int(bad_scale.sum())
        if n_bad:
            print(
                f"WARNING: {n_bad} shares_outstanding value(s) off by >10x vs "
                f"EntityCommonStockSharesOutstanding (likely a units error in "
                f"the underlying filing) -- using the cover-page figure "
                f"instead. Tickers affected: "
                f"{sorted(df.loc[bad_scale, 'ticker'].unique())}"
            )
            shares = shares.mask(bad_scale, cover)

    df["shares_outstanding"] = shares

    # Resolve StockholdersEquity by priority, same pattern as CapEx: the
    # first available tag in EQUITY_PRIORITY wins (plain StockholdersEquity
    # first), never summed. This backfills CWT's post-2021-Q1 gap with its
    # NCI-inclusive tag without disturbing any filer that still reports
    # plain StockholdersEquity every quarter.
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
        df["StockholdersEquity"] = equity
    df["equity_source_tag"] = equity_source

    # Liabilities is tagged by only a minority of these filers, but the
    # accounting identity Assets = Liabilities + Equity always holds.
    derived_liab = df.get("Assets") - df.get("StockholdersEquity")
    if "Liabilities" in df.columns:
        n = int(df["Liabilities"].isna().sum())
        df["Liabilities"] = df["Liabilities"].fillna(derived_liab)
    else:
        n = len(df)
        df["Liabilities"] = derived_liab
    print(f"Derived {n} Liabilities from Assets - StockholdersEquity")

    # Resolve NetIncomeLoss by priority, same pattern as StockholdersEquity
    # just above: the first available tag in NET_INCOME_PRIORITY wins
    # (parent-only NetIncomeLoss first), never summed. This backfills
    # ECL's (Ecolab) NetIncomeLoss gap with its consolidated ProfitLoss
    # tag without disturbing any filer that already reports NetIncomeLoss
    # every quarter.
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

    # --- Free Cash Flow = Operating Cash Flow - CapEx ---
    # Resolve capex by priority, not by summing (see CAPEX_PRIORITY).
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
        # Explicit exclusion, applied regardless of whether a narrow tag
        # happened to match -- see FCF_EXCLUDED_TICKERS above for why.
        excluded = df["ticker"].isin(FCF_EXCLUDED_TICKERS)
        capex = capex.mask(excluded, pd.NA)
        capex_source = capex_source.mask(excluded, "EXCLUDED_no_true_capex_tag")
        if excluded.any():
            print(
                f"Excluded free_cash_flow for {sorted(df.loc[excluded, 'ticker'].unique())} "
                f"({int(excluded.sum())} rows) -- no genuine aggregate capex tag; "
                f"see FCF_EXCLUDED_TICKERS"
            )

        # SEC reports cash OUTFLOWS as positive numbers under "Payments..."
        # tags, so capex is subtracted as an absolute value. Taking it at
        # face value would ADD capex back for any filer that happened to
        # report it signed negative, inflating FCF.
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

    # =========================================================================
    # NEW FEATURES
    # =========================================================================

    # --- 1. Current Ratio (instant / instant) -------------------------------
    # A pure liquidity measure. No TTM logic needed: both inputs are
    # balance-sheet snapshots at the same date.
    # Guard on > 0 rather than != 0 because a zero or negative
    # LiabilitiesCurrent is a tagging error, not a real balance sheet,
    # and would produce an infinite or sign-flipped ratio.
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

    # --- 2. Asset Growth (level vs level, 4 quarters apart) -----------------
    # Year-over-year, NOT quarter-over-quarter: utility asset bases are
    # seasonal (construction is summer-weighted) and QoQ would mostly
    # measure that seasonality rather than genuine balance-sheet growth.
    assets = pd.to_numeric(df["Assets"], errors="coerce")
    assets_lag4 = pd.Series(lag_quarters(df, "Assets", 4), index=df.index)
    df["asset_growth"] = np.where(assets_lag4 > 0, assets / assets_lag4 - 1, np.nan)

    # --- 3. TTM flows, needed by the next two features -----------------------
    df["ni_ttm"] = ttm_sum(df, "NetIncomeLoss")
    df["capex_ttm"] = ttm_sum(df, "capex") if "capex" in df.columns else np.nan
    # NOTE: no revenue_ttm here. It existed only to feed asset_turnover,
    # which has been removed (see module docstring). Revenues itself is
    # still carried through as a raw diagnostic column, unused.

    # --- 4. Net Income Growth (TTM YoY) -------------------------------------
    # TTM rather than single-quarter for the same seasonality reason as
    # above, and because a single loss-making quarter would otherwise
    # blow up the growth rate.
    # CRITICAL: NA when the base year's TTM income is <= 0. Growth off a
    # negative base is not interpretable -- income going from -10 to -5
    # computes as -50% "growth" when it is actually an improvement, and
    # feeding that sign-flipped value to a linear model teaches it the
    # exact opposite of the truth. Same reasoning that killed pe_ratio.
    ni_ttm_lag4 = pd.Series(lag_quarters(df, "ni_ttm", 4), index=df.index)
    df["ni_growth"] = np.where(
        ni_ttm_lag4 > 0, df["ni_ttm"] / ni_ttm_lag4 - 1, np.nan
    )

    # --- 5. CapEx Intensity (TTM flow / current level) ----------------------
    # Scaled by total assets rather than revenue, deliberately: revenue
    # coverage is the weakest column in this dataset (see docstring),
    # and scaling by it would import that sparsity into a feature that
    # otherwise has near-complete coverage.
    df["capex_intensity"] = np.where(
        assets > 0, pd.to_numeric(df["capex_ttm"], errors="coerce") / assets, np.nan
    )

    return df.drop(columns=["_qidx"])


def forward_fill_within_ticker(df, cols):
    """
    Carry the last known value forward per ticker. This mirrors how an
    analyst actually works -- you use the most recent reported figure
    until a new one lands -- and it never looks forward, so it's safe
    for backtesting. Flags which cells were filled so you can audit or
    exclude them later.

    Note this fills the RATIOS, not their raw inputs. Filling raw
    Assets and then computing asset_growth would compare a real figure
    against a stale copy of itself and report 0% growth, which looks
    like a real observation. Filling the finished ratio at least
    carries forward a value that was genuinely true at some point.
    """
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
    # New features get their own sanity flags. Thresholds are
    # deliberately loose -- these mark "look at this", not "drop this".
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
    df["is_derived"] = False
    if not derived.empty:
        df = pd.concat([df, derived], ignore_index=True)

    df = keep_single_quarter_flows(df)
    df = dedupe_point_in_time(df)

    wide = pivot_wide(df)
    wide = compute_features(wide)

    # Coverage BEFORE forward-fill: this is the honest picture of what
    # was actually reported. The post-fill numbers below look better
    # but are partly carried-forward values.
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

    # The number that actually matters: glmnet and pls drop any row with
    # ANY NA, so this is the real size of the training set these six
    # features leave you with.
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