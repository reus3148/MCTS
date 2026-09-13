"""Survival after advanced disease in GENIE BPC - the horizon's missing tail.

The simulator's five-year decision horizon has, since v0.2, counted life after
year five for nothing. v1.8 showed what that does once a decision is opened in
follow-up: every late decision is worth less than an early one for no clinical
reason, and the searcher declines salvage 62% of the time in year 1 and 100% in
year 4. Before crediting a terminal value (v0.7), we need to know how much life
the cliff was cutting off. GENIE BPC records overall survival from the first
advanced-disease diagnosis for 894 index cancers; this module reads it.

Kaplan-Meier only. No adjustment, no comparison - a distribution to size a
declared parameter against, not an effect.
"""

from __future__ import annotations

import pandas as pd
from lifelines import KaplanMeierFitter

#: Years at which the survival function is reported.
REPORT_YEARS = (1, 2, 3, 5, 8, 10)


def post_advanced_survival(cancers: pd.DataFrame,
                           years: tuple[int, ...] = REPORT_YEARS) -> dict:
    """Median and S(t) of overall survival from advanced-disease diagnosis.

    Uses ``tt_os_adv_yrs`` / ``os_adv_status``; rows without an advanced
    diagnosis carry NaN there and are excluded, which is the population this
    number is about. Returns plain floats so it can go straight into
    ``metrics.json``.
    """
    advanced = cancers.dropna(subset=["tt_os_adv_yrs", "os_adv_status"])
    if advanced.empty:
        raise ValueError("no index cancer carries an advanced-disease survival time")
    fitter = KaplanMeierFitter().fit(
        advanced["tt_os_adv_yrs"].astype(float),
        advanced["os_adv_status"].astype(int))
    return {
        "n": int(len(advanced)),
        "events": int(advanced["os_adv_status"].sum()),
        "median_years": float(fitter.median_survival_time_),
        "survival": {
            str(year): float(fitter.survival_function_at_times(year).iloc[0])
            for year in years
        },
    }
