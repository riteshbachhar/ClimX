"""Hold-out validation: score predicted ClimX index maps against the truth with the official metric.

The same chain is applied to the truth and to the prediction, so every formatting choice
affects both equally:

    daily fields (physical units)
      --compute_indices-->  15 official indices (3 monthly, 12 yearly)     [as Emulator.evaluate]
      --to_annual------->   one yearly Dataset with 15 variables           [as format_indices]
      --to_kaggle_grid-->   192 x 288 grid (lite: nearest-neighbour upsampling, as the playground)
      --to_dataframe---->   Kaggle rows (time, lat, lon) x 15 columns      [as the submission writer]
      --rnNSE.score----->   official score; per-index scores by scoring one column at a time

Note: rnNSE.py only applies its region-wise scoring to complete 192 x 288 grids; a smaller grid
silently falls back to a global score. That is why lite indices are upsampled before scoring.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # avoid netCDF lock deadlocks on shared filesystems

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rnNSE  # noqa: E402  (official Kaggle metric, repo root)
from src import indices_xclim as I  # noqa: E402
from src.utils import kaggle_submission as ks  # noqa: E402
from src.utils.indices_utils import unify_calendar  # noqa: E402
from src.utils.original_data_values import LAT, LON  # noqa: E402

TARGETS_FOR_INDICES = ["tas", "tasmax", "tasmin", "pr"]
MONTHLY_REDUCER = {"TXx": "max", "TNn": "min", "Rx5day": "max"}  # monthly -> annual, as format_indices
ROW_ID = "row_id"


# --------------------------------------------------------------------------------------
# Daily fields -> annual index maps
# --------------------------------------------------------------------------------------
def _strip_scenario(obj):
    """Drop a size-1 forcing_scenario dimension/coordinate (as Emulator.evaluate does)."""
    if "forcing_scenario" in obj.dims:
        obj = obj.isel(forcing_scenario=0, drop=True)
    if "forcing_scenario" in obj.coords:
        obj = obj.drop_vars("forcing_scenario")
    return obj


def compute_indices(daily: xr.Dataset, historical_baseline) -> dict[str, xr.DataArray]:
    """The 15 official indices from daily fields in physical units, exactly as Emulator.evaluate().

    `historical_baseline` is what evaluate() receives as `historical_data_path`; the playground
    passes `dataset.hist`. Use the same object for truth and prediction so the percentile
    thresholds of WSDI, CSDI and R95p are identical.
    """
    daily = _strip_scenario(daily)
    out = {}
    for name, func in I.SELECTED_INDICES.items():
        kwargs = {}
        if "historical_data_path" in inspect.signature(func).parameters:
            kwargs["historical_data_path"] = historical_baseline
        out[name] = func(daily, **kwargs).load()
    return out


def to_annual(indices: dict[str, xr.DataArray]) -> xr.Dataset:
    """One yearly Dataset with all 15 indices.

    Same operations as src.utils.indices_utils.format_indices (unify the calendar, then annual
    max of TXx/Rx5day and annual min of TNn), without its side effects (it writes files).
    """
    annual = {}
    for name, da in indices.items():
        da = _strip_scenario(unify_calendar(da))
        if name in MONTHLY_REDUCER:
            resampled = da.resample(time="YS")
            da = resampled.max() if MONTHLY_REDUCER[name] == "max" else resampled.min()
        annual[name] = da
    return xr.Dataset(annual)


def to_kaggle_grid(ds: xr.Dataset) -> xr.Dataset:
    """Put annual index maps on the 192 x 288 Kaggle grid.

    Full-resolution input is only sorted. Lite input is upsampled with nearest neighbour and
    edge extrapolation, exactly as the playground's submission cell does.
    """
    lat = np.asarray(LAT, dtype=float)
    lon = np.asarray(LON, dtype=float)
    ds = ds.sortby("lat").sortby("lon")
    if ds.sizes["lat"] == lat.size and ds.sizes["lon"] == lon.size:
        return ds
    return ds.interp(lat=lat, lon=lon, method="nearest", kwargs={"fill_value": "extrapolate"})


def to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Kaggle table: row_id + 15 index columns, rows ordered (time, lat, lon)."""
    return ks.indices_xr_to_kaggle_dataframe(ds, index_columns=ks.CLIMX_INDEX_COLUMNS, sort_coords=True)


# --------------------------------------------------------------------------------------
# Truth (computed once, cached)
# --------------------------------------------------------------------------------------
def true_annual_indices(dataset, scenario: str, cache_path: str | Path | None = None) -> xr.Dataset:
    """Annual true indices for one training scenario (e.g. the held-out 'ssp370').

    `dataset` is the ClimXVirtualDatasets from open_climx_virtual_datasets. The result is
    cached to `cache_path` (netCDF) because it never changes.
    """
    if cache_path is not None and Path(cache_path).exists():
        return xr.open_dataset(cache_path).load()
    daily = dataset.train.sel(forcing_scenario=[scenario])[TARGETS_FOR_INDICES]
    truth = to_annual(compute_indices(daily, dataset.hist))
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        truth.to_netcdf(cache_path)
    return truth


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def score_annual(pred: xr.Dataset, truth: xr.Dataset) -> tuple[float, pd.Series]:
    """Official score of annual predicted index maps against the truth.

    Returns (overall score, per-index scores). Both are region-wise nNSE in (-1, 1]:
    1 = perfect, 0 = no better than each cell's own time mean.
    """
    pred_g, truth_g = to_kaggle_grid(pred), to_kaggle_grid(truth)
    n_truth_years = truth_g.sizes["time"]
    pred_g, truth_g = xr.align(pred_g, truth_g, join="inner")  # common years only
    if truth_g.sizes["time"] == 0:
        raise ValueError("Prediction and truth share no years.")
    if truth_g.sizes["time"] != n_truth_years:
        warnings.warn(f"Scoring {truth_g.sizes['time']} of {n_truth_years} truth years (only those predicted).")

    sol, sub = to_dataframe(truth_g), to_dataframe(pred_g)
    cols = [c for c in sol.columns if c != ROW_ID]

    if sol[cols].isnull().any().any():
        raise ValueError(f"Truth contains NaN in {sol[cols].columns[sol[cols].isnull().any()].tolist()}")
    nan_cols = sub[cols].columns[sub[cols].isnull().any()].tolist()
    if nan_cols:  # the Kaggle submission writer does the same: NaN -> 0 with a warning
        warnings.warn(f"Prediction has NaN in {nan_cols}; replacing with 0 as the submission writer would.")
        sub[nan_cols] = sub[nan_cols].fillna(0.0)

    overall = rnNSE.score(sol.copy(), sub.copy(), ROW_ID)
    per_index = pd.Series(
        {c: rnNSE.score(sol[[ROW_ID, c]].copy(), sub[[ROW_ID, c]].copy(), ROW_ID) for c in cols},
        name="score",
    )
    return float(overall), per_index


def score_predictions(indices_pred: dict[str, xr.DataArray], truth: xr.Dataset) -> tuple[float, pd.Series]:
    """Score the `indices_pred` dict returned by Emulator.evaluate()."""
    return score_annual(to_annual(indices_pred), truth)


# --------------------------------------------------------------------------------------
# Sanity checks of the scorer itself
# --------------------------------------------------------------------------------------
def run_sanity_checks(truth: xr.Dataset) -> pd.DataFrame:
    """Score synthetic predictions whose result is known exactly.

    - truth vs itself                       -> 1.0
    - truth vs each cell's time mean        -> 0.0  (definition of the metric's zero)
    - truth + 0.5 x each cell's time std    -> 0.6  (R^2 = 1 - 0.5^2 = 0.75, nNSE = 0.75/1.25)
    - overall score equals the mean of the per-index scores
    """
    std = truth.std("time")  # ddof=0, as the metric
    cases = {
        "truth vs itself": (truth, 1.0),
        "truth vs its time mean": (truth.mean("time").broadcast_like(truth), 0.0),
        "truth + 0.5 std": (truth + 0.5 * std, 0.6),
    }
    rows = []
    for label, (pred, expected) in cases.items():
        overall, per_index = score_annual(pred, truth)
        rows.append({
            "check": label,
            "expected": expected,
            "overall": overall,
            "per-index min": per_index.min(),
            "per-index max": per_index.max(),
            "overall == mean(per-index)": bool(np.isclose(overall, per_index.mean())),
            "pass": bool(np.isclose(overall, expected, atol=1e-6) and np.allclose(per_index, expected, atol=1e-6)),
        })
    return pd.DataFrame(rows).set_index("check")


# --------------------------------------------------------------------------------------
# Command line: build the cached truth and run the sanity checks
# --------------------------------------------------------------------------------------
def main():
    from src.utils.hugging_face_utils import open_climx_virtual_datasets

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-root", required=True, help="folder containing lite/ and full/")
    p.add_argument("--variant", default="lite", choices=["lite", "full"])
    p.add_argument("--scenario", default="ssp370", help="held-out training scenario")
    p.add_argument("--cache", required=True, help="netCDF path for the cached true indices")
    args = p.parse_args()

    dataset = open_climx_virtual_datasets(args.data_root, args.variant)
    truth = true_annual_indices(dataset, args.scenario, args.cache)
    print(f"True indices for {args.scenario} ({args.variant}): {dict(truth.sizes)} -> {args.cache}", flush=True)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(run_sanity_checks(truth).round(6).to_string())


if __name__ == "__main__":
    main()
