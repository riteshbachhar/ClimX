"""Check ClimX NetCDF files for negative, tiny (0 < x < 1) and non-finite values.

Reads only the requested variables, lazily and in the file's own chunks, so checking
the forcings of the full dataset reads a few GB rather than the whole 164 GB.

Usage:
    python check_negatives.py <data_dir>             # forcing variables only
    python check_negatives.py <data_dir> --targets   # also all 7 daily target variables (reads the large arrays)

All target variables are physically non-negative (temperatures are in K, pressure in Pa),
so any negative value in any of them indicates a data problem.
"""
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # avoid netCDF lock deadlocks on shared filesystems

from pathlib import Path

import dask
import numpy as np
import xarray as xr

FORCINGS = ["CO2_LBC", "N2O_LBC", "CH4_LBC", "CFC11eq_LBC", "CF2CL2_LBC",
            "BC_AX", "BC_N", "SO2", "SO4_PR", "OM_NI"]
AEROSOLS = {"BC_AX", "BC_N", "SO2", "SO4_PR", "OM_NI"}
TARGETS = ["tas", "tasmax", "tasmin", "pr", "huss", "psl", "sfcWind"]

# The files store one map per time step (storage chunk 1 x 1 x lat x lon). Reading one map per
# dask task would create ~31,000 tasks per daily variable, so group several steps per chunk instead.
# These sizes are multiples of the storage chunk (aligned) and ~80 MB per chunk at full resolution.
CHUNKS = {"time": 365, "time_month": 120, "time_year": -1}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    data_dir = Path(args[0])
    check_targets = "--targets" in sys.argv

    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    dask.config.set(scheduler="threads", num_workers=n_threads)

    files = sorted(data_dir.glob("*.nc"))
    if not files:
        sys.exit(f"No .nc files found in {data_dir}")
    print(f"Checking {len(files)} files in {data_dir} with {n_threads} thread(s); "
          f"targets included: {check_targets}", flush=True)

    total_negatives = total_nonfinite = 0
    for path in files:
        with xr.open_dataset(path, chunks=CHUNKS) as ds:  # lazy; chunks aligned with storage
            wanted = FORCINGS + (TARGETS if check_targets else [])
            for var in [v for v in wanted if v in ds]:
                da = ds[var]
                # All four reductions are computed in a single pass over the data.
                vmin, vmax, n_neg, n_tiny, n_bad = dask.compute(
                    da.min(), da.max(),
                    (da < 0).sum(),
                    ((da > 0) & (da < 1)).sum(),
                    (~np.isfinite(da)).sum(),
                )
                total_negatives += int(n_neg)
                total_nonfinite += int(n_bad)
                tiny = f"{int(n_tiny)}" if var in AEROSOLS else "n/a"  # only meaningful for aerosol fluxes
                print(f"{path.name:62s} {var:12s} size={da.size:>13,d}  min={float(vmin):11.4g}  "
                      f"max={float(vmax):11.4g}  negatives={int(n_neg)}  tiny(0<x<1)={tiny}  "
                      f"nonfinite={int(n_bad)}", flush=True)

    print(f"\nTOTAL negatives: {total_negatives}   TOTAL non-finite: {total_nonfinite}", flush=True)


if __name__ == "__main__":
    main()
