import re
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from .emu_utilities import EMU
from .resample import resample_to_latlon


class EMUSampling(EMU):
    def make_sampling_dataset(self, samp_data, samp_mean, samp_dt) -> xr.Dataset:
        # Sampling variables (Set by rd_samp.py)
        self.samp_data = samp_data
        self.samp_mean = samp_mean
        self.samp_dt = samp_dt

        samp_ds = xr.Dataset(
            data_vars={
                self.variable: (["time"], samp_data),
            },
            coords={
                "time": samp_dt,
            },
        )
        samp_ds.attrs["created"] = str(datetime.now().isoformat())
        samp_ds.attrs["run_name"] = self.run_name
        samp_ds.attrs["tool"] = self.tool
        samp_ds.attrs["variable"] = self.variable
        samp_ds[self.variable].attrs["units"] = self.units
        samp_ds[self.variable].attrs["short_name"] = self.short_name

        return samp_ds


def load_sample(run_directory: str) -> xr.Dataset:
    emu = EMUSampling(run_directory)
    if emu.tool != "samp":
        raise ValueError(f"Expected EMU tool 'samp', but got '{emu.tool}' from directory: {run_directory}")

    samp_files = list(emu.directory.glob("output/samp.out_*"))
    nrec = int(re.findall(r"\d+", samp_files[0].name)[0])

    with open(samp_files[0], "rb") as f:
        samp_data = np.fromfile(f, dtype=">f4", count=nrec).astype(np.float64)
        samp_mean = np.fromfile(f, dtype=">f4", count=1)[0]

    step_files = list(emu.directory.glob("output/samp.step_*"))

    with open(step_files[0], "rb") as f:
        samp_hr = np.fromfile(f, dtype=">i4", count=nrec)

    samp_dt = datetime(1992, 1, 1, 0) + np.array([timedelta(hours=float(hr)) for hr in samp_hr])

    samp_ds = emu.make_sampling_dataset(samp_data, samp_mean, samp_dt)

    return samp_ds
