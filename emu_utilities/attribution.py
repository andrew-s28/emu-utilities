import re
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from .emu_utilities import EMU
from .resample import resample_to_latlon


class EMUAttribution(EMU):
    def __init__(self, run_directory: str) -> None:
        super().__init__(run_directory)
        if self.tool != "atrb":
            raise ValueError(f"Expected EMU tool 'atrb', but got '{self.tool}' from directory: {self.run_name}")
        self.anomaly_variables = [
            "reference_run",
            "wind_stress",
            "heat_flux",
            "freshwater_flux",
            "salt_flux",
            "pressure_load",
            "initial_conditions",
        ]

    def make_attribution_dataset(self) -> xr.Dataset:
        atrb_files = list(self.directory.glob(pattern="output/atrb.out_*"))

        atrb_step_files = list(self.directory.glob("output/atrb.step_*"))
        with open(atrb_step_files[0], "rb") as f:
            atrb_hr = np.fromfile(f, dtype=">i4")
        atrb_dt = datetime(1992, 1, 1, 0) + np.array([timedelta(hours=float(hr)) for hr in atrb_hr])

        with open(atrb_files[0], "rb") as f:
            data = np.fromfile(f, dtype=">f4").astype(np.float32)
            means = data[-len(self.anomaly_variables) :]
            data = data[: -len(self.anomaly_variables)]
        atrb_data = data.reshape((len(self.anomaly_variables), atrb_dt.size))

        data_vars = {var: (["time"], data) for var, data in zip(self.anomaly_variables, atrb_data)}
        samp_ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": atrb_dt,
            },
        )
        samp_ds.attrs["created"] = str(datetime.now().isoformat())
        samp_ds.attrs["run_name"] = self.run_name
        samp_ds.attrs["tool"] = self.tool
        samp_ds.attrs["variable"] = self.variable
        # samp_ds[self.variable].attrs["units"] = self.units
        # samp_ds[self.variable].attrs["short_name"] = self.short_name

        return samp_ds


def load_attribution(run_directory: str) -> xr.Dataset:
    emu = EMUAttribution(run_directory)
    atrb_ds = emu.make_attribution_dataset()

    return atrb_ds
