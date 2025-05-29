import re
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from .emu_utilities import EMU
from .resample import llc_compact_to_tiles


class EMUTracerGradient(EMU):
    def __init__(self, run_directory: str) -> None:
        super().__init__(run_directory)
        if self.tool != "trc":
            raise ValueError(f"Expected EMU tool 'trcr', but got '{self.tool}' from directory: {self.run_name}")

    def make_tracer_gradient_dataset(self, run_directory: str, mean: bool = True) -> xr.Dataset:
        if mean:
            trcr_files = list(self.directory.glob("output/ptracer_mon_mean.*.data"))
        else:
            trcr_files = list(self.directory.glob("output/ptracer_mon_snap.*.data"))

        hours = np.full(len(trcr_files), np.nan)
        for i, trcr_file in enumerate(trcr_files):
            full_match = re.search(r"\.(\d+)\.data", trcr_file.name)
            if full_match:
                hours[i] = int(full_match.group(1))
            else:
                raise ValueError(f"Could not extract hours from file name: {trcr_file.name}")
        sort_idx = np.argsort(hours)
        hours = hours[sort_idx]
        trcr_files = [trcr_files[i] for i in sort_idx]
        trcr_dt = datetime(1992, 1, 1, 0) + np.array([timedelta(hours=float(hr)) for hr in hours])

        trcr_data = np.full(
            (trcr_dt.size, self.nr, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32
        )

        hfacc = EMU.get_model_grid("hfacc")
        drf = EMU.get_model_grid("drf")
        for i, trcr_file in enumerate(trcr_files):
            with open(trcr_file, "rb") as f:
                full_data = np.fromfile(f, dtype=">f4").astype(np.float32)
            trcr_data[i] = llc_compact_to_tiles(full_data.reshape((self.nr, self.ny, self.nx)))
            # print(trcr_data[i].shape, hfacc.shape, drf.shape)
            for k in range(self.nr):
                trcr_data[i, k, :, :, :] = (
                    trcr_data[i, k, :, :, :] * drf[k] * hfacc[k, :, :, :]
                )  # Apply vertical and horizontal area weights

        trcr_ds = xr.Dataset(
            data_vars={
                "tracer": (["time", "k", "tile", "j", "i"], trcr_data),
            },
            coords={
                "time": trcr_dt,
                "k": np.arange(self.nr),
                "tile": np.arange(self.ntiles),
                "j": np.arange(self.ny // self.ntiles),
                "i": np.arange(self.nx),
                "xc": (["tile", "j", "i"], self.xc),
                "yc": (["tile", "j", "i"], self.yc),
                "xg": (["tile", "j", "i"], self.xg),
                "yg": (["tile", "j", "i"], self.yg),
                "z": (["k"], self.rc),
            },
        )

        mask = xr.DataArray(
            data=self.hfacc,
            dims=["k", "tile", "j", "i"],
            coords={
                "k": np.arange(self.nr),
                "tile": np.arange(self.ntiles),
                "j": np.arange(self.ny // self.ntiles),
                "i": np.arange(self.nx),
            },
        )

        trcr_ds = trcr_ds.where(mask > 0)

        trcr_ds["tracer_depth_integrated"] = (trcr_ds["tracer"] * mask).sum(dim="k", min_count=1)

        trcr_ds.attrs["created"] = str(datetime.now().isoformat())
        trcr_ds.attrs["run_name"] = self.run_name
        trcr_ds.attrs["tool"] = self.tool

        return trcr_ds


def load_tracer_gradient(run_directory: str, mean: bool = True) -> xr.Dataset:
    emu = EMUTracerGradient(run_directory)
    ds = emu.make_tracer_gradient_dataset(run_directory, mean)

    return ds
