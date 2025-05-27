import re
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from .emu_utilities import EMU
from .resample import llc_compact_to_tiles


class EMUFowardGradient(EMU):
    def __init__(self, run_directory: str) -> None:
        super().__init__(run_directory)
        if self.tool != "fgrd":
            raise ValueError(f"Expected EMU tool 'fgrd', but got '{self.tool}' from directory: {self.run_name}")

    def make_forward_gradient_dataset(self, run_directory: str, daily: bool) -> xr.Dataset:
        if daily:
            fgrd_2d_files = list(self.directory.glob("output/state_2d_set1_day.*.data"))
            fgrd_3d_files = []
        else:
            fgrd_2d_files = list(self.directory.glob("output/state_2d_set1_mon.*.data"))
            fgrd_3d_files = list(self.directory.glob("output/state_3d_set1_mon.*.data"))
            if not len(fgrd_3d_files) == len(fgrd_2d_files):
                raise ValueError(
                    f"Number of 2D files ({len(fgrd_2d_files)}) does not match number of 3D files ({len(fgrd_3d_files)})"
                )

        hours = np.full(len(fgrd_2d_files), np.nan)
        for i, fgrd_file in enumerate(fgrd_2d_files):
            full_match = re.search(r"\.(\d+)\.data", fgrd_file.name)
            if full_match:
                hours[i] = int(full_match.group(1))
            else:
                raise ValueError(f"Could not extract hours from file name: {fgrd_file.name}")
        sort_idx = np.argsort(hours)
        hours = hours[sort_idx]
        fgrd_2d_files = [fgrd_2d_files[i] for i in sort_idx]
        fgrd_3d_files = [fgrd_3d_files[i] for i in sort_idx] if fgrd_3d_files else []
        fgrd_dt = datetime(1992, 1, 1, 0) + np.array([timedelta(hours=float(hr)) for hr in hours])

        data_2d_size = self.nx * self.ny
        data_3d_size = self.nr * self.ny * self.nx

        ssh_data = np.full((fgrd_dt.size, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32)
        obp_data = np.full((fgrd_dt.size, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32)

        for i, fgrd_file in enumerate(fgrd_2d_files):
            with open(fgrd_file, "rb") as f:
                full_data = np.fromfile(f, dtype=">f4").astype(np.float64)
            ssh_data[i] = llc_compact_to_tiles(full_data[:data_2d_size].reshape((self.ny, self.nx)))
            obp_data[i] = llc_compact_to_tiles(full_data[data_2d_size : 2 * data_2d_size].reshape((self.ny, self.nx)))

        data_vars = {
            "ssh": (["time", "tile", "j", "i"], ssh_data),
            "obp": (["time", "tile", "j", "i"], obp_data),
        }

        if not daily:
            temp_data = np.full(
                (fgrd_dt.size, self.nr, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32
            )
            salt_data = np.full(
                (fgrd_dt.size, self.nr, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32
            )
            uvel_data = np.full(
                (fgrd_dt.size, self.nr, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32
            )
            vvel_data = np.full(
                (fgrd_dt.size, self.nr, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32
            )

            for i, fgrd_file in enumerate(fgrd_3d_files):
                with open(fgrd_file, "rb") as f:
                    full_data = np.fromfile(f, dtype=">f4")
                temp_data[i] = llc_compact_to_tiles(full_data[:data_3d_size].reshape((self.nr, self.ny, self.nx)))
                salt_data[i] = llc_compact_to_tiles(
                    full_data[data_3d_size : 2 * data_3d_size].reshape((self.nr, self.ny, self.nx))
                )
                uvel_data[i] = llc_compact_to_tiles(
                    full_data[data_3d_size * 2 : data_3d_size * 3].reshape((self.nr, self.ny, self.nx))
                )
                vvel_data[i] = llc_compact_to_tiles(
                    full_data[data_3d_size * 3 : data_3d_size * 4].reshape((self.nr, self.ny, self.nx))
                )

            data_vars.update(
                {
                    "temp": (["time", "k", "tile", "j", "i"], temp_data),
                    "salt": (["time", "k", "tile", "j", "i"], salt_data),
                    "uvel": (["time", "k", "tile", "j", "i"], uvel_data),
                    "vvel": (["time", "k", "tile", "j", "i"], vvel_data),
                }
            )

        fgrd_ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": fgrd_dt,
                "tile": np.arange(self.ntiles),
                "k": np.arange(self.nr),
                "j": np.arange(self.ny // self.ntiles),
                "i": np.arange(self.nx),
                "xc": (["tile", "j", "i"], self.xc),
                "yc": (["tile", "j", "i"], self.yc),
                "xg": (["tile", "j", "i"], self.xg),
                "yg": (["tile", "j", "i"], self.yg),
            },
        )

        mask = xr.DataArray(
            data=self.hfacc[0],
            dims=["tile", "j", "i"],
            coords={
                "tile": np.arange(self.ntiles),
                "j": np.arange(self.ny // self.ntiles),
                "i": np.arange(self.nx),
            },
        )

        fgrd_ds = fgrd_ds.where(mask > 0, drop=True)

        fgrd_ds.attrs["created"] = str(datetime.now().isoformat())
        fgrd_ds.attrs["run_name"] = self.run_name
        fgrd_ds.attrs["tool"] = self.tool
        # fgrd_ds.attrs["variable"] = self.variable
        # fgrd_ds[self.variable].attrs["units"] = self.units
        # fgrd_ds[self.variable].attrs["short_name"] = self.short_name

        return fgrd_ds


def load_forward_gradient(run_directory: str, daily: bool = False) -> xr.Dataset:
    emu = EMUFowardGradient(run_directory)
    ds = emu.make_forward_gradient_dataset(run_directory, daily)

    return ds
