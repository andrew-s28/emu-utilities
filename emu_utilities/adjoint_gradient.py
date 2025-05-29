import re
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from numpy._typing._array_like import NDArray

from .emu_utilities import EMU
from .resample import llc_compact_to_tiles


class EMUAdjointGradient(EMU):
    def __init__(self, directory: str) -> None:
        super().__init__(directory)
        if self.tool != "adj":
            raise ValueError(f"Expected EMU tool 'adj', but got '{self.tool}' from directory: {self.run_name}")
        self.controls = ["empmr", "pload", "qnet", "qsw", "saltflux", "spflx", "tauu", "tauv"]
        self.set_controls()
        self.convert_to_tiles()

    def get_adjoint_data(self, variable: str) -> NDArray[np.float32]:
        adj_files = list(self.directory.glob(f"output/adxx_{variable}.*.data"))
        if not adj_files:
            raise FileNotFoundError(
                f"No adjoint data files found for variable '{variable}' in directory: {self.directory}"
            )
        with open(adj_files[0], "rb") as f:
            adj_data = np.fromfile(f, dtype=">f4").astype(np.float32)
        nlags = adj_data.size // (self.nx * self.ny)
        self.nlags = nlags
        if nlags == 0:
            raise ValueError(f"No records found for variable '{variable}' in file: {adj_files[0]}")
        adj_data = adj_data.reshape((nlags, self.ny, self.nx))
        return adj_data

    def set_controls(self):
        self.empmr = self.get_adjoint_data("empmr")
        self.pload = self.get_adjoint_data("pload")
        self.qnet = self.get_adjoint_data("qnet")
        self.qsw = self.get_adjoint_data("qsw")
        self.saltflux = self.get_adjoint_data("saltflux")
        self.spflx = self.get_adjoint_data("spflx")
        self.tauu = self.get_adjoint_data("tauu")
        self.tauv = self.get_adjoint_data("tauv")

    def convert_to_tiles(self):
        self.empmr = llc_compact_to_tiles(self.empmr, less_output=True)
        self.pload = llc_compact_to_tiles(self.pload, less_output=True)
        self.qnet = llc_compact_to_tiles(self.qnet, less_output=True)
        self.qsw = llc_compact_to_tiles(self.qsw, less_output=True)
        self.saltflux = llc_compact_to_tiles(self.saltflux, less_output=True)
        self.spflx = llc_compact_to_tiles(self.spflx, less_output=True)
        self.tauu = llc_compact_to_tiles(self.tauu, less_output=True)
        self.tauv = llc_compact_to_tiles(self.tauv, less_output=True)

    def get_control_metadata(self, variable: str) -> dict:
        metadata = {
            "units": "unknown",
            "short_name": "unknown",
        }
        if variable == "empmr":
            metadata["units"] = "kg/m^2/s"
            metadata["short_name"] = "upward_freshwater_flux"
        elif variable == "pload":
            metadata["units"] = "kg/m^2/s"
            metadata["short_name"] = "downward_surface_pressure_loading"
        elif variable == "qnet":
            metadata["units"] = "W/m^2"
            metadata["short_name"] = "net_upward_heat_flux"
        elif variable == "qsw":
            metadata["units"] = "W/m^2"
            metadata["short_name"] = "net_upward_shortwave_radiation"
        elif variable == "saltflux":
            metadata["units"] = "kg/m^2/s"
            metadata["short_name"] = "net_upward_salt_flux"
        elif variable == "spflx":
            metadata["units"] = "W/m^2"
            metadata["short_name"] = "net_downward_salt_plume_flux"
        elif variable == "tauu":
            metadata["units"] = "N/m^2"
            metadata["short_name"] = "westward_surface_stress"
        elif variable == "tauv":
            metadata["units"] = "N/m^2"
            metadata["short_name"] = "southward_surface_stress"
        return metadata

    def make_adjoint_gradient_dataset(self) -> xr.Dataset:
        data_vars = {var: (["lag", "tile", "j", "i"], getattr(self, var)) for var in self.controls}

        adj_ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "lag": np.arange(self.nlags - 1, -1, -1),
                "tile": np.arange(self.ntiles),
                "j": np.arange(self.ny // self.ntiles),
                "i": np.arange(self.nx),
                "xc": (["tile", "j", "i"], self.xc),
                "yc": (["tile", "j", "i"], self.yc),
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

        adj_ds = adj_ds.where(mask > 0)

        adj_ds.attrs["created"] = str(datetime.now().isoformat())
        adj_ds.attrs["run_name"] = self.run_name
        adj_ds.attrs["tool"] = self.tool
        adj_ds.attrs["variable"] = self.variable
        adj_ds.attrs["short_name"] = self.short_name

        for var in self.controls:
            adj_ds[var].attrs = self.get_control_metadata(var)

        return adj_ds


def load_adjoint_gradient(run_directory: str) -> xr.Dataset:
    emu = EMUAdjointGradient(run_directory)
    adj_ds = emu.make_adjoint_gradient_dataset()
    return adj_ds
