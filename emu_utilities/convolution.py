import re
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from numpy._typing._array_like import NDArray

from .emu_utilities import EMU
from .resample import llc_compact_to_tiles


class EMUColvolution(EMU):
    def __init__(self, directory: str, dims: int) -> None:
        super().__init__(directory)
        if self.tool != "conv":
            raise ValueError(f"Expected EMU tool 'conv', but got '{self.tool}' from directory: {self.run_name}")
        self.controls = ["empmr", "pload", "qnet", "qsw", "saltflux", "spflx", "tauu", "tauv"]
        self.dims = dims
        self.nweeks = 1357
        self.set_conv_info()
        self.set_controls()
        self.set_time()
        if self.dims == 2:
            self.convert_to_tiles()

    def set_time(self):
        # all time files are the same, just use the first one
        time_file = self.directory / "output/istep_empmr.data"
        with open(time_file, "rb") as f:
            time_data = np.fromfile(f, dtype=">i4").astype(np.float32)
        self.dt = datetime(1992, 1, 1, 0) + np.array([timedelta(hours=float(hr)) for hr in time_data])

    def set_conv_info(self):
        info_file = self.directory / "output/conv.out"
        zero_lag_week = None
        nlags = None
        with open(info_file, "r") as f:
            for i, line in enumerate(f):
                if i == 3:
                    zero_lag_week = int(line.strip())
                elif i == 4:
                    nlags = int(line.strip())
        if zero_lag_week is None or nlags is None:
            raise ValueError(f"Could not find zero_lag_week or nlags in conv.out file: {info_file}")
        self.nlags = nlags + 1
        self.zero_lag_week = zero_lag_week

    def get_1d_conv_data(self, variable: str) -> NDArray[np.float32]:
        conv_files = list(self.directory.glob(f"output/recon1d_{variable}.data"))
        if not conv_files:
            raise FileNotFoundError(
                f"No convolution data files found for variable '{variable}' in directory: {self.directory}"
            )
        with open(conv_files[0], "rb") as f:
            conv_data = np.fromfile(f, dtype=">f4").astype(np.float32)
        if self.nlags == 0:
            raise ValueError(f"No records found for variable '{variable}' in file: {conv_files[0]}")
        conv_data = conv_data.reshape((self.nlags, self.nweeks))
        if variable in ["tauu", "tauv"]:
            # convert to northward and eastward components
            conv_data = -conv_data
        return conv_data

    def get_2d_conv_data(self, variable: str) -> NDArray[np.float32]:
        conv_files = list(self.directory.glob(f"output/recon2d_{variable}.data"))
        if not conv_files:
            raise FileNotFoundError(
                f"No convolution data files found for variable '{variable}' in directory: {self.directory}"
            )
        with open(conv_files[0], "rb") as f:
            conv_data = np.fromfile(f, dtype=">f4").astype(np.float32)
        nlags = conv_data.size // (self.nx * self.ny)
        self.nlags = nlags
        if nlags == 0:
            raise ValueError(f"No records found for variable '{variable}' in file: {conv_files[0]}")
        conv_data = conv_data.reshape((nlags, self.ny, self.nx))
        # if variable in ["tauu", "tauv"]:
        #     # convert to northward and eastward components
        #     conv_data = conv_data
        return conv_data

    def set_controls(self):
        if self.dims == 1:
            self.get_conv_data = self.get_1d_conv_data
        elif self.dims == 2:
            self.get_conv_data = self.get_2d_conv_data
        self.empmr = self.get_conv_data("empmr")
        self.pload = self.get_conv_data("pload")
        self.qnet = self.get_conv_data("qnet")
        self.qsw = self.get_conv_data("qsw")
        self.saltflux = self.get_conv_data("saltflux")
        self.spflx = self.get_conv_data("spflx")
        self.tauu = self.get_conv_data("tauu")
        self.tauv = self.get_conv_data("tauv")
        self.sum = self.empmr + self.pload + self.qnet + self.qsw + self.saltflux + self.spflx + self.tauu + self.tauv

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
            metadata["short_name"] = "eastward_surface_stress"
        elif variable == "tauv":
            metadata["units"] = "N/m^2"
            metadata["short_name"] = "northward_surface_stress"
        return metadata

    def make_2d_conv_gradient_dataset(self) -> xr.Dataset:
        data_vars = {var: (["time", "tile", "j", "i"], getattr(self, var)) for var in self.controls}
        data_vars.update(
            {"sum": (["time", "tile", "j", "i"], self.sum)},
        )

        conv_ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": self.dt,
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

        conv_ds = conv_ds.where(mask > 0)

        conv_ds.attrs["created"] = str(datetime.now().isoformat())
        conv_ds.attrs["run_name"] = self.run_name
        conv_ds.attrs["tool"] = self.tool
        # conv_ds.attrs["variable"] = self.variable
        # conv_ds.attrs["short_name"] = self.short_name

        for var in self.controls:
            conv_ds[var].attrs = self.get_control_metadata(var)

        return conv_ds

    def make_1d_conv_gradient_dataset(self) -> xr.Dataset:
        data_vars = {var: (["lag", "time"], getattr(self, var)) for var in self.controls}
        data_vars.update(
            {"sum": (["lag", "time"], self.sum)},
        )

        conv_ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "lag": np.arange(self.nlags),
                "time": self.dt,
            },
        )

        conv_ds.attrs["created"] = str(datetime.now().isoformat())
        conv_ds.attrs["run_name"] = self.run_name
        conv_ds.attrs["tool"] = self.tool
        # conv_ds.attrs["variable"] = self.variable
        # conv_ds.attrs["short_name"] = self.short_name

        for var in self.controls:
            conv_ds[var].attrs = self.get_control_metadata(var)

        return conv_ds


def load_1d_conv_gradient(run_directory: str) -> xr.Dataset:
    emu = EMUColvolution(run_directory, dims=1)
    conv_ds = emu.make_1d_conv_gradient_dataset()
    return conv_ds


def load_2d_conv_gradient(run_directory: str) -> xr.Dataset:
    emu = EMUColvolution(run_directory, dims=2)
    conv_ds = emu.make_2d_conv_gradient_dataset()
    return conv_ds
