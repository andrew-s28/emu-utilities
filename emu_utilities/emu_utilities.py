from __future__ import division, print_function

from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import xarray as xr
from pandas import date_range

from .resample import llc_compact_to_tiles, resample_to_latlon


class EMU:
    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self.run_name = self.directory.name
        self.set_tool()
        self.set_variable()
        self.set_units()
        self.set_short_name()
        self.set_model_grid()

    def set_tool(self) -> None:
        if "samp" in self.run_name:
            self.tool = "samp"
        elif "fgrd" in self.run_name:
            self.tool = "fgrd"
        elif "adj" in self.run_name:
            self.tool = "adj"
        elif "conv" in self.run_name:
            self.tool = "conv"
        elif "trc" in self.run_name:
            self.tool = "trc"
        elif "budg" in self.run_name:
            self.tool = "budg"
        elif "msim" in self.run_name:
            self.tool = "msim"
        elif "atrb" in self.run_name:
            self.tool = "atrb"
        else:
            raise ValueError(f"EMU tool not recognized from directory name: {self.run_name}")

    def set_variable(self):
        if self.tool == "samp" or self.tool == "atrb":
            if "m_1_" in self.run_name or "d_1_" in self.run_name:
                self.variable = "SSH"
            elif "m_2_" in self.run_name or "d_2_" in self.run_name:
                self.variable = "OBP"
            elif "m_3_" in self.run_name or "d_3_" in self.run_name:
                self.variable = "THETA"
            elif "m_4_" in self.run_name or "d_4_" in self.run_name:
                self.variable = "SALT"
            elif "m_5_" in self.run_name or "d_5_" in self.run_name:
                self.variable = "UV"
            else:
                raise ValueError(f"EMU variable not recognized from directory name: {self.run_name}")
        if self.tool == "adj":
            if self.run_name.endswith("_1"):
                self.variable = "SSH"
            elif self.run_name.endswith("_2"):
                self.variable = "OBP"
            elif self.run_name.endswith("_3"):
                self.variable = "THETA"
            elif self.run_name.endswith("_4"):
                self.variable = "SALT"
            elif self.run_name.endswith("_5"):
                self.variable = "UV"
            else:
                raise ValueError(f"EMU variable not recognized from directory name: {self.run_name}")

    def set_units(self):
        if self.tool == "samp" or self.tool == "adj" or self.tool == "atrb":
            if self.variable == "SSH":
                self.units = "m"
            elif self.variable == "OBP":
                self.units = "m"
            elif self.variable == "THETA":
                self.units = "degree_C"
            elif self.variable == "SALT":
                self.units = "PSU"
            elif self.variable == "UV":
                self.units = "m/s"
            else:
                raise ValueError(f"Units not defined for variable: {self.variable}")

    def set_short_name(self):
        if self.tool == "samp" or self.tool == "adj" or self.tool == "atrb":
            if self.variable == "SSH":
                self.short_name = "sea_surface_height"
            elif self.variable == "OBP":
                self.short_name = "ocean_bottom_pressure"
            elif self.variable == "THETA":
                self.short_name = "potential_temperature"
            elif self.variable == "SALT":
                self.short_name = "practical_salinity"
            elif self.variable == "UV":
                self.short_name = "ocean_velocity"
            else:
                raise ValueError(f"Short name not defined for variable: {self.variable}")

    def set_model_grid(self):
        self.nx = EMU.get_model_grid("nx")
        self.ny = EMU.get_model_grid("ny")
        self.nr = EMU.get_model_grid("nr")
        self.ntiles = EMU.get_model_grid("ntiles")
        self.xc = EMU.get_model_grid("xc")
        self.yc = EMU.get_model_grid("yc")
        self.rc = EMU.get_model_grid("rc")
        self.xg = EMU.get_model_grid("xg")
        self.yg = EMU.get_model_grid("yg")
        self.hfacc = EMU.get_model_grid("hfacc")
        self.hfacw = EMU.get_model_grid("hfacw")
        self.hfacs = EMU.get_model_grid("hfacs")
        self.rac = EMU.get_model_grid("rac")

    @staticmethod
    def get_model_grid(grid_var: str):
        # Model grid variables (Set by rd_grid.py)
        grid_data = {}
        nx = 90
        ny = 1170
        nr = 50
        grid_data["nx"] = nx
        grid_data["ny"] = ny
        grid_data["nr"] = nr
        grid_data["ntiles"] = 13
        grid_data["xc"] = llc_compact_to_tiles(EMU.get_grid_data("XC.data", (ny, nx)))
        grid_data["yc"] = llc_compact_to_tiles(EMU.get_grid_data("YC.data", (ny, nx)))
        grid_data["rc"] = EMU.get_grid_data("RC.data", (nr,))
        grid_data["xg"] = llc_compact_to_tiles(EMU.get_grid_data("XG.data", (ny, nx)))
        grid_data["yg"] = llc_compact_to_tiles(EMU.get_grid_data("YG.data", (ny, nx)))
        grid_data["hfacc"] = llc_compact_to_tiles(EMU.get_grid_data("hFacC.data", (nr, ny, nx)))
        grid_data["hfacw"] = llc_compact_to_tiles(EMU.get_grid_data("hFacW.data", (nr, ny, nx)))
        grid_data["hfacs"] = llc_compact_to_tiles(EMU.get_grid_data("hFacS.data", (nr, ny, nx)))
        grid_data["rac"] = llc_compact_to_tiles(EMU.get_grid_data("RAC.data", (ny, nx)))
        grid_data["drf"] = EMU.get_grid_data("DRF.data", (nr,))
        return grid_data[grid_var]

    @staticmethod
    def get_grid_data(filename: str, dimensions: tuple):
        with files("emu_utilities.grid_data").joinpath(filename).open("rb") as f:
            data = np.fromfile(f, dtype=">f4").reshape(dimensions).astype(np.float64)
        return data


def load_output(pathname: str, grid: xr.Dataset, interp: bool = True) -> xr.Dataset:
    vnames = ["empmr", "pload", "qnet", "qsw", "saltflux", "spflx", "tauu", "tauv"]
    adj_full = []

    for vname in vnames:
        fname = f"{pathname}/output/adxx_{vname}.0000000129.data"
        adjfile = np.fromfile(fname, dtype=">f4")

        nx = 90
        ny = 1170
        nt = adjfile.shape[0] // nx // ny
        adj_tiles = llc_compact_to_tiles(adjfile.reshape(nt, ny, nx))
        if (vname == "tauu") | (vname == "tauv"):
            adj_tiles = -1 * adj_tiles  # switch to positive eastward convention

        # Regrid tile data to lat/lon using nearest neighbor

        new_grid_delta_lat = 1
        new_grid_delta_lon = 1

        new_grid_min_lat = -90
        new_grid_max_lat = 90

        new_grid_min_lon = -180
        new_grid_max_lon = 180

        if interp:
            lons, lats, new_grid_lon_edges, new_grid_lat_edges, adj_rg = resample_to_latlon(
                grid.XC,
                grid.YC,
                adj_tiles,
                new_grid_min_lat,
                new_grid_max_lat,
                new_grid_delta_lat,
                new_grid_min_lon,
                new_grid_max_lon,
                new_grid_delta_lon,
                fill_value=np.nan,
                mapping_method="nearest_neighbor",
                radius_of_influence=120000,
            )
            adj_full.append(
                xr.DataArray(
                    adj_rg,
                    dims=("lag", "lat", "lon"),
                    coords=dict(
                        lag=(["lag"], np.arange(nt - 1, -1, -1)), lon=(["lon"], lons[0]), lat=(["lat"], lats[:, 0])
                    ),
                    name=vname,
                )
            )

        else:
            adj_full.append(
                xr.DataArray(
                    adj_tiles,
                    dims=("lag", "tile", "x", "y"),
                    coords=dict(
                        lag=(["lag"], np.arange(nt - 1, -1, -1)),
                        tile=(["tile"], np.arange(13)),
                        x=(["x"], np.arange(90)),
                        y=(["y"], np.arange(90)),
                    ),
                    name=vname,
                )
            )

    return xr.merge(adj_full)


def get_ecco_forcing(varname, input_dir=None):
    """
    Read in 6-hourly variables from ecco forcing set on LLC90 grid

    Possible varnames are: atmPload, oceFWflx, oceQsw, oceSflux, oceSPflx, oceTAUE, oceTAUN, oceTAUX, oceTAUY, sIceLoad, sIceLoadPatmPload, sIceLoadPatmPload_nopabar, TFLUX

    """
    if input_dir == None:
        input_dir = "/glade/work/noahrose/MITGCM/MITgcm/ECCOV4/ecco.jpl.nasa.gov/drive/files/Version4/Release4/other/flux-forced/forcing"

    arr_ecco = []
    for yr in range(1992, 2018):
        test = np.fromfile(input_dir + f"/{varname}_6hourlyavg_{yr}", ">f4")
        nx = 90
        ny = 1170
        nt = test.shape[0] // nx // ny
        print(nt)
        test_tiles = llc_compact_to_tiles(test.reshape(nt, ny, nx))
        arr_ecco.append(test_tiles)

    arr_ecco = np.concatenate(arr_ecco)

    return arr_ecco


def remove_cycles(data, seasonal=True, diurnal=True):
    if data.ndim != 2:
        data = data.reshape(-1, np.prod(data.shape[1:]))

    dr = date_range(start="01-01-1992t00:00", periods=data.shape[0], freq="6h")

    data_xr = xr.DataArray(data=data, coords=[dr, np.arange(data.shape[1])], dims=["time", "x"])

    if seasonal:
        data_xr = data_xr.groupby("time.month") - data_xr.groupby("time.month").mean("time")
    if diurnal:
        data_xr = data_xr.groupby("time.hour") - data_xr.groupby("time.hour").mean("time")

    return data_xr.to_numpy()


def UXVYfromUEVN(e_fld, n_fld, grid):
    """
    Compute the x and y components of a vector field defined
    by its zonal (eastward) and meridional (northward) components
    with respect to the model grid orientation.

    Parameters
    ----------
    e_fld, n_fld : xarray DataArray
        Zonal (positive east) and meridional (positive north) components
        of a vector field provided at grid cell centers.


    grid : xarray Dataset
        Must contain 'CS' (cosine of grid orientation) and 'SN' (sine of grid orientation).

    Returns
    -------
    x_fld, y_fld : xarray DataArray
        x and y components of the input vector field aligned with the model grid.
    """

    # Check to make sure 'CS' and 'SN' are in coords
    required_fields = ["CS", "SN"]
    for var in required_fields:
        if var not in grid.variables:
            raise KeyError(f"Could not find {var} in coords Dataset")

    # Perform the inverse rotation
    x_fld = e_fld * grid["CS"] + n_fld * grid["SN"]
    y_fld = -e_fld * grid["SN"] + n_fld * grid["CS"]

    x_fld = x_fld.rename({"i": "i_g"})
    y_fld = y_fld.rename({"j": "j_g"})

    return x_fld, y_fld
