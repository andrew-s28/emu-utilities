from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from .emu_utilities import EMU, CoordinateFactory, find_time_from_file
from .resample import llc_compact_to_tiles

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["load_1d_convolution", "load_2d_convolution", "lagged_variance", "control_variance", "spatial_variance"]

CONTROLS = ["empmr", "pload", "qnet", "qsw", "saltflux", "spflx", "tauu", "tauv"]


class EMUColvolution(EMU):
    """Handles loading and processing of EMU convolution data.

    Processes convolution output, which represents the reconstruction of
    a model metric through a convolution of control variations with
    corresponding gradient fields.

    Attributes:
        controls: List of control variable names.
        dims: Dimensionality of the convolution (1D or 2D).
        nlags: Number of time lags in the data.
        nweeks: Number of weeks in the data.
    """

    def __init__(self, directory: str, dims: int) -> None:
        """Initialize the convolution processor.

        Args:
            directory: Path to the EMU run directory.
            dims: Dimensionality of the convolution (1 or 2).

        Raises:
            ValueError: If the EMU tool type is not 'conv'.
        """
        super().__init__(directory)
        self.validate_tool("conv")
        self.controls = CONTROLS
        self.dims = dims
        self.nweeks = 1357
        self.set_conv_info()
        self.set_controls()
        self.time = self.find_time()

    def find_time(self) -> list[datetime]:
        """Extract timestamps from convolution step files.

        Returns:
            List of datetime objects representing available timestamps.
        """
        return find_time_from_file(self.directory, "output/istep_empmr.data")

    def set_conv_info(self) -> None:
        """Extract convolution metadata from the conv.out file.

        Sets nlags and zero_lag_week attributes based on file contents.

        Raises:
            ValueError: If metadata cannot be extracted.
        """
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

    def set_controls(self) -> None:
        """Load convolution data for all control variables.

        Also computes the sum of all control contributions.
        """
        for control in self.controls:
            setattr(self, control, self.load_data(control))
        # Sum all control contributions to get total response
        self.sum = np.sum([np.array(getattr(self, var)) for var in self.controls], axis=0)

    def load_data(self, variable: str) -> NDArray[np.float32]:
        """Load convolution data based on dimensionality.

        Args:
            variable: Name of the control variable.

        Returns:
            Array of convolution data.

        Raises:
            ValueError: If dimensionality is unsupported.
        """
        if self.dims == 1:
            return self.load_1d_conv_data(variable)
        elif self.dims == 2:
            return self.load_2d_conv_data(variable)
        else:
            raise ValueError(f"Unsupported dimensions: {self.dims}. Only 1D and 2D convolutions are supported.")

    def load_1d_conv_data(self, variable: str) -> NDArray[np.float32]:
        """Load 1D convolution data for a specific variable.

        Args:
            variable: Name of the control variable.

        Returns:
            Array with dimensions [nlags, nweeks].

        Raises:
            FileNotFoundError: If no data files are found.
            ValueError: If no records are found.
        """
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
        return conv_data

    def load_2d_conv_data(self, variable: str) -> NDArray[np.float32]:
        """Load 2D convolution data for a specific variable.

        Args:
            variable: Name of the control variable.

        Returns:
            Array of convolution data in tiled format.

        Raises:
            FileNotFoundError: If no data files are found.
        """
        conv_files = list(self.directory.glob(f"output/recon2d_{variable}.data"))
        if not conv_files:
            raise FileNotFoundError(
                f"No convolution data files found for variable '{variable}' in directory: {self.directory}"
            )
        with open(conv_files[0], "rb") as f:
            conv_data = np.fromfile(f, dtype=">f4").astype(np.float32)
        conv_data = conv_data.reshape((self.nweeks, self.ny, self.nx))
        return llc_compact_to_tiles(conv_data)

    def make_dataset(self) -> xr.Dataset:
        """Create dataset based on convolution dimensionality.

        Returns:
            Dataset with convolution data organized appropriately.

        Raises:
            ValueError: If dimensionality is unsupported.
        """
        if self.dims == 1:
            return self.make_1d_conv_gradient_dataset()
        elif self.dims == 2:
            return self.make_2d_conv_gradient_dataset()
        else:
            raise ValueError(f"Unsupported dimensions: {self.dims}. Only 1D and 2D convolutions are supported.")

    def make_1d_conv_gradient_dataset(self) -> xr.Dataset:
        """Create dataset for 1D convolution data.

        Organizes data by control variable, lag, and time.

        Returns:
            Dataset containing 1D convolution data with appropriate coordinates.
        """
        data_vars = {var: (["lag", "time"], getattr(self, var)) for var in self.controls}
        data_vars.update(
            {"sum": (["lag", "time"], self.sum)},
        )
        coords = {
            "lag": np.arange(self.nlags),
            "time": self.time,
        }

        ds = self.create_base_dataset(data_vars, coords)

        # Reverse the lag dimension for intuitive ordering (recent lags first)
        ds = ds.reindex({"lag": np.arange(self.nlags - 1, -1, -1)})

        # Add standard metadata for each control variable
        for var in self.controls:
            ds[var].attrs = self.get_control_metadata(var)

        return ds

    def make_2d_conv_gradient_dataset(self) -> xr.Dataset:
        """Create dataset for 2D convolution data.

        Organizes data by control variable, time, and spatial coordinates.

        Returns:
            Dataset containing 2D convolution data with appropriate coordinates and masking.
        """
        data_vars = {var: (["time", "tile", "j", "i"], getattr(self, var)) for var in self.controls}
        data_vars.update(
            {"sum": (["time", "tile", "j", "i"], self.sum)},
        )
        coords = self._coordinate_factory.create_tile_coordinates(include_z=False, include_time=True, times=self.time)

        ds = self.create_base_dataset(data_vars, coords)

        # Apply ocean mask to exclude land areas
        mask = self._coordinate_factory.create_mask()
        ds = ds.where(mask > 0)

        # Add standard metadata for each control variable
        for var in self.controls:
            ds[var].attrs = self.get_control_metadata(var)

        return ds


def load_1d_convolution(run_directory: str) -> xr.Dataset:
    """Load 1D convolution data from an EMU run.

    High-level function for loading and processing 1D convolution data.

    Args:
        run_directory: Path to the EMU run directory.

    Returns:
        Dataset containing processed 1D convolution data.
    """
    emu = EMUColvolution(run_directory, dims=1)
    conv_ds = emu.make_dataset()
    return conv_ds


def load_2d_convolution(run_directory: str) -> xr.Dataset:
    """Load 2D convolution data from an EMU run.

    High-level function for loading and processing 2D convolution data.

    Args:
        run_directory: Path to the EMU run directory.

    Returns:
        Dataset containing processed 2D convolution data.
    """
    emu = EMUColvolution(run_directory, dims=2)
    conv_ds = emu.make_dataset()
    return conv_ds


def lagged_variance(conv_ds: xr.Dataset, variable: str) -> xr.DataArray:
    """Calculate explained variance as a function of lag for a control variable.

    Quantifies how much of the total variance is explained by a single
    control variable at different time lags.

    Args:
        conv_ds: Convolution dataset (from load_1d_convolution).
        variable: Name of the control variable.

    Returns:
        DataArray of explained variance values by lag.

    Raises:
        ValueError: If dataset structure is incorrect.
    """
    # Validate inputs
    if (
        variable not in conv_ds.data_vars
        or "sum" not in conv_ds.data_vars
        or "lag" not in conv_ds.dims
        or "time" not in conv_ds.dims
    ):
        raise ValueError(
            f"Incorrect dataset structure for variable '{variable}'. Make sure the dataset follows the output format of the emu_utilities.convolution module."
        )

    # Reindex lags for processing
    conv_ds = conv_ds.reindex({"lag": np.arange(conv_ds["lag"].size)})
    lags = conv_ds["lag"].data

    if lags.size == 0:
        raise ValueError(f"No lags found in dataset for variable '{variable}'.")
    if "lag" not in conv_ds[variable].dims or "time" not in conv_ds[variable].dims:
        raise ValueError(
            f"Incorrect dimensions on {variable}. Make sure the dataset follows the output format of the emu_utilities.convolution module."
        )

    # Calculate explained variance for each lag
    ev_lag_arr = np.full((len(conv_ds["lag"]),), np.nan)
    for i, lag in enumerate(conv_ds["lag"]):
        # Difference between total response and specific control contribution
        diff = conv_ds["sum"].isel(lag=-1) - conv_ds[variable].sel(lag=lag)
        # Explained variance = 1 - (variance of difference / variance of total)
        ev_lag_arr[i] = 1 - _calc_variance(diff) / _calc_variance(conv_ds["sum"].isel(lag=-1))

    # Create DataArray with metadata
    ev_lag = xr.DataArray(
        data=ev_lag_arr,
        dims=["lag"],
        coords={"lag": conv_ds["lag"].data},
        attrs={
            "units": "1",
            "long_name": f"lagged explained variance for {variable}",
            "short_name": f"ev_{variable}",
        },
    )
    return ev_lag


def control_variance(conv_ds: xr.Dataset, lag: int = -1) -> xr.DataArray:
    """Calculate explained variance for each control variable at a specific lag.

    Compares the contribution of each control variable to the total
    variance explained at a given time lag.

    Args:
        conv_ds: Convolution dataset (from load_2d_convolution or load_1d_convolution).
        lag: Time lag to calculate explained variance for. Default is -1 (latest lag).

    Returns:
        DataArray of explained variance values for each control variable.

    Raises:
        ValueError: If dataset structure is incorrect.
    """
    # Ensure all control variables are present in the dataset
    for ctrl in CONTROLS:
        if ctrl not in conv_ds.data_vars:
            raise ValueError(
                f"Control variable '{ctrl}' not found in dataset. Make sure the dataset follows the output format of the emu_utilities.convolution module."
            )
    if "sum" not in conv_ds.data_vars or "lag" not in conv_ds.dims or "time" not in conv_ds.dims:
        raise ValueError(
            "Incorrect dataset structure. Make sure the dataset follows the output format of the emu_utilities.convolution module."
        )
    ev_ctrl_arr = np.full((len(list(conv_ds.keys())) - 1,), np.nan)
    for i, data_var in enumerate(conv_ds.keys()):
        if data_var != "sum":
            if lag == -1:
                diff = conv_ds["sum"].isel(lag=lag) - conv_ds[data_var].isel(lag=lag)
                ev_ctrl_arr[i] = 1 - _calc_variance(diff) / _calc_variance(conv_ds["sum"].isel(lag=lag))
            else:
                diff = conv_ds["sum"].sel(lag=lag) - conv_ds[data_var].sel(lag=lag)
                ev_ctrl_arr[i] = 1 - _calc_variance(diff) / _calc_variance(conv_ds["sum"].sel(lag=lag))

    ev_ctrl = xr.DataArray(
        data=ev_ctrl_arr,
        dims=["control"],
        coords={"control": list(conv_ds.keys())[:-1]},
        attrs={
            "units": "1",
            "long_name": "explained variance for each control variable",
            "short_name": "ev_ctrl",
        },
    )

    return ev_ctrl


def spatial_variance(conv_ds: xr.Dataset, variable: str, perarea: bool = True) -> xr.DataArray:
    """Calculate spatial explained variance for a control variable.

    Assesses how much of the spatial variance in the total response
    can be attributed to variations in a specific control variable.

    Args:
        conv_ds: Convolution dataset (from load_2d_convolution).
        variable: Name of the control variable.
        perarea: If True, normalize explained variance by area (using RAC).

    Returns:
        DataArray of spatial explained variance values.

    Raises:
        ValueError: If dataset structure is incorrect.
    """
    if variable not in conv_ds.data_vars or "sum" not in conv_ds.data_vars or "time" not in conv_ds.dims:
        raise ValueError(
            f"Variable '{variable}' not found in dataset. Make sure the dataset follows the output format of the emu_utilities.convolution module."
        )
    if "tile" not in conv_ds[variable].dims or "j" not in conv_ds[variable].dims or "i" not in conv_ds[variable].dims:
        raise ValueError(
            f"Incorrect dimensions on {variable}. Make sure the dataset follows the output format of the emu_utilities.convolution module."
        )
    if "xc" not in conv_ds.coords or "yc" not in conv_ds.coords:
        raise ValueError(
            "Coordinates 'xc' and 'yc' not found in dataset. Make sure the dataset follows the output format of the emu_utilities.convolution module."
        )
    diff = conv_ds["sum"].sum(dim=["tile", "j", "i"]) - conv_ds[variable]
    spatial_var_arr = 1 - _calc_spatial_variance(diff) / _calc_variance(conv_ds["sum"].sum(dim=["tile", "j", "i"]))
    if perarea:
        rac = CoordinateFactory().rac
        spatial_var_arr_norm = np.divide(spatial_var_arr, rac)
    else:
        spatial_var_arr_norm = spatial_var_arr

    spatial_var = xr.DataArray(
        data=spatial_var_arr_norm,
        dims=["tile", "j", "i"],
        coords={
            "tile": np.arange(conv_ds.sizes["tile"]),
            "j": np.arange(conv_ds.sizes["j"]),
            "i": np.arange(conv_ds.sizes["i"]),
            "xc": (["tile", "j", "i"], conv_ds["xc"].data),
            "yc": (["tile", "j", "i"], conv_ds["yc"].data),
        },
        attrs={
            "long_name": f"spatial explained variance for {variable}",
            "short_name": f"spatial_exp_var_{variable}",
        },
    )
    return spatial_var


def _calc_variance(x: xr.DataArray) -> float:
    if x.size == 0:
        return np.nan
    x = x.data
    mean = np.mean(x)
    variance = float(np.mean((x - mean) ** 2))
    return variance


def _calc_spatial_variance(x: xr.DataArray) -> float:
    if x.size == 0:
        return np.nan
    x = x.data
    mean = np.mean(x, axis=0)
    spatial_variance = np.mean((x - mean) ** 2, axis=0)
    return spatial_variance
