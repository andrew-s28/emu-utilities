from __future__ import annotations, division, print_function

import re
from datetime import datetime, timedelta
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from .resample import llc_compact_to_tiles

if TYPE_CHECKING:
    from typing import Any

    from numpy import datetime64
    from numpy.typing import NDArray

__all__ = []


class EMU:
    """Base class for EMU (Estimating the Circulation and Climate of the Ocean Modeling Utilities) tools.

    Provides common functionality for all EMU tools, including directory management,
    tool identification, and dataset creation.

    Attributes:
        directory: Path to the EMU run directory.
        run_name: Name of the run (directory basename).
        tool: Type of EMU tool identified from the run name.
        nx, ny, nr, ntiles: Grid dimensions.
    """

    def __init__(self, directory: str) -> None:
        """Initialize the EMU base class.

        Args:
            directory: Path to the EMU run directory.
        """
        self.directory = Path(directory)
        self.run_name = self.directory.name
        self.set_tool()
        self._coordinate_factory = CoordinateFactory()
        self.nx = self._coordinate_factory.nx
        self.ny = self._coordinate_factory.ny
        self.nr = self._coordinate_factory.nr
        self.ntiles = self._coordinate_factory.ntiles

    def set_tool(self) -> None:
        """Determine the EMU tool type from the run directory name.

        Sets the tool attribute based on identifiers in the run name.

        Raises:
            ValueError: If the EMU tool type cannot be determined.
        """
        if "samp" in self.run_name:
            self.tool = "samp"  # Sampling
        elif "fgrd" in self.run_name:
            self.tool = "fgrd"  # Forward gradient
        elif "adj" in self.run_name:
            self.tool = "adj"  # Adjoint gradient
        elif "conv" in self.run_name:
            self.tool = "conv"  # Convolution
        elif "trc" in self.run_name:
            self.tool = "trc"  # Tracer
        elif "budg" in self.run_name:
            self.tool = "budg"  # Budget
        elif "msim" in self.run_name:
            self.tool = "msim"  # Model simulation
        elif "atrb" in self.run_name:
            self.tool = "atrb"  # Attribution
        else:
            raise ValueError(f"EMU tool not recognized from directory name: {self.run_name}")

    def validate_tool(self, expected_tool):
        """Ensure the EMU tool matches what's expected.

        Args:
            expected_tool: The expected tool identifier.

        Raises:
            ValueError: If the tool doesn't match the expected value.
        """
        if self.tool != expected_tool:
            raise ValueError(
                f"Expected EMU tool '{expected_tool}', but got '{self.tool}' from directory: {self.run_name}"
            )

    def create_base_dataset(self, data_vars, coords, attrs=None):
        """Create a dataset with standard EMU attributes.

        Args:
            data_vars: Dictionary of data variables to include in the dataset.
            coords: Dictionary of coordinates to include in the dataset.
            attrs: Optional additional attributes to include.

        Returns:
            xarray Dataset with standard EMU metadata.
        """
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Set standard attributes
        ds.attrs["created"] = str(datetime.now().isoformat())
        ds.attrs["run_name"] = self.run_name
        ds.attrs["tool"] = self.tool

        # Add any additional attributes
        if attrs:
            for key, value in attrs.items():
                ds.attrs[key] = value

        return ds

    def get_control_metadata(self, variable: str) -> dict:
        """Get standard metadata for a control variable.

        Args:
            variable: Name of the control variable.

        Returns:
            Dictionary containing standard metadata (units, short_name).
        """
        metadata = {
            "units": "unknown",
            "short_name": "unknown",
        }
        metadata_map = {
            "empmr": {"units": "kg/m^2/s", "short_name": "upward_freshwater_flux"},
            "pload": {"units": "kg/m^2/s", "short_name": "downward_surface_pressure_loading"},
            "qnet": {"units": "W/m^2", "short_name": "net_upward_heat_flux"},
            "qsw": {"units": "W/m^2", "short_name": "net_upward_shortwave_radiation"},
            "saltflux": {"units": "kg/m^2/s", "short_name": "net_upward_salt_flux"},
            "spflx": {"units": "W/m^2", "short_name": "net_downward_salt_plume_flux"},
            "tauu": {"units": "N/m^2", "short_name": "westward_surface_stress"},
            "tauv": {"units": "N/m^2", "short_name": "southward_surface_stress"},
        }
        if variable in metadata_map:
            return metadata_map[variable]
        return metadata


class CoordinateFactory:
    """Factory for creating standard coordinate systems for EMU datasets.

    Manages the loading of grid data and creation of coordinate arrays
    for various grid configurations (cell centers, edges, etc.).

    Attributes:
        nx, ny, nr, ntiles: Grid dimensions.
        xc, yc, rc: Cell center coordinates.
        xg, yg: Cell corner (grid) coordinates.
        hfacc, hfacw, hfacs: Cell fraction factors.
        rac: Cell areas.
        drf: Vertical cell thicknesses.
    """

    def __init__(self) -> None:
        """Initialize the coordinate factory and load grid data."""
        self.set_model_grid()

    def set_model_grid(self) -> None:
        """Load model grid data from embedded resources.

        Sets up all grid-related attributes (dimensions, coordinates, masks).
        """
        self.nx = 90
        self.ny = 1170
        self.nr = 50
        self.ntiles = 13
        self.xc = llc_compact_to_tiles(self.load_grid_data("XC.data", (self.ny, self.nx)))
        self.yc = llc_compact_to_tiles(self.load_grid_data("YC.data", (self.ny, self.nx)))
        self.rc = self.load_grid_data("RC.data", (self.nr,))
        self.xg = llc_compact_to_tiles(self.load_grid_data("XG.data", (self.ny, self.nx)))
        self.yg = llc_compact_to_tiles(self.load_grid_data("YG.data", (self.ny, self.nx)))
        self.hfacc = llc_compact_to_tiles(self.load_grid_data("hFacC.data", (self.nr, self.ny, self.nx)))
        self.hfacw = llc_compact_to_tiles(self.load_grid_data("hFacW.data", (self.nr, self.ny, self.nx)))
        self.hfacs = llc_compact_to_tiles(self.load_grid_data("hFacS.data", (self.nr, self.ny, self.nx)))
        self.rac = llc_compact_to_tiles(self.load_grid_data("RAC.data", (self.ny, self.nx)))
        self.drf = self.load_grid_data("DRF.data", (self.nr,))

    def load_grid_data(self, filename: str, dimensions: tuple):
        """Load grid data from resource files.

        Args:
            filename: Name of the grid data file to load.
            dimensions: Tuple of dimensions to reshape the data.

        Returns:
            Numpy array containing the grid data.
        """
        with files("emu_utilities.grid_data").joinpath(filename).open("rb") as f:
            data = np.fromfile(f, dtype=">f4").reshape(dimensions).astype(np.float32)
        return data

    def create_tile_coordinates(
        self,
        include_z: bool = False,
        include_g: bool = False,
        include_time: bool = False,
        times: list[datetime] | NDArray[datetime64] | None = None,
        include_lag: bool = False,
        lags: list[int] | NDArray[np.int32] | None = None,
    ) -> dict[str, Any]:
        """Create standard tile-based coordinates for datasets.

        Args:
            include_z: Whether to include vertical coordinates.
            include_g: Whether to include grid-point (corner) coordinates.
            include_time: Whether to include time coordinates.
            times: List of time values if include_time is True.
            include_lag: Whether to include lag coordinates.
            lags: List of lag values if include_lag is True.

        Returns:
            Dictionary of coordinate arrays suitable for xarray datasets.
        """
        coords = {
            "tile": np.arange(self.ntiles),
            "j": np.arange(self.ny // self.ntiles),
            "i": np.arange(self.nx),
            "xc": (["tile", "j", "i"], self.xc),
            "yc": (["tile", "j", "i"], self.yc),
        }

        if include_g:
            coords["j_g"] = np.arange(self.ny // self.ntiles)
            coords["i_g"] = np.arange(self.nx)
            coords["xg"] = (["tile", "j_g", "i_g"], self.xg)
            coords["yg"] = (["tile", "j_g", "i"], self.yg)

        if include_z:
            coords["z"] = (["k"], self.rc)
            coords["k"] = np.arange(len(self.rc))

        if include_time and times is not None:
            # prepend times to coords dict for correct ordering in xarray
            times_dict = {"time": times}
            coords = {**times_dict, **coords}

        if include_lag and lags is not None:
            # prepend lags to coords dict for correct ordering in xarray
            lags_dict = {"lag": lags}
            coords = {**lags_dict, **coords}

        return coords

    def create_mask(self, include_z: bool = False, i_g: bool = False, j_g: bool = False) -> xr.DataArray:
        """Create standard ocean mask based on hFacC.

        Creates a mask to separate ocean from land areas, optionally
        for different grid locations.

        Args:
            include_z: Whether to include the vertical dimension.
            i_g: Whether to use i-direction grid points instead of centers.
            j_g: Whether to use j-direction grid points instead of centers.

        Returns:
            DataArray containing the mask values (>0 for ocean).
        """
        if include_z:
            da = xr.DataArray(
                data=self.hfacc,
                dims=["k", "tile", "j", "i"],
                coords={
                    "k": np.arange(self.nr),
                    "tile": np.arange(self.ntiles),
                    "j": np.arange(self.ny // self.ntiles),
                    "i": np.arange(self.nx),
                },
            )
        else:
            da = xr.DataArray(
                data=self.hfacc[0],  # Surface level only
                dims=["tile", "j", "i"],
                coords={
                    "tile": np.arange(self.ntiles),
                    "j": np.arange(self.ny // self.ntiles),
                    "i": np.arange(self.nx),
                },
            )
        da = da.rename({"j": "j_g"}) if j_g else da
        da = da.rename({"i": "i_g"}) if i_g else da
        return da


def find_time_from_file(directory: Path, glob_pattern: str) -> list[datetime]:
    """Extract timestamps stored inside a binary file.

    Args:
        directory: Directory containing files.
        glob_pattern: Pattern to match files (e.g., "output/*.data").

    Returns:
        List of datetime objects corresponding to the timestamps in the files.
    """
    files = list(directory.glob(glob_pattern))
    hours = np.full(len(files), np.nan)

    # only one file expected, read the first one
    with open(files[0], "rb") as f:
        hours = np.fromfile(f, dtype=">i4")

    # convert hours to datetime objects (from 1992-01-01 reference date)
    datetimes = [datetime(1992, 1, 1, 0) + timedelta(hours=float(hr)) for hr in hours]

    return datetimes


def find_time_from_file_names(directory: Path, glob_pattern: str, time_regex: str = r"\.(\d+)\.data") -> list[datetime]:
    """Extract timestamps from file names using a regex pattern.

    Args:
        directory: Directory containing files.
        glob_pattern: Pattern to match files (e.g., "output/*.data").
        time_regex: Regular expression to extract timestamp from file names.

    Returns:
        List of datetime objects corresponding to the timestamps in the file names.
    """
    files = list(directory.glob(glob_pattern))
    hours = np.full(len(files), np.nan)

    # Extract time from file names using regex
    for i, file in enumerate(files):
        match = re.search(time_regex, file.name)
        if match:
            hours[i] = int(match.group(1))
        else:
            raise ValueError(f"Could not extract time from filename: {file.name}")

    # convert hours to datetime objects (from 1992-01-01 reference date)
    datetimes = [datetime(1992, 1, 1, 0) + timedelta(hours=float(hr)) for hr in hours]

    return datetimes
