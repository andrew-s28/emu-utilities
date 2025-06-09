from __future__ import annotations

import numpy as np
import xarray as xr
from numpy._typing._array_like import NDArray

from .emu_utilities import EMU
from .resample import llc_compact_to_tiles

__all__ = ["load_adjoint_gradient"]


class EMUAdjointGradient(EMU):
    """Handles loading and processing of EMU adjoint gradient data.

    Processes adjoint gradient output, which represents sensitivities of
    model metrics to control variables at different time lags.

    Attributes:
        controls: List of control variable names.
        nlags: Number of time lags in the data.
    """

    def __init__(self, directory: str) -> None:
        """Initialize the adjoint gradient processor.

        Args:
            directory: Path to the EMU run directory.

        Raises:
            ValueError: If the EMU tool type is not 'adj'.
        """
        super().__init__(directory)
        self.validate_tool("adj")
        self.controls = ["empmr", "pload", "qnet", "qsw", "saltflux", "spflx", "tauu", "tauv"]
        self.set_controls()

    def load_data(self, variable: str) -> NDArray[np.float32]:
        """Load adjoint gradient data for a specific control variable.

        Args:
            variable: Name of the control variable.

        Returns:
            Array of gradient data with dimensions [nlags, ny, nx].

        Raises:
            FileNotFoundError: If no adjoint data files are found.
            ValueError: If no records are found in the file.
        """
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
        """Load adjoint gradient data for all control variables.

        Converts from compact format to tiles and sets attributes for each control.
        """
        for control in self.controls:
            setattr(self, control, llc_compact_to_tiles(self.load_data(control)))

    def make_adjoint_gradient_dataset(self) -> xr.Dataset:
        """Create an xarray Dataset from adjoint gradient data.

        Organizes data by control variable, lag, and spatial coordinates.

        Returns:
            Dataset containing adjoint gradient data with appropriate coordinates and masking.
        """
        data_vars = {var: (["lag", "tile", "j", "i"], getattr(self, var)) for var in self.controls}
        coords = self._coordinate_factory.create_tile_coordinates(
            include_z=False, include_lag=True, lags=np.arange(self.nlags - 1, -1, -1)
        )
        ds = self.create_base_dataset(data_vars, coords)

        # Apply ocean mask to exclude land areas
        mask = self._coordinate_factory.create_mask(include_z=False)
        ds = ds.where(mask > 0)

        # Add standard metadata for each control variable
        for var in self.controls:
            ds[var].attrs = self.get_control_metadata(var)

        return ds


def load_adjoint_gradient(run_directory: str) -> xr.Dataset:
    """Load adjoint gradient data from an EMU run.

    High-level function to load and process adjoint gradient data.

    Args:
        run_directory: Path to the EMU run directory.

    Returns:
        Dataset containing processed adjoint gradient data.
    """
    emu = EMUAdjointGradient(run_directory)
    adj_ds = emu.make_adjoint_gradient_dataset()
    return adj_ds
