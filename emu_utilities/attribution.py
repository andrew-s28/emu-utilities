from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from .emu_utilities import EMU, find_time_from_file

if TYPE_CHECKING:
    from numpy import datetime64
    from numpy.typing import NDArray

__all__ = ["load_attribution"]


class EMUAttribution(EMU):
    """Handles loading and processing of EMU attribution data.

    Processes attribution output, which quantifies the contribution of
    different forcing factors to the overall ocean state.

    Attributes:
        anomaly_variables: List of attribution factor names.
    """

    def __init__(self, run_directory: str) -> None:
        """Initialize the attribution processor.

        Args:
            run_directory: Path to the EMU run directory.

        Raises:
            ValueError: If the EMU tool type is not 'atrb'.
        """
        super().__init__(run_directory)
        self.validate_tool("atrb")
        self.anomaly_variables = [
            "reference_run",
            "wind_stress",
            "heat_flux",
            "freshwater_flux",
            "salt_flux",
            "pressure_load",
            "initial_conditions",
        ]
        self.time = self.find_time()

    def find_time(self) -> NDArray[datetime64]:
        """Extract timestamps from attribution step files.

        Returns:
            Array of datetime64 objects representing available timestamps.
        """
        return np.array(find_time_from_file(self.directory, "output/atrb.step_*"))

    def load_data(self) -> NDArray[np.float32]:
        """Load attribution data from binary files.

        The file contains time series for each attribution factor, with means
        stored at the end of the file.

        Returns:
            Array of attribution data (without means).
        """
        atrb_files = list(self.directory.glob("output/atrb.out_*"))
        with open(atrb_files[0], "rb") as f:
            atrb_data = np.fromfile(f, dtype=">f4").astype(np.float32)
        # time means are stored at end of the file
        # means = atrb_data[-len(self.anomaly_variables) :]
        # return only the data without the means for now
        data = atrb_data[: -len(self.anomaly_variables)]
        return data

    def make_attribution_dataset(self) -> xr.Dataset:
        """Create an xarray Dataset from attribution data.

        Organizes the data by attribution factor and time.

        Returns:
            Dataset containing attribution data organized by factor and time.
        """
        data = self.load_data()
        data = data.reshape((len(self.anomaly_variables), self.time.size))

        data_vars = {var: (["time"], data) for var, data in zip(self.anomaly_variables, data)}
        coords = {"time": self.time}
        ds = self.create_base_dataset(data_vars, coords)

        return ds


def load_attribution(run_directory: str) -> xr.Dataset:
    """Load attribution data from an EMU run.

    High-level function to load and process attribution data.

    Args:
        run_directory: Path to the EMU run directory.

    Returns:
        Dataset containing processed attribution data.
    """
    emu = EMUAttribution(run_directory)
    atrb_ds = emu.make_attribution_dataset()

    return atrb_ds
