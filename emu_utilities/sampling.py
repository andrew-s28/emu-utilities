from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from .emu_utilities import EMU, find_time_from_file

if TYPE_CHECKING:
    from numpy import datetime64
    from numpy.typing import NDArray

__all__ = ["load_sample"]


class EMUSampling(EMU):
    """Handles loading and processing of EMU sampling data.

    Manages sampling data from EMU experiments, which represent time series
    data at specific sampling points or regions of interest.
    """

    def __init__(self, run_directory: str) -> None:
        """Initialize the sampling data processor.

        Args:
            run_directory: Path to the EMU run directory.

        Raises:
            ValueError: If the EMU tool type is not 'samp'.
        """
        super().__init__(run_directory)
        self.validate_tool("samp")
        self.time = self.find_time()

    def find_time(self) -> NDArray[datetime64]:
        """Extract timestamps from sampling step files.

        Returns:
            Array of datetime64 objects representing available timestamps.
        """
        return np.array(find_time_from_file(self.directory, "output/samp.step_*"))

    def load_data(self) -> NDArray[np.float32]:
        """Load sampling data from binary files.

        Combines the raw data with the mean value to get the final sampling values.

        Returns:
            Array of sampling data values.

        Raises:
            FileNotFoundError: If no sampling data files are found.
        """
        samp_files = list(self.directory.glob("output/samp.out_*"))
        if not samp_files:
            raise FileNotFoundError(f"No sampling data files found in directory: {self.directory}")

        # Extract number of records from file name
        nrec = int(re.findall(r"\d+", samp_files[0].name)[0])
        with open(samp_files[0], "rb") as f:
            samp_data = np.fromfile(f, dtype=">f4", count=nrec).astype(np.float32)
            # Mean value is appended to the end of the file
            samp_mean = np.fromfile(f, dtype=">f4", count=1)[0]

        # Add mean back to data to get absolute values
        return samp_data + samp_mean

    def make_dataset(self) -> xr.Dataset:
        """Create an xarray Dataset from sampling data.

        Returns:
            Dataset containing sampling data with time coordinates.
        """
        data_vars = {
            "var": (["time"], self.load_data()),
        }
        coords = {
            "time": self.time,
        }

        ds = self.create_base_dataset(data_vars, coords)

        return ds


def load_sample(run_directory: str) -> xr.Dataset:
    """Load sampling data from an EMU run.

    High-level function to load and process sampling data from an EMU run directory.

    Args:
        run_directory: Path to the EMU run directory.

    Returns:
        Dataset containing processed sampling data.
    """
    emu = EMUSampling(run_directory)
    samp_ds = emu.make_dataset()
    return samp_ds
