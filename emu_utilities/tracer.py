from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from .emu_utilities import EMU, find_time_from_file_names
from .resample import llc_compact_to_tiles

if TYPE_CHECKING:
    from numpy import datetime64
    from numpy.typing import NDArray

__all__ = ["load_tracer"]


class EMUTracerGradient(EMU):
    """Handles the loading and processing of EMU tracer data.

    Processes tracer data from EMU tracer output files, converting raw binary
    data into structured arrays and datasets with proper dimensions and coordinates.

    Attributes:
        mean: Whether to use monthly mean or snapshot tracer files.
    """

    def __init__(self, run_directory: str, mean: bool) -> None:
        """Initialize the tracer gradient processor.

        Args:
            run_directory: Path to the EMU run directory.
            mean: If True, use monthly mean files; if False, use snapshot files.

        Raises:
            ValueError: If the EMU tool type is not 'trc'.
        """
        super().__init__(run_directory)
        if self.tool != "trc":
            raise ValueError(f"Expected EMU tool 'trcr', but got '{self.tool}' from directory: {self.run_name}")
        self.mean = mean

    def find_time(self) -> NDArray[datetime64]:
        """Extract timestamps from tracer file names.

        Returns:
            Array of datetime64 objects representing available timestamps.
        """
        pattern = "output/ptracer_mon_mean.*.data" if self.mean else "output/ptracer_mon_snap.*.data"
        return np.array(find_time_from_file_names(self.directory, pattern))

    def load_data(self, trcr_files) -> NDArray[np.float32]:
        """Load tracer data from binary files and apply appropriate scaling.

        Applies vertical and horizontal area weights to correctly represent
        the tracer concentration in each cell.

        Args:
            trcr_files: List of tracer data files to load.

        Returns:
            Structured array of tracer data with applied weights.
        """
        data = np.full(
            (self.time.size, self.nr, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32
        )

        for i, trcr_file in enumerate(trcr_files):
            with open(trcr_file, "rb") as f:
                full_data = np.fromfile(f, dtype=">f4").astype(np.float32)
            data[i] = llc_compact_to_tiles(full_data.reshape((self.nr, self.ny, self.nx)))
            for k in range(self.nr):
                # Apply vertical and horizontal area weights to correctly represent tracer concentration
                data[i, k, :, :, :] = (
                    data[i, k, :, :, :] * self._coordinate_factory.drf[k] * self._coordinate_factory.hfacc[k, :, :, :]
                )

        return data

    def make_dataset(self) -> xr.Dataset:
        """Create an xarray Dataset from tracer data.

        Processes tracer files, sorts by time, applies masks and creates
        a properly structured dataset with metadata.

        Returns:
            Dataset containing tracer data with appropriate coordinates and metadata.
        """
        if self.mean:
            trcr_files = list(self.directory.glob("output/ptracer_mon_mean.*.data"))
        else:
            trcr_files = list(self.directory.glob("output/ptracer_mon_snap.*.data"))
        time_unsorted = self.find_time()
        sort_idx = np.argsort(time_unsorted)
        self.time = time_unsorted[sort_idx]
        trcr_files = [trcr_files[i] for i in sort_idx]

        data = self.load_data(trcr_files)

        coords = self._coordinate_factory.create_tile_coordinates(include_z=True, include_time=True, times=self.time)
        data_vars = {
            "tracer": (["time", "k", "tile", "j", "i"], data),
        }
        ds = self.create_base_dataset(data_vars, coords)

        # Apply ocean mask to exclude land areas
        mask = self._coordinate_factory.create_mask(include_z=True)
        ds = ds.where(mask > 0)

        # Calculate depth-integrated tracer values
        ds["tracer_depth_integrated"] = (ds["tracer"] * mask).sum(dim="k", min_count=1)

        return ds


def load_tracer(run_directory: str, mean: bool = True) -> xr.Dataset:
    """Load tracer data from an EMU run.

    High-level function to load and process tracer data from an EMU run directory.

    Args:
        run_directory: Path to the EMU run directory.
        mean: If True (default), use monthly mean files; if False, use snapshot files.

    Returns:
        Dataset containing processed tracer data.
    """
    emu = EMUTracerGradient(run_directory, mean)
    ds = emu.make_dataset()

    return ds
