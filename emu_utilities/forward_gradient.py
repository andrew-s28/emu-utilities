from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from .emu_utilities import EMU, find_time_from_file_names
from .resample import llc_compact_to_tiles

if TYPE_CHECKING:
    from numpy import datetime64
    from numpy.typing import NDArray

__all__ = ["load_forward_gradient"]


class EMUFowardGradient(EMU):
    """Handles loading and processing of EMU forward gradient data.

    Processes forward gradient output, which represents the model state
    variables (temperature, salinity, velocities, sea surface height, etc.)
    from forward model runs.

    Attributes:
        daily: Whether to use daily or monthly output files.
    """

    def __init__(self, run_directory: str, daily: bool) -> None:
        """Initialize the forward gradient processor.

        Args:
            run_directory: Path to the EMU run directory.
            daily: If True, use daily files; if False, use monthly files.

        Raises:
            ValueError: If the EMU tool type is not 'fgrd'.
        """
        super().__init__(run_directory)
        self.validate_tool("fgrd")
        self.daily = daily
        self.time = self.find_time()
        self.sort_idx = np.argsort(self.time)
        self.time = self.time[self.sort_idx]

    def find_time(self) -> NDArray[datetime64]:
        """Extract timestamps from forward gradient file names.

        Returns:
            Array of datetime64 objects representing available timestamps.
        """
        return np.array(
            find_time_from_file_names(
                self.directory, "output/state_2d_set1_day.*.data" if self.daily else "output/state_2d_set1_mon.*.data"
            )
        )

    def load_data(
        self,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray] | tuple[NDArray, NDArray, None, None, None, None]:
        """Load all forward gradient data based on temporal resolution.

        Returns:
            For daily data: tuple of (ssh_data, obp_data, None, None, None, None)
            For monthly data: tuple of (ssh_data, obp_data, temp_data, salt_data, uvel_data, vvel_data)
        """
        if self.daily:
            fgrd_2d_files = list(self.directory.glob("output/state_2d_set1_day.*.data"))
            data_2d = self.load_2d_data(fgrd_2d_files)
            data_3d = (None, None, None, None)  # No 3D data for daily
            return data_2d + data_3d
        else:
            fgrd_2d_files = list(self.directory.glob("output/state_2d_set1_mon.*.data"))
            fgrd_3d_files = list(self.directory.glob("output/state_3d_set1_mon.*.data"))
            if not len(fgrd_3d_files) == len(fgrd_2d_files):
                raise ValueError(
                    f"Number of 2D files ({len(fgrd_2d_files)}) does not match number of 3D files ({len(fgrd_3d_files)})"
                )
            data_2d = self.load_2d_data(fgrd_2d_files)
            data_3d = self.load_3d_data(fgrd_3d_files)
            return data_2d + data_3d

    def load_2d_data(self, fgrd_2d_files: list[Path]) -> tuple[NDArray, NDArray]:
        """Load 2D forward gradient data (SSH and OBP).

        Args:
            fgrd_2d_files: List of 2D gradient files to load.

        Returns:
            Tuple of (ssh_data, obp_data) arrays.
        """
        fgrd_2d_files = [fgrd_2d_files[i] for i in self.sort_idx]
        data_2d_size = self.nx * self.ny
        ssh_data = np.full((self.time.size, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32)
        obp_data = np.full((self.time.size, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32)

        for i, fgrd_file in enumerate(fgrd_2d_files):
            with open(fgrd_file, "rb") as f:
                full_data = np.fromfile(f, dtype=">f4").astype(np.float32)
            ssh_data[i] = llc_compact_to_tiles(full_data[:data_2d_size].reshape((self.ny, self.nx)))
            obp_data[i] = llc_compact_to_tiles(full_data[data_2d_size : 2 * data_2d_size].reshape((self.ny, self.nx)))

        return ssh_data, obp_data

    def load_3d_data(self, fgrd_3d_files: list[Path]) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Load 3D forward gradient data (temperature, salinity, velocities).

        Args:
            fgrd_3d_files: List of 3D gradient files to load.

        Returns:
            Tuple of (temp_data, salt_data, uvel_data, vvel_data) arrays.
        """
        fgrd_3d_files = [fgrd_3d_files[i] for i in self.sort_idx]
        data_3d_size = self.nr * self.ny * self.nx
        temp_data = np.full(
            (self.time.size, self.nr, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32
        )
        salt_data = np.full(
            (self.time.size, self.nr, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32
        )
        uvel_data = np.full(
            (self.time.size, self.nr, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32
        )
        vvel_data = np.full(
            (self.time.size, self.nr, self.ntiles, self.ny // self.ntiles, self.nx), np.nan, dtype=np.float32
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
        return temp_data, salt_data, uvel_data, vvel_data

    def make_dataset(self) -> xr.Dataset:
        """Create dataset based on temporal resolution.

        Returns:
            Dataset with either daily or monthly forward gradient data.
        """
        if self.daily:
            return self.make_daily_forward_gradient_dataset()
        else:
            return self.make_monthly_forward_gradient_dataset()

    def make_daily_forward_gradient_dataset(self) -> xr.Dataset:
        """Create dataset for daily forward gradient data (2D only).

        Returns:
            Dataset containing SSH and OBP with appropriate coordinates and masking.
        """
        ssh_data, obp_data, _, _, _, _ = self.load_data()
        data_vars = {
            "ssh": (["time", "tile", "j", "i"], ssh_data),
            "obp": (["time", "tile", "j", "i"], obp_data),
        }
        coords = self._coordinate_factory.create_tile_coordinates(include_z=False, include_time=True, times=self.time)
        ds = self.create_base_dataset(data_vars, coords)
        mask = self._coordinate_factory.create_mask(include_z=False)
        ds = ds.where(mask > 0)

        return ds

    def make_monthly_forward_gradient_dataset(self) -> xr.Dataset:
        """Create dataset for monthly forward gradient data (2D and 3D).

        Returns:
            Dataset containing SSH, OBP, temperature, salinity, and velocities
            with appropriate coordinates and masking.
        """
        ssh_data, obp_data, temp_data, salt_data, uvel_data, vvel_data = self.load_data()
        data_vars = {
            "ssh": (["time", "tile", "j", "i"], ssh_data),
            "obp": (["time", "tile", "j", "i"], obp_data),
            "temp": (["time", "k", "tile", "j", "i"], temp_data),
            "salt": (["time", "k", "tile", "j", "i"], salt_data),
            "uvel": (["time", "k", "tile", "j", "i_g"], uvel_data),
            "vvel": (["time", "k", "tile", "j_g", "i"], vvel_data),
        }
        coords = self._coordinate_factory.create_tile_coordinates(
            include_z=True, include_g=True, include_time=True, times=self.time
        )

        ds = self.create_base_dataset(data_vars, coords)

        # Apply appropriate masks for each variable based on their grid location
        mask = self._coordinate_factory.create_mask(include_z=False)
        ds["ssh"] = ds["ssh"].where(mask > 0)
        ds["obp"] = ds["obp"].where(mask > 0)

        mask = self._coordinate_factory.create_mask(include_z=True)
        ds["temp"] = ds["temp"].where(mask > 0)
        ds["salt"] = ds["salt"].where(mask > 0)

        mask = self._coordinate_factory.create_mask(include_z=True, i_g=True)
        ds["uvel"] = ds["uvel"].where(mask > 0)

        mask = self._coordinate_factory.create_mask(include_z=True, j_g=True)
        ds["vvel"] = ds["vvel"].where(mask > 0)

        return ds


def load_forward_gradient(run_directory: str, daily: bool = False) -> xr.Dataset:
    """Load forward gradient data from an EMU run.

    High-level function to load and process forward gradient data.

    Args:
        run_directory: Path to the EMU run directory.
        daily: If True, use daily files; if False (default), use monthly files.

    Returns:
        Dataset containing processed forward gradient data.
    """
    emu = EMUFowardGradient(run_directory, daily)
    ds = emu.make_dataset()

    return ds
