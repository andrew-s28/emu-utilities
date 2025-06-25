from __future__ import annotations

import struct
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import TYPE_CHECKING
from .emu_utilities import EMU
import re
if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["load_budget"]


class EMUBudget(EMU):
    """Handles loading and processing of EMU budget data."""

    def __init__(self, run_directory: str) -> None:
        super().__init__(run_directory)
        self.validate_tool("budg")
        self.budget_file = self.find_budget_file()
        self.time = self._build_time_axis()

    def find_budget_file(self) -> Path:
        files = sorted(self.directory.glob("output/emu_budg.sum_*"))
        if not files:
            raise FileNotFoundError("No emu_budg.sum_* file found.")
        return files[0]

    def _build_time_axis(self) -> NDArray[datetime]:
        """Construct time axis as datetime array (monthly, starts at Jan 1992)."""
        with open(self.budget_file, "rb") as f:
            f.seek(4)  # Skip ibud
            nmonths = struct.unpack(">l", f.read(4))[0]

        base_date = datetime(1992, 1, 15)
        return np.array([base_date + relativedelta(months=i) for i in range(nmonths)], dtype="O")

    def _parse_budget_info(self, info_path: Path) -> dict:
        """Parse the budg.info file and return metadata with separate variable name and units."""
        meta = {}
    
        if not info_path.exists():
            print(f"Warning: budg.info not found at {info_path}")
            return meta
    
        with open(info_path, "r") as f:
            for line in f:
                line = line.strip()
    
                if "budget sampled from" in line.lower():
                    meta["source_path"] = line.split("from")[-1].strip()
    
                elif "budget is monthly" in line.lower():
                    meta["temporal_resolution"] = "monthly"
    
                elif re.search(r"budget variable\s*:", line, re.IGNORECASE):
                    match = re.search(r"budget variable\s*:\s*(.+)", line, re.IGNORECASE)
                    if match:
                        full = match.group(1).strip()
                        # Split into variable name and units
                        var_match = re.match(r"(.+?)\s*\((.+)\)", full)
                        if var_match:
                            meta["budget_name"] = var_match.group(1).strip()
                            meta["budget_units"] = var_match.group(2).strip() + " / s"
                        else:
                            meta["budget_name"] = full
                            meta["budget_units"] = "unknown"
    
                elif "pert_i" in line and "pert_j" in line and "pert_k" in line:
                    match = re.search(r"pert_i,\s*pert_j,\s*pert_k\s*=\s*(\d+)\s+(\d+)\s+(\d+)", line)
                    if match:
                        meta["i"] = int(match.group(1))
                        meta["j"] = int(match.group(2))
                        meta["k"] = int(match.group(3))
    
                elif "budget model grid location" in line.lower():
                    meta["location_str"] = line.split("-->")[-1].strip()
    
                elif "iobjf =" in line:
                    match = re.search(r"iobjf\s*=\s*(\d+)", line)
                    if match:
                        meta["iobjf (multiplier)"] = int(match.group(1))
    
                elif "ifunc =" in line:
                    match = re.search(r"ifunc\s*=\s*(\d+)", line)
                    if match:
                        meta["ifunc"] = int(match.group(1))
    
        return meta

    def load_budget_data(self) -> tuple[list[str], NDArray[np.float32]]:
        """Read EMU budget binary file and return names and 2D array [nvar, ntime]."""
        with open(self.budget_file, "rb") as f:
            ibud = struct.unpack(">l", f.read(4))[0]
            if not (1 <= ibud <= 5):
                raise ValueError(f"Invalid budget ID: {ibud}")
            ibud -= 1

            nmonths = struct.unpack(">l", f.read(4))[0]
            emubudg_name = []
            emubudg = np.zeros((0, nmonths), dtype=np.float32)

            while True:
                fvar_bytes = f.read(12)
                if len(fvar_bytes) < 12:
                    break
                fvar = fvar_bytes.decode("utf-8").strip()
                emubudg_name.append(fvar)

                fdum = np.array(struct.unpack(f">{nmonths}f", f.read(4 * nmonths)), dtype=np.float32)
                emubudg = np.vstack((emubudg, fdum))

        return emubudg_name, emubudg, ibud

    def make_dataset(self) -> xr.Dataset:
        names, emubudg, ibud = self.load_budget_data()
        nmonths = len(self.time)
        nvar = len(names)

        lhs = emubudg[1, :]
        rhs = emubudg[2, :].copy()
        for i in range(3, nvar):
            rhs += emubudg[i, :]

        # adv
        adv = np.zeros(nmonths, dtype=np.float32)
        for i in range(nvar):
            if "adv" in names[i]:
                adv += emubudg[i, :]

        # mix
        mix = np.zeros(nmonths, dtype=np.float32)
        for i in range(nvar):
            if "mix" in names[i]:
                mix += emubudg[i, :]

        # frc
        frc = np.zeros(nmonths, dtype=np.float32)
        for i in range(nvar):
            if names[i] not in ["dt", "lhs"]:
                if all(k not in names[i] for k in ["adv", "mix", "lhs", "dt"]):
                    frc += emubudg[i, :]

        # Build dataset
        data_vars = {
            "lhs": (["time"], lhs),
            "rhs": (["time"], rhs),
            "advection": (["time"], adv),
            "mixing": (["time"], mix),
            "surface_forcing": (["time"], frc),
        }

        coords = {"time": self.time}
        ds = self.create_base_dataset(data_vars, coords)

        info_files = list(self.directory.glob("output/budg.info"))

        if info_files:
            meta = self._parse_budget_info(info_files[0])
            ds.attrs.update(meta)
        else:
            print("Warning: No budg.info file found.")

        return ds


def load_budget(run_directory: str) -> xr.Dataset:
    """Load EMU budget data as an xarray Dataset."""
    emu = EMUBudget(run_directory)
    return emu.make_dataset()
