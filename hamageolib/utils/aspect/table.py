"""
Utilities for reading ASPECT-style tabular output files.

This module provides a generic abstraction for ASPECT outputs that follow the
common format:

- A commented header line starting with '#', containing column names
- Whitespace-separated numeric data
- Potentially multiple timesteps stacked vertically

Example files include:
- depth average outputs
- statistics files
- material statistics
- velocity statistics
- custom postprocessor tables
"""

from pathlib import Path
import pandas as pd
import numpy as np


class AspectTable:
    """
    Generic reader for ASPECT-style table output files.

    The expected format is:
    - First commented line (# ...) contains column names
    - Data rows are whitespace-separated
    - Multiple timesteps may be stacked

    Attributes:
        path (Path): Path to the source file
        df (pandas.DataFrame): Parsed table data
    """

    def __init__(self, path):
        """
        Initialize and read the ASPECT table.

        Parameters:
            path : str or pathlib.Path
                Path to the ASPECT table output file.
        """
        self.path = Path(path)
        self.df = self._read()

    def _read(self):
        """
        Internal reader for ASPECT-style table files.

        Returns:
            pandas.DataFrame
                Parsed table with column names inferred from header.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        # --- Extract header ---
        header = None
        with self.path.open("r") as f:
            for line in f:
                if line.startswith("#"):
                    header = line.lstrip("#").strip().split()
                    break

        if header is None:
            raise ValueError(
                f"No header line starting with '#' found in file: {self.path}"
            )

        # --- Read data ---
        df = pd.read_csv(
            self.path,
            delim_whitespace=True,
            comment="#",
            names=header,
            header=None
        )

        # --- Basic validation ---
        if df.shape[1] != len(header):
            raise ValueError(
                f"Column mismatch in {self.path}: "
                f"header has {len(header)} columns, "
                f"but data has {df.shape[1]}"
            )

        return df

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def columns(self):
        """
        Return the list of column names.
        """
        return list(self.df.columns)

    def available_times(self):
        """
        Return sorted list of available timesteps.

        Requires a 'time' column to exist.
        """
        if "time" not in self.df.columns:
            raise ValueError("No 'time' column available in table")
        if self.df.shape[0] == 0:
            raise ValueError("Table contains no data")
        return sorted(self.df["time"].unique())


    def at_time(self, t, *, tol=1e-8):
        """
        Return subset of the table at a given time, using adaptive tolerance.

        Parameters:
            t : float or int
                Time value to filter by.
            tol : float, optional
                Tolerance used for comparison. If |t| < small_threshold, this is
                treated as an absolute tolerance (atol). Otherwise, it is treated
                as a relative tolerance (rtol).
            small_threshold : float, optional
                Threshold below which t is treated as "small".

        Returns:
            pandas.DataFrame
        """
        if "time" not in self.df.columns:
            raise ValueError("No 'time' column available in table")
        if self.df.shape[0] == 0:
            raise ValueError("Table contains no data")

        times = self.df["time"].to_numpy()

        if abs(t) < 1e-6:
            mask = np.isclose(times, t, atol=tol, rtol=0.0)
        else:
            mask = np.isclose(times, t, atol=0.0, rtol=tol)

        subset = self.df[mask]

        if subset.empty:
            raise ValueError(f"No rows found for time = {t}")

        return subset

    def nearest_time(self, t, *, return_time=False):
        """
        Return subset of the table at the time closest to the requested value.

        Parameters:
            t : float or int
                Target time value.
            return_time : bool, optional
                If True, return (actual_time, DataFrame) instead of just DataFrame.

        Returns:
            pandas.DataFrame
            or (float, pandas.DataFrame) if return_time=True
        """
        if "time" not in self.df.columns:
            raise ValueError("No 'time' column available in table")
        if self.df.shape[0] == 0:
            raise ValueError("Table contains no data")

        times = self.df["time"].to_numpy()

        if len(times) == 0:
            raise ValueError("Table contains no data")

        # Find nearest time value
        idx = np.argmin(np.abs(times - t))
        nearest = times[idx]

        subset = self.at_time(nearest)

        if return_time:
            return nearest, subset

        return subset

    def head(self, n=5):
        """
        Return the first n rows of the table.
        """
        return self.df.head(n)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(path={self.path}, "
            f"columns={len(self.df.columns)}, rows={len(self.df)})"
        )


class DepthAverageTable(AspectTable):
    """
    Specialized table for ASPECT depth-average output files.
    """

    REQUIRED_COLUMNS = {"time", "depth"}

    def __init__(self, path):
        super().__init__(path)
        self._validate()

    def _validate(self):
        """
        Validate that required columns exist for a depth-average file.
        """
        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns for DepthAverageTable: {missing}"
            )

    # ------------------------------------------------------------------
    # Domain-specific helpers
    # ------------------------------------------------------------------

    def profile(self, time, field, *, tol=1e-8):
        """
        Return a depth profile of a given field at a specific time.

        Parameters:
            time : float or int
                Target time.
            field : str
                Column name to extract (e.g., 'temperature', 'viscosity').
            tol : float, optional
                Tolerance passed to at_time.

        Returns:
            pandas.DataFrame
                DataFrame with columns ['depth', field]
        """
        if field not in self.df.columns:
            raise ValueError(f"Field '{field}' not found in table")

        df_t = self.at_time(time, tol=tol)

        return df_t[["depth", field]].copy()

    def available_fields(self):
        """
        Return available physical fields (excluding time and depth).
        """
        return [c for c in self.df.columns if c not in {"time", "depth"}]
