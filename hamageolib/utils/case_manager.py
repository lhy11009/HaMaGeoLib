"""CSV-backed numerical case registry, migrated from gdmate.aspect.case_manager."""

import pandas as pd
from pathlib import Path

class CaseManager:
    """
    Minimal manager for numerical cases using a CSV file.

    CSV format:
        case_id,case_path
    """

    def __init__(self, csv_file):
        self.csv_file = Path(csv_file)

        if self.csv_file.exists():
            print("CaseManager: read from existing csv file %s" % self.csv_file)
            self.df = pd.read_csv(self.csv_file)
        else:
            # initialize empty dataframe with required structure
            print("CaseManager: csv_file doesn't exist and we start from a blank object")
            self.df = pd.DataFrame(columns=["case_id", "case_path"])


        # normalize paths
        if not self.df.empty:
            self.df["case_path"] = self.df["case_path"].apply(
                lambda p: str(Path(p).resolve())
            )

        # ensure unique case IDs
        assert self.df["case_id"].is_unique, "case_id must be unique"

    class TabelRowError(Exception):
        '''
        Error class for case path
        '''
        pass

    def _get_row(self, mask, error_msg):
        """Internal helper returning (row_index, row)."""
        subset = self.df[mask]

        if subset.empty:
            raise self.TabelRowError(error_msg)

        idx = subset.index[0]
        row = subset.iloc[0]

        return idx, row

    def have_column(self, column_name):
        """
        Check whether the dataframe contains a given column.

        Parameters
        ----------
        column_name : str
            Name of the column

        Returns
        -------
        bool
            True if the column exists, False otherwise.
        """
        return column_name in self.df.columns

    def add_column(self, column_name, default_value=None):
        """
        Add a new column to the dataframe while ensuring `case_path`
        remains the last column.

        Parameters
        ----------
        column_name : str
            Name of the new column
        default_value : optional
            Default value assigned to the column
        """

        if column_name in self.df.columns:
            raise ValueError(f"Column '{column_name}' already exists")

        # find index of case_path column
        case_path_index = self.df.columns.get_loc("case_path")

        # insert new column before case_path
        self.df.insert(case_path_index, column_name, default_value)

    def get_case_by_id(self, case_id):
        """Return (row_index, row) for a given case ID."""
        return self._get_row(
            self.df["case_id"] == case_id,
            f"Case {case_id} not found in the table"
        )

    def get_case_by_path(self, case_path):
        """Return (row_index, row) for a given case path."""
        case_path = str(Path(case_path).resolve())

        return self._get_row(
            self.df["case_path"] == case_path,
            f"Case with path '{case_path}' not found"
        )

    def list_cases(self):
        """Return full dataframe."""
        return self.df

    def save(self):
        """Save the current dataframe to the CSV file."""
        self.df.to_csv(self.csv_file, index=False)
        print("CaseManager: csv file saved to %s" % self.csv_file)

    def next_case_id(self):
        """
        Return the next available case_id.
        """
        if self.df.empty:
            return 0
        return int(self.df["case_id"].max()) + 1

    def add_case(self, case_path, case_id=None):
        """
        Add a new case.

        Parameters
        ----------
        case_path : str or Path
            Absolute or relative path to the case directory
        case_id : int or None
            Optional case ID. If None, auto-generate.
        """
        case_path = str(Path(case_path).resolve())

        # ensure path not already registered
        if case_path in self.df["case_path"].values:
            raise ValueError(f"Case path already exists: {case_path}")

        if case_id is None:
            case_id = self.next_case_id()

        if case_id in self.df["case_id"].values:
            raise ValueError(f"case_id already exists: {case_id}")

        new_row = pd.DataFrame(
            {"case_id": [case_id], "case_path": [case_path]}
        )

        self.df = pd.concat([self.df, new_row], ignore_index=True)

        return self.get_case_by_id(case_id)

    def set_case_row(self, new_row):
        """
        Replace the row corresponding to case_id.

        Parameters
        ----------
        new_row : pandas.Series or dict
            Row data with the same columns as the dataframe
        """

        # use the case_id in the row
        case_id = new_row["case_id"]

        # get row index
        try:
            idx, _ = self.get_case_by_id(case_id)
        except self.TabelRowError as e:
            raise self.TabelRowError("Case with id %d doesn't exist in the table, please add that first." % case_id)

        # fix row data
        if isinstance(new_row, dict):
            new_row = pd.Series(new_row)

        # ensure all required columns exist
        missing_cols = set(self.df.columns) - set(new_row.index)
        if missing_cols:
            raise ValueError(f"Missing columns in new_row: {missing_cols}")

        # normalize case_path
        if "case_path" in new_row:
            new_row["case_path"] = str(Path(new_row["case_path"]).resolve())

        self.df.loc[idx] = new_row[self.df.columns]

    def remove_case(self, case_id):
        """
        Remove a case using case_id.
        """
        idx, _ = self.get_case_by_id(case_id)
        self.df = self.df.drop(idx).reset_index(drop=True)
