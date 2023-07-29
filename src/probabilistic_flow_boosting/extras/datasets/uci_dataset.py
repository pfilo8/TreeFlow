from typing import Any, Dict

import numpy as np
import pandas as pd

from kedro.io import AbstractDataSet


class UCIDataSet(AbstractDataSet):

    def __init__(self, filepath_data: str, filepath_index_columns: str, filepath_index_rows: str,
                 load_args: Dict[str, Any] = None):
        """Creates a new instance of UCIDataSet to load / save image data at the given filepath.

        Args:
            filepath_data: The location of the file with data.
            filepath_index_columns: The location of the file with columns.
            filepath_index_rows: The location of the file with rows.
        """
        self._filepath_data = filepath_data
        self._filepath_index_columns = filepath_index_columns
        self._filepath_index_rows = filepath_index_rows
        self._load_args = load_args if load_args is not None else {}

    def _load(self) -> np.ndarray:
        data = np.loadtxt(self._filepath_data, delimiter=self._load_args.get("data_delimiter"))
        data = pd.DataFrame(data)
        index_columns = np.loadtxt(self._filepath_index_columns, dtype=np.int32,
                                   delimiter=self._load_args.get("index_columns_delimiter"))
        index_columns = index_columns.reshape(-1)  # Issues when loading target column
        index_rows = np.loadtxt(self._filepath_index_rows, dtype=np.int32,
                                delimiter=self._load_args.get("index_rows_delimiter"))
        return data.iloc[index_rows, index_columns]

    def _save(self, data: np.ndarray) -> None:
        raise ValueError("Saving not supported.")

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset"""
        return dict(
            filepath_data=self._filepath_data,
            filepath_index_columns=self._filepath_index_columns,
            filepath_index_rows=self._filepath_index_rows
        )
