"""Readers: turn a file into Dataframe chunks"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from rag_explorer.data.exceptions import DataLoadError

DEFAULT_COLUMNS = ["id", "title", "abstract", "categories", "update_date"]

class DataReader(ABC):
    """Yield the file's rows as DataFrame chunks"""

    def __init__(self, columns: list[str] = DEFAULT_COLUMNS, chunk_size: int = 50_000):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.columns = columns
        self.chunk_size = chunk_size

    @abstractmethod
    def read_chunks(self, path: Path) -> Iterator[pd.DataFrame]:
        """Yield DataFrame chunks from the file at the given path"""

    def _select_columns(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Select only the specified columns from the DataFrame chunk"""
        missing = set(self.columns) - set(chunk.columns)

        if missing:
            raise DataLoadError(f"Missing columns in the data: {missing}")

        return chunk[self.columns]


class JsonLinesReader(DataReader):
    """Reads JSON Lines files (one JSON object per line), e.g. the arXiv dump."""

    def read_chunks(self, path: Path) -> Iterator[pd.DataFrame]:
        with pd.read_json(path, lines=True, chunksize=self.chunk_size,
                          dtype={"id": str}) as reader:
            for chunk in reader:
                yield self._select_columns(chunk)


class CsvReader(DataReader):
    """Reads CSV files, e.g. a previously saved processed dataset."""

    def read_chunks(self, path: Path) -> Iterator[pd.DataFrame]:
        for chunk in pd.read_csv(path, chunksize=self.chunk_size, dtype={"id": str}):
            yield self._select_columns(chunk)


READERS: dict[str, type[DataReader]] = {
    ".json": JsonLinesReader,
    ".jsonl": JsonLinesReader,
    ".csv": CsvReader,
}

def reader_for(path: str | Path) -> DataReader:
    """Return the right reader based on the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix not in READERS:
        raise DataLoadError(f"Unsupported file type: {suffix}")
    return READERS[suffix]()