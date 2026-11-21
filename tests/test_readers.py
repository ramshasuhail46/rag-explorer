"""Tests for readers.py: normal cases, edge cases, invalid inputs."""
import pandas as pd
import pytest

from rag_explorer.data.exceptions import DataLoadError
from rag_explorer.data.readers import CsvReader, DataReader, JsonLinesReader, reader_for


# ---------- normal cases ----------

def test_csv_reader_reads_all_rows(tiny_docs, tmp_path):
    path = tmp_path / "docs.csv"
    tiny_docs.to_csv(path, index=False)

    chunks = list(CsvReader().read_chunks(path))
    df = pd.concat(chunks)

    assert len(df) == 5
    assert list(df.columns) == ["id", "title", "abstract", "categories", "update_date"]


def test_json_reader_reads_all_rows(tiny_docs, tmp_path):
    path = tmp_path / "docs.json"
    tiny_docs.to_json(path, orient="records", lines=True)

    df = pd.concat(JsonLinesReader().read_chunks(path))

    assert len(df) == 5
    assert pd.api.types.is_string_dtype(df["id"]) 


def test_reader_drops_extra_columns(tiny_docs, tmp_path):
    path = tmp_path / "docs.csv"
    tiny_docs.assign(authors="someone").to_csv(path, index=False)

    df = pd.concat(CsvReader().read_chunks(path))

    assert "authors" not in df.columns


def test_reader_splits_into_chunks(tiny_docs, tmp_path):
    path = tmp_path / "docs.csv"
    tiny_docs.to_csv(path, index=False)

    chunks = list(CsvReader(chunk_size=2).read_chunks(path))

    assert [len(c) for c in chunks] == [2, 2, 1]


@pytest.mark.parametrize(
    "filename, expected",
    [("a.json", JsonLinesReader), ("a.jsonl", JsonLinesReader),
     ("a.csv", CsvReader), ("A.CSV", CsvReader)],
)
def test_reader_for_picks_correct_reader(filename, expected):
    assert isinstance(reader_for(filename), expected)


# ---------- invalid inputs ----------

def test_missing_column_raises(tiny_docs, tmp_path):
    path = tmp_path / "docs.csv"
    tiny_docs.drop(columns=["abstract"]).to_csv(path, index=False)

    with pytest.raises(DataLoadError, match="abstract"):
        list(CsvReader().read_chunks(path))


def test_unsupported_extension_raises():
    with pytest.raises(DataLoadError, match="Unsupported"):
        reader_for("data.xlsx")


@pytest.mark.parametrize("size", [0, -5])
def test_invalid_chunk_size_raises(size):
    with pytest.raises(ValueError):
        CsvReader(chunk_size=size)


def test_base_class_cannot_be_created():
    with pytest.raises(TypeError):
        DataReader()