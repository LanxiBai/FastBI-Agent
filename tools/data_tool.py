"""Read Excel/CSV datasets and produce a reusable EDA summary."""

from __future__ import annotations

from datetime import date, datetime
from os import PathLike
from pathlib import Path
from typing import IO, Any, TypeAlias

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


FileSource: TypeAlias = str | PathLike[str] | IO[bytes] | IO[str]
DataSummary: TypeAlias = dict[str, Any]
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class DataToolError(ValueError):
    """Raised when a dataset cannot be read or analyzed."""


def load_data(source: FileSource, filename: str | None = None) -> pd.DataFrame:
    """Load a CSV or Excel source based on its filename extension."""
    extension = _get_extension(source, filename)
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DataToolError(
            f"Unsupported file type '{extension or '(none)'}'. Supported types: {supported}."
        )

    _ensure_source_is_not_empty(source)

    if extension == ".csv":
        dataframe = _read_csv(source)
    else:
        dataframe = _read_excel(source, extension)

    if dataframe.empty:
        raise DataToolError("The uploaded dataset contains no data rows.")

    return dataframe


def analyze_dataframe(dataframe: pd.DataFrame) -> DataSummary:
    """Return structure, quality, and basic statistics for a DataFrame."""
    if dataframe.empty:
        raise DataToolError("The dataset contains no data rows.")

    column_names = [str(column) for column in dataframe.columns]
    date_columns = _detect_date_columns(dataframe)
    numeric_columns = [
        column
        for column in dataframe.columns
        if is_numeric_dtype(dataframe[column].dtype)
        and not is_bool_dtype(dataframe[column].dtype)
    ]
    categorical_columns = [
        column
        for column in dataframe.columns
        if _is_categorical(dataframe[column]) and str(column) not in date_columns
    ]

    row_count, column_count = dataframe.shape
    missing = {
        str(column): {
            "count": int(dataframe[column].isna().sum()),
            "rate": float(dataframe[column].isna().mean()),
        }
        for column in dataframe.columns
    }

    numeric_summary = {
        str(column): _numeric_statistics(dataframe[column])
        for column in numeric_columns
    }
    categorical_summary = {
        str(column): _categorical_statistics(dataframe[column])
        for column in categorical_columns
    }

    return {
        "shape": {"rows": int(row_count), "columns": int(column_count)},
        "columns": column_names,
        "dtypes": {
            str(column): str(dataframe[column].dtype) for column in dataframe.columns
        },
        "missing": missing,
        "duplicates": int(dataframe.duplicated().sum()),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "date_columns": date_columns,
    }


def analyze_data_file(
    source: FileSource, filename: str | None = None
) -> tuple[pd.DataFrame, DataSummary]:
    """Load a dataset and return both its DataFrame and EDA summary."""
    dataframe = load_data(source, filename)
    return dataframe, analyze_dataframe(dataframe)


def _get_extension(source: FileSource, filename: str | None) -> str:
    source_name = filename or getattr(source, "name", None)
    if source_name is None and isinstance(source, (str, PathLike)):
        source_name = str(source)
    return Path(str(source_name)).suffix.lower() if source_name else ""


def _ensure_source_is_not_empty(source: FileSource) -> None:
    if isinstance(source, (str, PathLike)):
        path = Path(source)
        if not path.exists():
            raise DataToolError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise DataToolError("The uploaded file is empty.")
        return

    getbuffer = getattr(source, "getbuffer", None)
    if callable(getbuffer) and getbuffer().nbytes == 0:
        raise DataToolError("The uploaded file is empty.")


def _rewind(source: FileSource) -> None:
    seek = getattr(source, "seek", None)
    if callable(seek):
        seek(0)


def _read_csv(source: FileSource) -> pd.DataFrame:
    decoding_errors: list[str] = []
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            _rewind(source)
            return pd.read_csv(source, encoding=encoding)
        except UnicodeError as exc:
            decoding_errors.append(str(exc))
        except pd.errors.EmptyDataError as exc:
            raise DataToolError("The CSV file is empty or has no readable columns.") from exc
        except pd.errors.ParserError as exc:
            raise DataToolError(f"The CSV file could not be parsed: {exc}") from exc
        except OSError as exc:
            raise DataToolError(f"The CSV file could not be read: {exc}") from exc

    detail = decoding_errors[-1] if decoding_errors else "unknown encoding error"
    raise DataToolError(
        "The CSV encoding is not supported. Tried UTF-8 and GB18030. "
        f"Last error: {detail}"
    )


def _read_excel(source: FileSource, extension: str) -> pd.DataFrame:
    try:
        _rewind(source)
        return pd.read_excel(source)
    except ImportError as exc:
        dependency = "xlrd" if extension == ".xls" else "openpyxl"
        raise DataToolError(
            f"Reading {extension} files requires the optional '{dependency}' dependency."
        ) from exc
    except Exception as exc:
        raise DataToolError(f"The Excel file could not be read: {exc}") from exc


def _numeric_statistics(series: pd.Series) -> dict[str, int | float | None]:
    return {
        "count": int(series.count()),
        "mean": _to_number_or_none(series.mean()),
        "std": _to_number_or_none(series.std()),
        "min": _to_number_or_none(series.min()),
        "median": _to_number_or_none(series.median()),
        "max": _to_number_or_none(series.max()),
    }


def _categorical_statistics(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    top_value = non_null.value_counts().index[0] if not non_null.empty else None
    return {
        "unique": int(series.nunique(dropna=True)),
        "top": _to_python_value(top_value),
    }


def _is_categorical(series: pd.Series) -> bool:
    dtype = series.dtype
    return bool(
        is_object_dtype(dtype)
        or is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or is_bool_dtype(dtype)
    )


def _detect_date_columns(dataframe: pd.DataFrame) -> list[str]:
    date_columns: list[str] = []
    safe_string_formats = (
        (r"\d{4}-\d{1,2}-\d{1,2}", "%Y-%m-%d"),
        (r"\d{4}/\d{1,2}/\d{1,2}", "%Y/%m/%d"),
    )

    for column in dataframe.columns:
        series = dataframe[column]
        if is_datetime64_any_dtype(series.dtype):
            date_columns.append(str(column))
            continue
        if not (is_object_dtype(series.dtype) or is_string_dtype(series.dtype)):
            continue

        values = series.dropna().astype("string").str.strip()
        if values.empty:
            continue

        for pattern, date_format in safe_string_formats:
            if values.str.fullmatch(pattern).all():
                parsed = pd.to_datetime(values, format=date_format, errors="coerce")
                if parsed.notna().all():
                    date_columns.append(str(column))
                    break

    return date_columns


def _to_number_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _to_python_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    item = getattr(value, "item", None)
    value = item() if callable(item) else value
    return value if isinstance(value, (str, int, float, bool)) else str(value)
