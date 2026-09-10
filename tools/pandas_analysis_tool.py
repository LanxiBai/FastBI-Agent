"""Execute validated aggregation requests without evaluating generated code."""

from __future__ import annotations

import json
import math
from typing import Any, Literal, NotRequired, TypedDict

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


class AnalysisRequest(TypedDict):
    """One aggregation, with optional grouping and AND-combined row filters."""

    group_by: str | None
    metric: str
    aggregation: Literal["sum", "mean", "count", "min", "max"]
    sort: NotRequired[Literal["asc", "desc"] | None]
    filters: NotRequired[list[dict[str, Any]]]


class PandasAnalysisError(ValueError):
    """The request cannot be safely executed against the dataset."""


def execute_analysis(
    dataframe: pd.DataFrame, request: AnalysisRequest | dict[str, Any]
) -> dict[str, Any]:
    """Aggregate real rows; count counts non-null metric values, not all rows.

    Filters run before aggregation. Missing group keys are retained; all-null
    sums remain null. Return up to 100 sorted groups with explicit truncation.
    """
    _validate_request(dataframe, request)
    filtered = dataframe
    for condition in request.get("filters", []):
        filtered = _filter_rows(filtered, condition)

    metric = request["metric"]
    group_by = request.get("group_by")
    aggregation = request["aggregation"]
    if filtered.empty:
        return {
            "status": "empty", "request": request, "matched_rows": 0,
            "total_groups": 0, "records": [], "truncated": False,
        }

    values = (
        filtered.groupby(group_by, dropna=False, observed=True, sort=False)[metric]
        if group_by is not None else filtered[metric]
    )
    # Only these explicitly selected pandas operations are executable.
    if aggregation == "sum":
        aggregated = values.sum(min_count=1)
    elif aggregation == "mean":
        aggregated = values.mean()
    elif aggregation == "count":
        aggregated = values.count()
    elif aggregation == "min":
        aggregated = values.min()
    else:
        aggregated = values.max()

    result = (
        aggregated.reset_index(name=metric)
        if group_by is not None else pd.DataFrame({metric: [aggregated]})
    )
    direction = request.get("sort")
    if direction is not None:
        result = result.sort_values(
            metric, ascending=direction == "asc", kind="stable", na_position="last"
        )
    return {
        "status": "success",
        "request": request,
        "matched_rows": int(len(filtered)),
        "total_groups": int(len(result)),
        "records": json.loads(
            result.head(100).to_json(orient="records", date_format="iso")
        ),
        "truncated": len(result) > 100,
    }


def _validate_request(dataframe: pd.DataFrame, request: Any) -> None:
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise PandasAnalysisError("The dataset contains no data rows.")
    if not dataframe.columns.is_unique:
        raise PandasAnalysisError("Duplicate column names are not supported.")
    allowed = {"group_by", "metric", "aggregation", "sort", "filters"}
    if not isinstance(request, dict) or set(request) - allowed:
        raise PandasAnalysisError("Only structured aggregation parameters are allowed.")
    metric = request.get("metric")
    group_by = request.get("group_by")
    for column in [metric] + ([group_by] if group_by is not None else []):
        if not isinstance(column, str) or column not in dataframe.columns:
            raise PandasAnalysisError(f"Column {column!r} does not exist.")
    if group_by == metric:
        raise PandasAnalysisError("Group and metric columns must be different.")
    aggregation = request.get("aggregation")
    if aggregation not in ("sum", "mean", "count", "min", "max"):
        raise PandasAnalysisError("Supported aggregations: sum, mean, count, min, max.")
    if aggregation != "count" and (
        not is_numeric_dtype(dataframe[metric]) or is_bool_dtype(dataframe[metric])
    ):
        raise PandasAnalysisError(f"Metric {metric!r} must be numeric for {aggregation}.")
    if request.get("sort") not in (None, "asc", "desc"):
        raise PandasAnalysisError("Sort must be asc, desc, or null.")
    filters = request.get("filters", [])
    if not isinstance(filters, list) or len(filters) > 10:
        raise PandasAnalysisError("Filters must be a list of at most 10 conditions.")
    for condition in filters:
        if not isinstance(condition, dict) or set(condition) != {"column", "op", "value"}:
            raise PandasAnalysisError("Each filter requires column, op, and value.")
        column = condition["column"]
        if not isinstance(column, str) or column not in dataframe.columns:
            raise PandasAnalysisError(f"Filter column {column!r} does not exist.")
        if condition["op"] not in ("eq", "ne", "gt", "gte", "lt", "lte"):
            raise PandasAnalysisError("Filter operator must be eq/ne/gt/gte/lt/lte.")
        value = condition["value"]
        if not isinstance(value, (str, int, float, bool)) or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            raise PandasAnalysisError("Filter values must be finite JSON scalars.")


def _filter_rows(dataframe: pd.DataFrame, condition: dict[str, Any]) -> pd.DataFrame:
    series = dataframe[condition["column"]]
    value = condition["value"]
    op = condition["op"]
    try:
        if op == "eq":
            mask = series.eq(value)
        elif op == "ne":
            mask = series.ne(value)
        elif op == "gt":
            mask = series.gt(value)
        elif op == "gte":
            mask = series.ge(value)
        elif op == "lt":
            mask = series.lt(value)
        else:
            mask = series.le(value)
        return dataframe.loc[mask.fillna(False) & series.notna()]
    except (TypeError, ValueError) as exc:
        raise PandasAnalysisError(f"Invalid filter on {condition['column']!r}: {exc}") from exc
