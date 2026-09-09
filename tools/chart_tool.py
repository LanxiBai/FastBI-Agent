"""Generate basic Plotly charts from analyzed tabular data."""

from typing import Literal, TypedDict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tools.data_tool import add_date_from_parts, analyze_dataframe


ChartType = Literal["line", "bar", "histogram"]
SUPPORTED_CHART_TYPES: tuple[ChartType, ...] = ("line", "bar", "histogram")


class ChartToolError(ValueError):
    """Raised when a chart cannot be generated from the selected data."""


class ChartSuggestion(TypedDict):
    """A deterministic chart recommendation and its selected fields."""

    chart_type: ChartType
    x_column: str
    y_column: str | None
    reason: str


def suggest_chart(dataframe: pd.DataFrame) -> ChartSuggestion:
    """Recommend a chart using simple field-type rules."""
    dataframe = add_date_from_parts(dataframe)
    groups = _column_groups(dataframe)
    numeric_columns = groups["numeric"]
    date_columns = groups["date"]
    categorical_columns = groups["categorical"]

    if date_columns and numeric_columns:
        return {
            "chart_type": "line",
            "x_column": date_columns[0],
            "y_column": numeric_columns[0],
            "reason": "A date field and a numeric field are available.",
        }
    if categorical_columns and numeric_columns:
        return {
            "chart_type": "bar",
            "x_column": categorical_columns[0],
            "y_column": numeric_columns[0],
            "reason": "A categorical field and a numeric field are available.",
        }
    if numeric_columns:
        return {
            "chart_type": "histogram",
            "x_column": numeric_columns[0],
            "y_column": None,
            "reason": "A numeric field is available for a distribution chart.",
        }

    raise ChartToolError(
        "No suitable fields were found. A numeric column is required to create a chart."
    )


def generate_chart(
    dataframe: pd.DataFrame,
    chart_type: ChartType | str,
    x_column: str | None = None,
    y_column: str | None = None,
) -> go.Figure:
    """Generate a validated line, bar, or histogram Plotly figure."""
    if dataframe.empty:
        raise ChartToolError("Cannot generate a chart from an empty dataset.")
    if chart_type not in SUPPORTED_CHART_TYPES:
        supported = ", ".join(SUPPORTED_CHART_TYPES)
        raise ChartToolError(
            f"Unsupported chart type '{chart_type}'. Supported types: {supported}."
        )

    dataframe = add_date_from_parts(dataframe)
    groups = _column_groups(dataframe)

    if chart_type == "line":
        _require_column(dataframe, x_column, "x")
        _require_column(dataframe, y_column, "y")
        _require_type(x_column, groups["date"], "date/time", "line chart x-axis")
        _require_type(y_column, groups["numeric"], "numeric", "line chart y-axis")
        figure = _line_chart(dataframe, x_column, y_column)
    elif chart_type == "bar":
        _require_column(dataframe, x_column, "x")
        _require_column(dataframe, y_column, "y")
        _require_type(
            x_column, groups["categorical"], "categorical", "bar chart x-axis"
        )
        _require_type(y_column, groups["numeric"], "numeric", "bar chart y-axis")
        figure = _bar_chart(dataframe, x_column, y_column)
    else:
        _require_column(dataframe, x_column, "x")
        _require_type(x_column, groups["numeric"], "numeric", "histogram x-axis")
        figure = _histogram(dataframe, x_column)

    figure.update_layout(template="plotly_white")
    return figure


def _column_groups(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    if dataframe.empty:
        raise ChartToolError("Cannot analyze chart fields in an empty dataset.")

    summary = analyze_dataframe(dataframe)
    derived_dates = list(summary["derived_date_columns"])
    other_dates = [
        column for column in summary["date_columns"] if column not in derived_dates
    ]
    return {
        "numeric": list(summary["numeric_summary"]),
        "categorical": list(summary["categorical_summary"]),
        "date": derived_dates + other_dates,
    }


def _require_column(
    dataframe: pd.DataFrame, column: str | None, axis_name: str
) -> None:
    if not column:
        raise ChartToolError(f"A column must be selected for the {axis_name}-axis.")
    if column not in dataframe.columns:
        raise ChartToolError(f"Column '{column}' does not exist in the dataset.")


def _require_type(
    column: str | None,
    compatible_columns: list[str],
    expected_type: str,
    chart_role: str,
) -> None:
    if column not in compatible_columns:
        raise ChartToolError(
            f"Column '{column}' must be {expected_type} for the {chart_role}."
        )


def _line_chart(
    dataframe: pd.DataFrame, x_column: str, y_column: str
) -> go.Figure:
    chart_data = dataframe[[x_column, y_column]].dropna().copy()
    if chart_data.empty:
        raise ChartToolError(
            "The selected line chart fields contain no plottable rows."
        )

    chart_data[x_column] = pd.to_datetime(chart_data[x_column])
    chart_data = chart_data.sort_values(x_column)
    return px.line(
        chart_data,
        x=x_column,
        y=y_column,
        markers=True,
        title=f"{y_column} over {x_column}",
    )


def _bar_chart(
    dataframe: pd.DataFrame, x_column: str, y_column: str
) -> go.Figure:
    chart_data = dataframe[[x_column, y_column]].dropna()
    if chart_data.empty:
        raise ChartToolError("The selected bar chart fields contain no plottable rows.")

    figure = px.bar(
        chart_data,
        x=x_column,
        y=y_column,
        title=f"{y_column} by {x_column}",
    )
    figure.update_yaxes(rangemode="tozero")
    return figure


def _histogram(dataframe: pd.DataFrame, x_column: str) -> go.Figure:
    chart_data = dataframe[[x_column]].dropna()
    if chart_data.empty:
        raise ChartToolError("The selected histogram field contains no plottable rows.")

    return px.histogram(
        chart_data,
        x=x_column,
        title=f"{x_column} distribution",
    )
