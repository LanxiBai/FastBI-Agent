"""Minimal Streamlit entry point for the FastBI analyst agent."""

import pandas as pd
import streamlit as st

from agents.analyst_agent import run_agent
from tools.chart_tool import ChartToolError, generate_chart, suggest_chart
from tools.data_tool import DataSummary, DataToolError, analyze_data_file


def main() -> None:
    """Show minimal dataset analysis and agent test interfaces."""
    st.title("FastBI-Agent")

    st.subheader("Dataset analysis")
    uploaded_file = st.file_uploader(
        "Upload an Excel or CSV file", type=["xlsx", "xls", "csv"]
    )
    if uploaded_file is not None:
        try:
            dataframe, summary = analyze_data_file(uploaded_file, uploaded_file.name)
            first_metric, second_metric, third_metric = st.columns(3)
            first_metric.metric("Rows", summary["shape"]["rows"])
            second_metric.metric("Columns", summary["shape"]["columns"])
            third_metric.metric("Duplicate rows", summary["duplicates"])
            st.json(summary)
            st.caption("First 5 rows")
            st.dataframe(dataframe.head())
            _show_chart_controls(dataframe, summary)
        except DataToolError as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Agent test")
    user_input = st.text_input("Ask the analyst a question")

    if st.button("Run"):
        if not user_input.strip():
            st.warning("Please enter a question.")
            return

        try:
            with st.spinner("Thinking..."):
                st.write(run_agent(user_input))
        except ValueError as exc:
            st.error(str(exc))


def _show_chart_controls(dataframe: pd.DataFrame, summary: DataSummary) -> None:
    """Render minimal chart selection controls for an uploaded dataset."""
    st.subheader("Data visualization")
    try:
        suggestion = suggest_chart(dataframe)
        st.caption(f"Recommended: {suggestion['chart_type']}. {suggestion['reason']}")
    except ChartToolError as exc:
        suggestion = None
        st.info(str(exc))

    chart_types = ["line", "bar", "histogram"]
    suggested_type = suggestion["chart_type"] if suggestion else "histogram"
    chart_type = st.selectbox(
        "Chart type", chart_types, index=chart_types.index(suggested_type)
    )

    numeric_columns = list(summary["numeric_summary"])
    if chart_type == "line":
        x_options = list(summary["date_columns"])
        y_options = numeric_columns
    elif chart_type == "bar":
        x_options = list(summary["categorical_summary"])
        y_options = numeric_columns
    else:
        x_options = numeric_columns
        y_options = []

    x_column = _column_selectbox(
        "X column", x_options, suggestion["x_column"] if suggestion else None, "chart_x"
    )
    y_column = None
    if chart_type != "histogram":
        y_column = _column_selectbox(
            "Y column",
            y_options,
            suggestion["y_column"] if suggestion else None,
            "chart_y",
        )

    fields_available = bool(x_column) and (chart_type == "histogram" or bool(y_column))
    if st.button("Generate chart", disabled=not fields_available):
        try:
            figure = generate_chart(dataframe, chart_type, x_column, y_column)
            st.plotly_chart(figure, use_container_width=True)
        except ChartToolError as exc:
            st.error(str(exc))


def _column_selectbox(
    label: str,
    options: list[str],
    suggested_column: str | None,
    key: str,
) -> str | None:
    """Show a column selector and prefer the suggested field when available."""
    if not options:
        st.info(f"No compatible fields are available for {label.lower()}.")
        return None
    index = options.index(suggested_column) if suggested_column in options else 0
    return st.selectbox(label, options, index=index, key=key)


if __name__ == "__main__":
    main()
