"""Streamlit interface for FastBI dataset analysis."""

import pandas as pd
import streamlit as st

from agents.analyst_agent import AgentError, AgentRunResult, run_agent
from tools.chart_tool import ChartToolError, generate_chart, suggest_chart
from tools.data_tool import DataSummary, DataToolError, analyze_data_file


def main() -> None:
    """Run the dataset summary, AI analysis, and manual chart interfaces."""
    st.title("FastBI-Agent")

    uploaded_file = st.file_uploader(
        "Upload an Excel or CSV file", type=["xlsx", "xls", "csv"]
    )
    user_question = st.text_input("Analysis question")

    dataframe: pd.DataFrame | None = None
    summary: DataSummary | None = None
    data_error: str | None = None
    if uploaded_file is not None:
        try:
            dataframe, summary = analyze_data_file(uploaded_file, uploaded_file.name)
            _show_dataset_summary(dataframe, summary)
        except DataToolError as exc:
            data_error = str(exc)
            st.error(data_error)

    if st.button("Analyze"):
        if uploaded_file is None:
            st.error("Please upload an Excel or CSV file before analysis.")
        elif not user_question.strip():
            st.error("Please enter an analysis question.")
        elif data_error or dataframe is None:
            st.error("The dataset could not be analyzed. Fix the upload error first.")
        else:
            try:
                with st.spinner("Analyzing the dataset..."):
                    result = run_agent(user_question, dataframe)
                _show_agent_result(result)
            except (AgentError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The analysis request failed: {exc}")

    if dataframe is not None and summary is not None:
        with st.expander("Manual chart test"):
            _show_chart_controls(dataframe, summary)


def _show_dataset_summary(dataframe: pd.DataFrame, summary: DataSummary) -> None:
    """Display a compact dataset overview."""
    st.subheader("Dataset Summary")
    first_metric, second_metric, third_metric = st.columns(3)
    first_metric.metric("Rows", summary["shape"]["rows"])
    second_metric.metric("Columns", summary["shape"]["columns"])
    third_metric.metric("Duplicate rows", summary["duplicates"])
    st.json(summary)
    st.caption("First 5 rows")
    st.dataframe(dataframe.head())


def _show_agent_result(result: AgentRunResult) -> None:
    """Display the LLM analysis and optional generated chart."""
    st.subheader("AI Analysis")
    st.markdown(result["final_response"])
    calculation = result["calculation_result"]
    if calculation["status"] == "success":
        st.caption("Calculated results from the uploaded dataset")
        st.dataframe(pd.DataFrame(calculation["records"]))
        if calculation["truncated"]:
            st.caption(f"Showing 100 of {calculation['total_groups']} groups.")
    elif calculation["status"] in ("empty", "unavailable"):
        st.info(calculation.get("reason", "No rows matched the requested filters."))

    st.subheader("Recommended Visualization")
    if result["chart_figure"] is not None:
        st.plotly_chart(result["chart_figure"], use_container_width=True)
    elif result["chart_result"]["status"] == "unavailable":
        st.info(result["chart_result"]["message"])
    else:
        st.caption("No chart was needed for this question.")


def _show_chart_controls(dataframe: pd.DataFrame, summary: DataSummary) -> None:
    """Render the existing manual chart selection controls."""
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
