"""Minimal Streamlit entry point for the FastBI analyst agent."""

import streamlit as st

from agents.analyst_agent import run_agent
from tools.data_tool import DataToolError, analyze_data_file


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


if __name__ == "__main__":
    main()
